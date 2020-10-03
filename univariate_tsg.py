#import packages
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas.plotting import register_matplotlib_converters
import numpy as np
from numpy import array

# LSTM libraries
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from sklearn.metrics import mean_squared_error

#to plot within notebook
import plotly.offline as pyoff
import plotly.graph_objs as go

#for normalizing data
from sklearn.preprocessing import MinMaxScaler

def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')

# Read the file
col_list = ["Date", "Close"]
df = pd.read_csv('ENB.csv', parse_dates=['Date'], index_col=0, date_parser=parser, usecols=col_list)

train = df

scaler = MinMaxScaler()
scaler.fit(train)
data = scaler.transform(train)  
target = scaler.transform(train)

# Define generator
n_input = 60
n_features = 1
# batch_size = int(len(data)/5)
batch_size = 64
generator = TimeseriesGenerator(data, target, length=n_input, batch_size=batch_size)

# Define model
model = Sequential()
model.add(LSTM(300, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.15))
model.add(Dense(1))

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse')

# Fit model
history = model.fit(generator, steps_per_epoch=len(generator), epochs=20, verbose=1)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plot_data = [
    go.Scatter(
        x=hist['epoch'],
        y=hist['loss'],
        name='loss'
    )
]

plot_layout = go.Layout(
        title='Training loss'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.write_image("tsg_close_training_loss.png")

pred_list = []

batch = data[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0, n_input+1) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

df_predict = pd.DataFrame(scaler.inverse_transform(pred_list), index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pd.concat([df,df_predict], axis=1)

plot_data = [
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Close'],
        name='actual'
    ),
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Prediction'],
        name='prediction'
    )
]

plot_layout = go.Layout(
        title='Valor de cierre'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.write_image("tsg_close_value.png")