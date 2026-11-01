# Weather Query

Get current and forecast weather data for a specified latitude/longitude location.

## Request

```
GET /v1.0/end-user/services/weather/recent
```

## Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| codes | String | Yes | JSON array string specifying which weather attributes to query. E.g. `["w.temp","w.humidity","w.condition","w.hour.7"]` |
| lat | String | Yes | Latitude |
| lon | String | Yes | Longitude |

## Request Example

```
GET /v1.0/end-user/services/weather/recent?codes=["w.temp","w.humidity","w.condition","w.hour.7"]&lat=39.9042&lon=116.4074
```

> When constructing URLs manually, the `codes` value must be URL-encoded (e.g. `%5B%22w.temp%22%5D`). The Python SDK handles encoding automatically.

## Response Example

```json
{
  "success": true,
  "result": {
    "data": {
      "w.condition.0": "mixed precipitation",
      "w.condition.1": "mixed precipitation",
      "w.temp.0": 7,
      "w.temp.1": 6,
      "w.temp.2": 6,
      "w.humidity.0": 44,
      "w.humidity.1": 46,
      "w.humidity.2": 48
    },
    "expiration": 15
  }
}
```

## Response Field Description

Response data keys follow the format `{attribute}.{time_index}`:

- `w.temp.0` — Current temperature; `w.temp.1` — Temperature in 1 hour; and so on
- `w.humidity.0` — Current humidity
- `w.condition.0` — Current weather condition (English description)
- `expiration` — Cache expiration time (minutes)

The time granularity is determined by `w.hour.N` in the `codes` parameter. For example, `w.hour.7` returns data for the next 7 hourly time points.

## Available Attribute Codes

Supported attribute codes for this API:

| Code | Description |
|------|-------------|
| w.temp | Temperature |
| w.humidity | Humidity |
| w.condition | Weather condition description (English) |
| w.conditionNum | Weather condition numeric code |
| w.pressure | Atmospheric pressure |
| w.realFeel | Feels-like temperature |
| w.uvi | UV index |
| w.windDir | Wind direction |
| w.windLevel | Wind level |
| w.windSpeed | Wind speed |
| w.hour.N | Time granularity (N = hours, e.g. `w.hour.7` = next 7 hours) |

> Full supported list: condition, conditionNum, dateTime, humidity, mark, pressure, realFeel, sunRiseTimeStamp, sunSetTimeStamp, temp, uvi, windDir, windLevel, windSpeed

## Weather Condition Mapping

| English Description | Chinese | Category |
|--------------------|---------|----------|
| no precipitation | Sunny (晴) | Sunny |
| clear | Clear (晴朗) | Clear |
| partly cloudy | Partly Cloudy (少云) | Cloud |
| mostly cloudy | Mostly Cloudy (多云) | Cloud |
| mixed precipitation | Mostly Sunny (大部晴朗) | Cloud |
| overcast | Overcast (阴) | Cloud |
| foggy | Foggy (雾) | Fog |
| haze fog | Haze (霾) | Cloud |
| drizzle / light rain | Light Rain (小雨) | Rainy |
| precipitation | Rain (雨) | Rainy |
| rain | Moderate Rain (中雨) | Rainy |
| heavy rain | Heavy Rain (大雨) | Rainy |
| heavy precipitation | Rainstorm (暴雨) | Rainy |
| rainy big | Heavy Rainstorm (大暴雨) | Rainy |
| very heavy snow | Extreme Rainstorm (特大暴雨) | Rainy |
| shower | Shower (阵雨) | Rainy |
| shower with thunder | Thundershower (雷阵雨) | Rainy |
| shower with hail | Thundershower with Hail (雷阵雨伴有冰雹) | Snow |
| thunderstorms | Thunderstorm (雷暴) | Rainy |
| light snow | Light Snow (小雪) | Snow |
| snow | Snow / Moderate Snow (雪/中雪) | Snow |
| heavy snow | Heavy Snow (大雪) | Snow |
| blizzard | Blizzard (暴雪) | Snow |
| flurries | Snow Showers (阵雪) | Rainy |
| possible flurries | Light Snow Showers (小阵雪) | Snow |
| light sleet | Sleet (雨夹雪) | Snow |
| ice rain | Freezing Rain (冻雨) | Rainy |
| frozen fog | Frozen Fog (冻雾) | Snow |
| ice particle | Ice Pellets (冰粒) | Snow |
| heavy sleet | Needle Ice (冰针) | Snow |
| sleet | Hail (冰雹) | Snow |
| blowing sand | Blowing Sand (扬沙) | Sunny |
| float dust | Floating Dust (浮尘) | Sunny |
| dangerously windy | Sandstorm (沙尘暴) | Sunny |
| strong sand storm | Severe Sandstorm (强沙尘暴) | Rainy |
| dust devil | Dust Devil (尘卷风) | Sunny |

## Common City Coordinates

This API requires latitude and longitude. Here are reference coordinates for common cities:

| City | Latitude (lat) | Longitude (lon) |
|------|----------------|-----------------|
| Beijing | 39.9042 | 116.4074 |
| Shanghai | 31.2304 | 121.4737 |
| Guangzhou | 23.1291 | 113.2644 |
| Shenzhen | 22.5431 | 114.0579 |
| Hangzhou | 30.2741 | 120.1551 |
| Chengdu | 30.5728 | 104.0668 |
| Wuhan | 30.5928 | 114.3055 |
| Nanjing | 32.0603 | 118.7969 |
| Chongqing | 29.5630 | 106.5516 |
| Xi'an | 34.3416 | 108.9398 |

> If the user does not provide coordinates, ask for their city and use the corresponding latitude/longitude to call the API.
