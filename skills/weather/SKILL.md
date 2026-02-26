# Weather Skill

Get real-time weather data for any city using OpenWeatherMap.

## Setup

1. Get a free API key at https://openweathermap.org/api
2. Add to `~/.hermes/.env`:
```
   OPENWEATHERMAP_API_KEY=your_key_here
```

## Tools

### `weather_current` - Current conditions
```
What is the weather in Istanbul?
What is the temperature in Tokyo right now?
Is it raining in London?
```

### `weather_forecast` - Multi-day forecast (up to 5 days)
```
What is the weather forecast for Paris this week?
Will it rain in New York tomorrow?
Give me a 3-day forecast for Berlin
```

### `weather_alerts` - Active alerts + Air Quality Index
```
Are there any weather alerts for Miami?
What is the air quality in Beijing?
Any severe weather warnings for Chicago?
```

## Notes

- Add country code for accuracy: "London,GB" or "Springfield,US"
- Units default to metric (C). Pass units="imperial" for F.
- AQI scale: 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
