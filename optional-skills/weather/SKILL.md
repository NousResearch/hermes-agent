---
name: weather
description: Real-time weather and forecasts for any city worldwide — current conditions, hourly/daily forecasts, multi-location compare, and severe weather alerts. Uses Open-Meteo (free, no API key required).
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [weather, forecast, temperature, rain, wind, alerts, climate, open-meteo]
    category: utilities
    requires_toolsets: [terminal]
---

# Weather Skill

Real-time weather data and forecasts for any location worldwide.
5 commands: current conditions, hourly forecast, daily summary, multi-city compare, and alerts.

Free, no API key. Uses Open-Meteo + OpenStreetMap geocoding. Zero dependencies.

---

## When to Use
- User asks about current weather in any city
- User wants a weather forecast (today, this week)
- User wants to compare weather across multiple cities
- User asks if there are weather alerts (frost, heat wave, storm, strong wind)
- User is planning travel and wants to know what weather to expect

---

## Prerequisites
Python 3.8+ stdlib only. No pip installs.

Script path: `~/.hermes/skills/weather/scripts/weather_client.py`

---

## Quick Reference

```
SCRIPT=~/.hermes/skills/weather/scripts/weather_client.py

python3 $SCRIPT now "Paris"
python3 $SCRIPT now "New York"
python3 $SCRIPT forecast "London" --days 3
python3 $SCRIPT daily "Tokyo" --days 7
python3 $SCRIPT compare "Paris" "London" "Berlin"
python3 $SCRIPT alerts "Miami"
```

---

## Commands

### now LOCATION
Current temperature, feels like, humidity, wind, weather description, precipitation.
```bash
python3 $SCRIPT now "Marseille"
python3 $SCRIPT now "San Francisco"
```

### forecast LOCATION [--days N]
Hourly forecast. Default 3 days, max 7.
```bash
python3 $SCRIPT forecast "Barcelona" --days 5
```
Output per hour: time, temperature, precipitation_probability, weather description, wind_speed.

### daily LOCATION [--days N]
Daily summary. Default 7 days.
```bash
python3 $SCRIPT daily "Rome" --days 7
```
Output per day: date, temp_max, temp_min, precipitation_sum, sunrise, sunset, weather description.

### compare LOCATION1 LOCATION2 [...]
Side-by-side current weather for multiple cities.
```bash
python3 $SCRIPT compare "Paris" "London" "Amsterdam" "Berlin"
```

### alerts LOCATION
Check for severe weather: frost (<2C), heat (>35C), strong wind (>60 km/h), heavy rain (>10mm/h).
```bash
python3 $SCRIPT alerts "New Orleans"
```
Returns active alerts with severity level.

---

## Pitfalls
- Location names must be recognizable by OpenStreetMap geocoding. Use full city names.
- Forecasts are in local timezone of the location.
- Temperatures in Celsius, wind in km/h.

---

## Verification
```bash
python3 ~/.hermes/skills/weather/scripts/weather_client.py now "Paris"
# Should return temperature, humidity, wind, weather description
```
