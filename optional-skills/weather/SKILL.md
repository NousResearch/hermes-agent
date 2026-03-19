---
name: weather
description: Real-time weather and forecasts for any city worldwide — current conditions, hourly/daily forecasts, multi-location compare, and severe weather alerts. Uses Open-Meteo (free, no API key required).
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [weather, forecast, temperature, rain, wind, alerts, open-meteo]
    category: utilities
    requires_toolsets: [terminal]
---

# Weather Skill

Real-time weather data and forecasts for any location worldwide.
5 commands: current conditions, hourly forecast, daily summary, multi-city compare, alerts.

Free, no API key. Uses Open-Meteo + OpenStreetMap geocoding. Zero dependencies.

---

## When to Use
- User asks about current weather in any city
- User wants a weather forecast (today, this week)
- User wants to compare weather across multiple cities
- User asks if there are weather alerts (frost, heat wave, storm, strong wind)

---

## Prerequisites
Python 3.8+ stdlib only. No pip installs.
Script path: `~/.hermes/skills/weather/scripts/weather_client.py`

---

## Quick Reference

```
SCRIPT=~/.hermes/skills/weather/scripts/weather_client.py
python3 $SCRIPT now "Paris"
python3 $SCRIPT forecast "London" --days 3
python3 $SCRIPT daily "Tokyo" --days 7
python3 $SCRIPT compare "Paris" "London" "Berlin"
python3 $SCRIPT alerts "Miami"
```

---

## Commands

### now LOCATION
Current temperature, feels like, humidity, wind, weather description, precipitation.

### forecast LOCATION [--days N]
Hourly forecast, default 3 days, max 7.

### daily LOCATION [--days N]
Daily summary with temp max/min, precipitation, sunrise/sunset. Default 7 days.

### compare LOCATION1 LOCATION2 [...]
Side-by-side current weather for multiple cities.

### alerts LOCATION
Detect severe conditions: frost (<2C), heat (>35C), strong wind (>60 km/h), heavy rain (>10mm/h).

---

## Pitfalls
- Use full city names for best geocoding results.
- Temperatures in Celsius, wind in km/h.

---

## Verification
```bash
python3 ~/.hermes/skills/weather/scripts/weather_client.py now "Paris"
```
