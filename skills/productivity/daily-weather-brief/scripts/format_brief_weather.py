#!/usr/bin/env python3
"""
Daily-brief weather formatter — mirrors GordonClaw's format_brief_weather.py.
Reads locations from skills/productivity/daily-weather-brief/locations.json,
fetches today + tomorrow + 3-day outlook from Open-Meteo (no API key), emits
canonical Markdown for the weather section.
"""

import json, os, socket, sys, time, urllib.parse, urllib.request, urllib.error
from datetime import datetime

DEFAULT_LOCATIONS_JSON = "/opt/data/skills/productivity/daily-weather-brief/locations.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → short human strings
WMO = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm w/ hail", 99: "thunderstorm w/ hail",
}

def _load_locations(path):
    with open(path) as f:
        data = json.load(f)
    out = []
    for loc in data.get("locations", []):
        try:
            out.append({"name": loc["name"], "lat": float(loc["lat"]), "lon": float(loc["lon"])})
        except (KeyError, TypeError, ValueError):
            continue
    return out

def _fetch(lat, lon):
    params = {
        "latitude": f"{lat}", "longitude": f"{lon}",
        "daily": "temperature_2m_max,temperature_2m_min,weathercode,windspeed_10m_max",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "auto", "forecast_days": "5",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, socket.timeout, TimeoutError):
            if attempt == 1:
                time.sleep(1.5)
                continue
            raise

def _describe(code):
    return WMO.get(int(code), f"code {int(code)}") if code is not None else "—"

def _format_day(label, idx, daily):
    hi = int(round(float(daily["temperature_2m_max"][idx])))
    lo = int(round(float(daily["temperature_2m_min"][idx])))
    code = daily["weathercode"][idx]
    wind_val = float(daily["windspeed_10m_max"][idx])
    wind_desc = "breezy" if 10 <= wind_val < 20 else ("windy" if wind_val >= 20 else "")
    parts = [f"{hi}°/{lo}°", _describe(code)]
    if wind_desc:
        parts.append(wind_desc)
    return f"• {label}: " + ", ".join(parts)

def _summarize_next_three(daily):
    # Temperature trend
    try:
        today_max = daily["temperature_2m_max"][0]
        future_maxes = [daily["temperature_2m_max"][i] for i in range(2, 5)]
        avg = sum(future_maxes) / len(future_maxes)
        temp_trend = "warming" if avg - today_max > 5 else ("cooling" if today_max - avg > 5 else "stable")
    except Exception:
        temp_trend = ""
    # Precip: wet if any code in ranges 61-67, 71-87, 95/96/99
    precip_codes = set(range(61, 68)) | set(range(71, 88)) | {95, 96, 99}
    wet_days = []
    for i in range(2, 5):
        if daily["weathercode"][i] in precip_codes:
            wet_days.append(datetime.fromisoformat(daily["time"][i]).strftime("%a"))
    precip_trend = "wet on " + ", ".join(wet_days) if wet_days else "dry"
    return temp_trend + ", " + precip_trend if temp_trend else precip_trend

def _format_location(loc, payload):
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    lines = [f"*{loc['name']}*", _format_day("Today", 0, daily)]
    if len(times) > 1:
        lines.append(_format_day("Tmrw", 1, daily))
    summary = _summarize_next_three(daily)
    if len(times) >= 5:
        start = datetime.fromisoformat(times[2]).strftime("%a")
        end = datetime.fromisoformat(times[4]).strftime("%a")
        lines.append(f"• {start}–{end}: {summary}")
    else:
        lines.append(f"• {summary}")
    return "\n".join(lines)

def main():
    locations = _load_locations(os.environ.get("DAILY_BRIEF_LOCATIONS_JSON", DEFAULT_LOCATIONS_JSON))
    blocks = ["*Weather*"]
    for loc in locations:
        try:
            payload = _fetch(loc["lat"], loc["lon"])
            blocks.append(_format_location(loc, payload))
        except Exception as e:
            blocks.append(f"*{loc['name']}*\n• (error: {e})")
    print("\n\n".join(blocks))

if __name__ == "__main__":
    main()
