#!/usr/bin/env python3
"""Current weather and forecasts via Open-Meteo (no API key required).

Pure standard library: urllib, json, argparse. Geocodes a city name with the
Open-Meteo geocoding API, then fetches current conditions plus an hourly and
daily forecast from the Open-Meteo forecast API.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10  # seconds

# WMO weather interpretation codes (WW) as documented at
# https://open-meteo.com/en/docs#weather_variable_documentation
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

COMPASS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def wmo_description(code) -> str:
    """Map a WMO weather code to a human-readable description."""
    try:
        return WMO_CODES.get(int(code), f"Unknown (code {code})")
    except (TypeError, ValueError):
        return "Unknown"


def wind_compass(degrees) -> str:
    """Convert wind direction in degrees to a 16-point compass label."""
    try:
        idx = int((float(degrees) + 11.25) // 22.5) % 16
    except (TypeError, ValueError):
        return "?"
    return COMPASS[idx]


def fetch_json(url: str) -> dict:
    """GET a URL and parse the JSON body (10s timeout)."""
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-weather-skill/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode(city: str) -> dict:
    """Resolve a city name to the top geocoding match. Raises LookupError on miss."""
    query = urllib.parse.urlencode({"name": city, "count": 1, "format": "json"})
    data = fetch_json(f"{GEOCODE_URL}?{query}")
    results = data.get("results") or []
    if not results:
        raise LookupError(f"City not found: {city!r}")
    return results[0]


def get_forecast(lat: float, lon: float, days: int, units: str) -> dict:
    """Fetch current, hourly, and daily forecast data for a coordinate."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m", "apparent_temperature", "relative_humidity_2m",
            "weather_code", "wind_speed_10m", "wind_direction_10m", "precipitation",
        ]),
        "hourly": "temperature_2m,precipitation_probability,weather_code",
        "daily": ",".join([
            "weather_code", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "precipitation_probability_max", "wind_speed_10m_max",
        ]),
        "forecast_days": days,
        "timezone": "auto",
    }
    if units == "imperial":
        params.update({
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
        })
    return fetch_json(f"{FORECAST_URL}?{urllib.parse.urlencode(params)}")


def format_text(place: dict, data: dict, units: str) -> str:
    """Render a compact chat-friendly text report."""
    t_unit = "C" if units == "metric" else "F"
    w_unit = "km/h" if units == "metric" else "mph"
    p_unit = "mm" if units == "metric" else "in"

    name = place.get("name", "?")
    region = place.get("admin1") or ""
    country = place.get("country") or ""
    location = ", ".join(x for x in (name, region, country) if x)

    lines = [f"Weather for {location}"]

    cur = data.get("current", {})
    lines.append(
        "Now: {desc}, {temp:.0f}{tu} (feels {feels:.0f}{tu}), "
        "humidity {hum:.0f}%, wind {wind:.0f} {wu} {wdir}".format(
            desc=wmo_description(cur.get("weather_code")),
            temp=cur.get("temperature_2m", float("nan")),
            feels=cur.get("apparent_temperature", float("nan")),
            hum=cur.get("relative_humidity_2m", float("nan")),
            wind=cur.get("wind_speed_10m", float("nan")),
            wdir=wind_compass(cur.get("wind_direction_10m")),
            tu=f" deg{t_unit}", wu=w_unit,
        )
    )

    daily = data.get("daily", {})
    dates = daily.get("time") or []
    if dates:
        lines.append("Forecast:")
    for i, date in enumerate(dates):
        lines.append(
            "  {date}: {desc}, {lo:.0f}/{hi:.0f} deg{tu}, "
            "precip {psum:.1f} {pu} ({pprob:.0f}%), wind up to {wmax:.0f} {wu}".format(
                date=date,
                desc=wmo_description(daily["weather_code"][i]),
                lo=daily["temperature_2m_min"][i],
                hi=daily["temperature_2m_max"][i],
                psum=daily["precipitation_sum"][i] or 0.0,
                pprob=daily["precipitation_probability_max"][i] or 0.0,
                wmax=daily["wind_speed_10m_max"][i],
                tu=t_unit, pu=p_unit, wu=w_unit,
            )
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="weather.py",
        description="Current weather and forecast via Open-Meteo (no API key).",
    )
    parser.add_argument("city", nargs="+", help="City name (multi-word OK)")
    parser.add_argument("--days", type=int, default=3,
                        help="Forecast days, 1-16 (default: 3)")
    parser.add_argument("--units", choices=["metric", "imperial"], default="metric",
                        help="Unit system (default: metric)")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        dest="fmt", help="Output format (default: text)")
    args = parser.parse_args(argv)

    if not 1 <= args.days <= 16:
        parser.error("--days must be between 1 and 16")

    city = " ".join(args.city)
    try:
        place = geocode(city)
        data = get_forecast(place["latitude"], place["longitude"],
                            args.days, args.units)
    except LookupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Error: network request failed: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError:
        print("Error: unexpected non-JSON response from Open-Meteo", file=sys.stderr)
        return 1

    if args.fmt == "json":
        print(json.dumps({"location": place, "forecast": data}, indent=2))
    else:
        print(format_text(place, data, args.units))
    return 0


if __name__ == "__main__":
    sys.exit(main())
