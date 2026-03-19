#!/usr/bin/env python3
"""
weather_client.py - A CLI weather tool using Open-Meteo API (no API key required).
Part of the Hermes Agent project.

Usage:
  weather_client.py now LOCATION
  weather_client.py forecast LOCATION [--days N]
  weather_client.py daily LOCATION [--days N]
  weather_client.py compare LOCATION1 LOCATION2 [LOCATION3 ...]
  weather_client.py alerts LOCATION
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Heavy thunderstorm with hail",
}

ALERT_THRESHOLDS = {
    "frost": {"param": "temperature", "op": "lt", "value": 2, "severity": "warning",
              "message": "Frost risk: temperature below 2°C"},
    "heat": {"param": "temperature", "op": "gt", "value": 35, "severity": "warning",
             "message": "Heat alert: temperature above 35°C"},
    "strong_wind": {"param": "wind_speed", "op": "gt", "value": 60, "severity": "warning",
                    "message": "Strong wind: speed above 60 km/h"},
    "heavy_rain": {"param": "precipitation", "op": "gt", "value": 10, "severity": "danger",
                   "message": "Heavy rain: precipitation above 10 mm/h"},
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_json(data):
    """Print data as formatted JSON to stdout."""
    print(json.dumps(data, indent=2))


def wmo_description(code):
    """Return human-readable description for a WMO weather code."""
    if code is None:
        return "Unknown"
    return WMO_CODES.get(int(code), f"Unknown (code {code})")


def fetch_url(url, params=None, retries=5, backoff_base=1.0):
    """
    Fetch a URL with optional query params. Retries on HTTP 429 with
    exponential backoff. Returns parsed JSON dict on success.
    Raises SystemExit on unrecoverable errors.
    """
    if params:
        url = url + "?" + urllib.parse.urlencode(params)

    attempt = 0
    while attempt < retries:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HermesAgent-WeatherClient/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = backoff_base * (2 ** attempt)
                sys.stderr.write(
                    f"Rate limited (429). Retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{retries})...\n"
                )
                time.sleep(wait)
                attempt += 1
                continue
            else:
                sys.stderr.write(
                    f"HTTP error {exc.code} fetching {url}: {exc.reason}\n"
                )
                sys.exit(1)
        except urllib.error.URLError as exc:
            sys.stderr.write(f"Network error fetching {url}: {exc.reason}\n")
            sys.exit(1)
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"Failed to parse JSON response: {exc}\n")
            sys.exit(1)

    sys.stderr.write(f"Max retries exceeded for {url}\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode(location_name):
    """
    Resolve a location name to coordinates via Open-Meteo geocoding API.
    Returns a dict with keys: name, latitude, longitude, timezone, country.
    Raises SystemExit if location is not found.
    """
    data = fetch_url(GEOCODING_URL, params={"name": location_name, "count": 1})

    results = data.get("results")
    if not results:
        sys.stderr.write(
            f"Location not found: '{location_name}'. "
            "Try a different spelling or a nearby city.\n"
        )
        sys.exit(1)

    result = results[0]
    return {
        "name": result.get("name", location_name),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "timezone": result.get("timezone", "UTC"),
        "country": result.get("country", ""),
        "admin1": result.get("admin1", ""),
    }


def location_label(loc):
    """Return a short display label for a location dict."""
    parts = [loc["name"]]
    if loc.get("admin1"):
        parts.append(loc["admin1"])
    if loc.get("country"):
        parts.append(loc["country"])
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Command: now
# ---------------------------------------------------------------------------

def cmd_now(location_name):
    """Fetch and display current weather conditions for a location."""
    loc = geocode(location_name)

    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "current": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "is_day",
            "precipitation",
        ]),
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    data = fetch_url(WEATHER_URL, params=params)
    current = data.get("current", {})
    cu = data.get("current_units", {})

    result = {
        "location": {
            "name": location_label(loc),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timezone": loc["timezone"],
        },
        "current_weather": {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "weather_code": current.get("weather_code"),
            "weather_description": wmo_description(current.get("weather_code")),
            "is_day": bool(current.get("is_day")),
            "precipitation": current.get("precipitation"),
        },
        "units": {
            "temperature": "°C",
            "feels_like": "°C",
            "humidity": "%",
            "wind_speed": "km/h",
            "wind_direction": "°",
            "precipitation": "mm",
        },
    }

    print_json(result)


# ---------------------------------------------------------------------------
# Command: forecast (hourly)
# ---------------------------------------------------------------------------

def cmd_forecast(location_name, days=3):
    """Fetch and display hourly forecast for the given number of days."""
    days = max(1, min(days, 7))
    loc = geocode(location_name)

    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "forecast_days": days,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation_probability",
            "weather_code",
            "wind_speed_10m",
            "precipitation",
        ]),
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    data = fetch_url(WEATHER_URL, params=params)
    hourly = data.get("hourly", {})

    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip_prob = hourly.get("precipitation_probability", [])
    codes = hourly.get("weather_code", [])
    winds = hourly.get("wind_speed_10m", [])
    precip = hourly.get("precipitation", [])

    hours = []
    for i, t in enumerate(times):
        hours.append({
            "time": t,
            "temperature": temps[i] if i < len(temps) else None,
            "precipitation_probability": precip_prob[i] if i < len(precip_prob) else None,
            "precipitation": precip[i] if i < len(precip) else None,
            "weather_code": codes[i] if i < len(codes) else None,
            "weather_description": wmo_description(codes[i] if i < len(codes) else None),
            "wind_speed": winds[i] if i < len(winds) else None,
        })

    result = {
        "location": {
            "name": location_label(loc),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timezone": loc["timezone"],
        },
        "forecast_days": days,
        "total_hours": len(hours),
        "hourly_forecast": hours,
        "units": {
            "temperature": "°C",
            "precipitation_probability": "%",
            "precipitation": "mm",
            "wind_speed": "km/h",
        },
    }

    print_json(result)


# ---------------------------------------------------------------------------
# Command: daily
# ---------------------------------------------------------------------------

def cmd_daily(location_name, days=7):
    """Fetch and display daily weather summary for the given number of days."""
    days = max(1, min(days, 7))
    loc = geocode(location_name)

    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "forecast_days": days,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "sunrise",
            "sunset",
            "weather_code",
        ]),
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    data = fetch_url(WEATHER_URL, params=params)
    daily = data.get("daily", {})

    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precip_sum = daily.get("precipitation_sum", [])
    precip_prob_max = daily.get("precipitation_probability_max", [])
    sunrises = daily.get("sunrise", [])
    sunsets = daily.get("sunset", [])
    codes = daily.get("weather_code", [])

    days_list = []
    for i, d in enumerate(dates):
        days_list.append({
            "date": d,
            "temp_max": temp_max[i] if i < len(temp_max) else None,
            "temp_min": temp_min[i] if i < len(temp_min) else None,
            "precipitation_sum": precip_sum[i] if i < len(precip_sum) else None,
            "precipitation_probability_max": precip_prob_max[i] if i < len(precip_prob_max) else None,
            "sunrise": sunrises[i] if i < len(sunrises) else None,
            "sunset": sunsets[i] if i < len(sunsets) else None,
            "weather_code": codes[i] if i < len(codes) else None,
            "weather_description": wmo_description(codes[i] if i < len(codes) else None),
        })

    result = {
        "location": {
            "name": location_label(loc),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timezone": loc["timezone"],
        },
        "forecast_days": days,
        "daily_summary": days_list,
        "units": {
            "temperature": "°C",
            "precipitation_sum": "mm",
            "precipitation_probability_max": "%",
        },
    }

    print_json(result)


# ---------------------------------------------------------------------------
# Command: compare
# ---------------------------------------------------------------------------

def _fetch_current_for_location(loc):
    """Helper: fetch current weather dict for an already-geocoded location."""
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "current": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "is_day",
            "precipitation",
        ]),
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    data = fetch_url(WEATHER_URL, params=params)
    current = data.get("current", {})
    return {
        "time": current.get("time"),
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_direction": current.get("wind_direction_10m"),
        "weather_code": current.get("weather_code"),
        "weather_description": wmo_description(current.get("weather_code")),
        "is_day": bool(current.get("is_day")),
        "precipitation": current.get("precipitation"),
    }


def cmd_compare(location_names):
    """Compare current weather across multiple locations."""
    if len(location_names) < 2:
        sys.stderr.write("compare requires at least two locations.\n")
        sys.exit(1)

    comparisons = []
    for name in location_names:
        loc = geocode(name)
        weather = _fetch_current_for_location(loc)
        comparisons.append({
            "location": {
                "name": location_label(loc),
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "timezone": loc["timezone"],
            },
            "current_weather": weather,
        })

    result = {
        "comparison": comparisons,
        "units": {
            "temperature": "°C",
            "feels_like": "°C",
            "humidity": "%",
            "wind_speed": "km/h",
            "wind_direction": "°",
            "precipitation": "mm",
        },
    }

    print_json(result)


# ---------------------------------------------------------------------------
# Command: alerts
# ---------------------------------------------------------------------------

def cmd_alerts(location_name):
    """Check for severe weather conditions and return active alerts."""
    loc = geocode(location_name)

    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "current": ",".join([
            "temperature_2m",
            "wind_speed_10m",
            "precipitation",
            "weather_code",
            "is_day",
        ]),
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    data = fetch_url(WEATHER_URL, params=params)
    current = data.get("current", {})

    readings = {
        "temperature": current.get("temperature_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "precipitation": current.get("precipitation"),
    }

    active_alerts = []

    for alert_name, cfg in ALERT_THRESHOLDS.items():
        param = cfg["param"]
        value = readings.get(param)
        if value is None:
            continue

        triggered = False
        if cfg["op"] == "lt" and value < cfg["value"]:
            triggered = True
        elif cfg["op"] == "gt" and value > cfg["value"]:
            triggered = True

        if triggered:
            active_alerts.append({
                "alert": alert_name,
                "severity": cfg["severity"],
                "message": cfg["message"],
                "observed_value": value,
                "threshold": cfg["value"],
                "parameter": param,
            })

    result = {
        "location": {
            "name": location_label(loc),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timezone": loc["timezone"],
        },
        "alert_check_time": current.get("time"),
        "active_alerts": active_alerts,
        "alert_count": len(active_alerts),
        "current_readings": {
            "temperature": readings["temperature"],
            "wind_speed": readings["wind_speed"],
            "precipitation": readings["precipitation"],
            "weather_code": current.get("weather_code"),
            "weather_description": wmo_description(current.get("weather_code")),
        },
        "units": {
            "temperature": "°C",
            "wind_speed": "km/h",
            "precipitation": "mm",
        },
        "thresholds": {
            "frost_below": "2°C",
            "heat_above": "35°C",
            "strong_wind_above": "60 km/h",
            "heavy_rain_above": "10 mm/h",
        },
    }

    print_json(result)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="weather_client.py",
        description=(
            "A CLI weather tool powered by Open-Meteo (no API key required).\n"
            "Part of the Hermes Agent project.\n\n"
            "Examples:\n"
            "  weather_client.py now London\n"
            "  weather_client.py forecast 'New York' --days 5\n"
            "  weather_client.py daily Tokyo --days 3\n"
            "  weather_client.py compare Paris Berlin Madrid\n"
            "  weather_client.py alerts Sydney\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # now
    p_now = subparsers.add_parser(
        "now",
        help="Current weather conditions for a location.",
        description=(
            "Fetch current weather for a location.\n"
            "Shows: temperature, feels_like, humidity, wind, weather description, is_day."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_now.add_argument(
        "location",
        nargs="+",
        help="Location name (e.g. London, 'New York', Tokyo).",
    )

    # forecast
    p_forecast = subparsers.add_parser(
        "forecast",
        help="Hourly forecast for a location.",
        description=(
            "Fetch hourly forecast for a location.\n"
            "Shows: time, temperature, precipitation_probability, weather description, wind_speed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_forecast.add_argument(
        "location",
        nargs="+",
        help="Location name.",
    )
    p_forecast.add_argument(
        "--days",
        type=int,
        default=3,
        metavar="N",
        help="Number of forecast days (1-7, default: 3).",
    )

    # daily
    p_daily = subparsers.add_parser(
        "daily",
        help="Daily weather summary for a location.",
        description=(
            "Fetch daily weather summaries for a location.\n"
            "Shows: date, temp_max, temp_min, precipitation, sunrise, sunset, weather description."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_daily.add_argument(
        "location",
        nargs="+",
        help="Location name.",
    )
    p_daily.add_argument(
        "--days",
        type=int,
        default=7,
        metavar="N",
        help="Number of days to show (1-7, default: 7).",
    )

    # compare
    p_compare = subparsers.add_parser(
        "compare",
        help="Compare current weather across multiple locations.",
        description=(
            "Compare current weather side-by-side for two or more locations.\n"
            "Example: weather_client.py compare Paris Berlin Madrid"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_compare.add_argument(
        "locations",
        nargs="+",
        metavar="LOCATION",
        help="Two or more location names to compare.",
    )

    # alerts
    p_alerts = subparsers.add_parser(
        "alerts",
        help="Check for severe weather alerts at a location.",
        description=(
            "Check current conditions for severe weather at a location.\n"
            "Alerts: frost (<2°C), heat (>35°C), strong wind (>60 km/h), heavy rain (>10 mm/h)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_alerts.add_argument(
        "location",
        nargs="+",
        help="Location name.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "now":
        location_name = " ".join(args.location)
        cmd_now(location_name)

    elif args.command == "forecast":
        location_name = " ".join(args.location)
        cmd_forecast(location_name, days=args.days)

    elif args.command == "daily":
        location_name = " ".join(args.location)
        cmd_daily(location_name, days=args.days)

    elif args.command == "compare":
        cmd_compare(args.locations)

    elif args.command == "alerts":
        location_name = " ".join(args.location)
        cmd_alerts(location_name)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
