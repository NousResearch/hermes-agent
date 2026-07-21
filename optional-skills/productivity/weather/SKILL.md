---
name: weather
description: Current weather and forecasts via Open-Meteo, no API key.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Weather, Forecast, Utilities]
    related_skills: []
---

# Weather

Look up current conditions and a multi-day forecast for any city using the
free Open-Meteo APIs (geocoding + forecast), which require no API key. The
script is pure Python standard library and prints a compact, chat-friendly
text report.

## When to Use

- The user asks about current weather, temperature, wind, or precipitation
  for a named place.
- The user asks for a forecast ("what's the weather in Berlin this week?").
- You need machine-readable weather data (`--format json`) for a follow-up
  computation.

## Prerequisites

None beyond `python3` (3.8+). The script uses only the standard library
(`urllib`, `json`, `argparse`) — no pip installs, no API key, no config.

## How to Run

Run through the `terminal` tool:

```bash
python3 ~/.hermes/skills/productivity/weather/scripts/weather.py "New York"
```

Common variants (same script path):

```bash
weather.py Berlin --days 7
weather.py Tokyo --units imperial
weather.py Paris --days 5 --format json
```

## Quick Reference

| Flag | Values | Default | Meaning |
| --- | --- | --- | --- |
| `city` (positional) | one or more words | required | City name; multi-word names work quoted or unquoted |
| `--days` | 1-16 | 3 | Number of forecast days |
| `--units` | `metric`, `imperial` | `metric` | degC/km/h/mm vs degF/mph/inch |
| `--format` | `text`, `json` | `text` | Compact text for chat, or raw JSON |

## Procedure

1. Run the script with the city name the user gave. Multi-word names are
   joined automatically (`weather.py New York` works).
2. If the user implies a unit preference (US locations often expect
   Fahrenheit), pass `--units imperial`.
3. Relay the output. The first line names the resolved location
   (city, region, country) — mention it so the user can catch a wrong match.
4. For programmatic needs, use `--format json` and parse the `location` and
   `forecast` keys.

## Pitfalls

- **Geocoding ambiguity**: the script takes the top geocoding match
  (e.g. "Springfield" resolves to Springfield, Missouri). Always echo the
  resolved location line back to the user; add a state/country to the query
  ("Springfield Illinois") to disambiguate.
- **Rate limits**: Open-Meteo's free tier allows roughly 10,000 calls/day
  for non-commercial use. Fine for chat usage; don't loop it in bulk jobs.
- **WMO code coverage**: only the documented WMO weather codes are mapped;
  an unexpected code prints as `Unknown (code N)` rather than failing.
- **Failures**: city-not-found and network errors exit 1 with a message on
  stderr — check the exit code, not just stdout.

## Verification

```bash
python3 ~/.hermes/skills/productivity/weather/scripts/weather.py London --days 1
```

Expect a "Weather for London, England, United Kingdom" header, a "Now:"
line, and one forecast line.
