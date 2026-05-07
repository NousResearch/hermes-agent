---
name: flights
description: Search flights via SerpAPI Google Flights engine — one-way or round-trip, with pricing and booking links.
category: productivity
requires_toolsets:
  - terminal
tags:
  - flights
  - travel
  - serpapi
  - google-flights
---

# Flights Search (SerpAPI)

## When to Use

- User asks to find flights between two airports (IATA codes)
- User wants flight prices, durations, or schedules for a specific date
- User needs a Google Flights link to refine/book

## Prerequisites

- `SERPAPI_KEY` environment variable must be set (already provisioned in this container)
- Python 3 with stdlib only (no pip deps)

## CLI Usage

```bash
python3 /opt/data/skills/productivity/flights/scripts/flights_client.py \
  search <DEPARTURE_IATA> <ARRIVAL_IATA> <YYYY-MM-DD> [OPTIONS]
```

### Required Arguments

| Arg | Description |
|-----|-------------|
| `DEP` | Departure airport IATA code (e.g. SFO, JFK, LHR) |
| `ARR` | Arrival airport IATA code (e.g. LAX, NRT, CDG) |
| `DATE` | Outbound date in YYYY-MM-DD format |

### Optional Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--return YYYY-MM-DD` | *(none — one-way)* | Return date for round-trip |
| `--adults N` | 1 | Number of adult passengers |
| `--currency XXX` | USD | Currency code for prices |
| `--cabin first\|business\|economy` | *(all)* | Cabin class filter |

### Examples

```bash
# One-way SFO → LAX on May 15
python3 .../flights_client.py search SFO LAX 2026-05-15

# Round-trip JFK → LHR, 2 adults, GBP pricing
python3 .../flights_client.py search JFK LHR 2026-06-01 --return 2026-06-15 --adults 2 --currency GBP
```

## Output Format

JSON object with:
- `google_flights_url` — direct link to Google Flights for the same query
- `flights` — array of up to 5 results, each containing:
  - `price` — integer price in requested currency
  - `total_duration` — total trip duration in minutes
  - `legs` — array of flight segments:
    - `airline` — carrier name
    - `flight_number` — e.g. "UA 1234"
    - `departure_airport` — IATA code
    - `departure_time` — ISO-ish datetime string
    - `arrival_airport` — IATA code
    - `arrival_time` — ISO-ish datetime string
    - `duration` — segment duration in minutes

## Pitfalls

- IATA codes must be uppercase (script uppercodes automatically)
- Dates in the past will return empty results from SerpAPI
- SerpAPI free tier has limited monthly searches — don't loop/retry excessively
- Some routes return no "best_flights" — the script falls back to "other_flights"
- `curl` is not available in this container — use the script (Python urllib) for all HTTP calls

## Hawaii Price Tracker

Ongoing monitoring for cheap SFO → Hawaii (OGG/HNL/KOA/LIH) round-trip United **first class** fares, **7-day trip**. Sends a daily report to Telegram at 2 PM PT — always, not just on new lows. Reports all three window prices independently (4w, 8w, 12w). No all-time low tracking.

Script: `/opt/data/scripts/hawaii-price-checker.py`
Tokens via wrapper: `/opt/data/scripts/hawaii-price-checker-wrapper.py` (reads `/opt/data/.env.tokens` — cron agent has no container env vars, wrapper is mandatory)
State: `/opt/data/hawaii-price-tracker/state.json`
Cron: job ID `0189c547e497`, `0 21 * * *` UTC (2 PM PT), deliver=telegram

Report format (Markdown, Telegram-native):
```
✈️ *Hawaii First Class — SFO RT (7 days)*
_May 06_

*Maui (OGG)*: 4w $455 (Fri Jun 05) | 8w $460 (Fri Jul 03) | 12w $445 (Fri Jul 31)
*Honolulu (HNL)*: 4w $472 (Fri Jun 05) | 8w $478 (Fri Jul 03) | 12w $470 (Fri Jul 31)
*Kona (KOA)*: 4w $407 (Fri Jun 05) | 8w $415 (Fri Jul 03) | 12w $400 (Fri Jul 31)
*Kauai (LIH)*: 4w $481 (Fri Jun 05) | 8w $490 (Fri Jul 03) | 12w $475 (Fri Jul 31)

_United first class · SFO round-trip · 7-day trip_
_4/8/12-week departure windows_
```

On SerpAPI error, failed windows show `⚠️` and a warning line is prepended to the report. Errors are NOT silently swallowed.

See `references/hawaii-price-tracker-impl.md` for implementation detail.

## Route Intelligence (California Short-Haul)

**OAK ↔ SBA (Southwest only):** Only ~1 flight/day, departures mid-afternoon to evening (earliest ~14:25). NOT a morning-departure route. Route reportedly ends October 2026.

**SFO ↔ SBA (United only, nonstop):** Best same-day options. First SFO→SBA: ~10:27–10:43 AM. Last SBA→SFO: ~8:24 PM (CRJ-700). Gives ~9.5 hours in SB — workable but not a full day.

**SFO is the correct airport** if you need a real early-morning departure to SB with evening return. OAK Southwest flights don't depart until the afternoon.

| Route Intelligence (California Short-Haul) | `references/sfo-sba-may30-2026.md` — May 30, 2026 schedule sample |

## Troubleshooting

### HTTP 401 "Invalid API key"
The key exists in `SERPAPI_KEY` but SerpAPI rejects it. Validate independently:
```python
python3 -c "
import urllib.request, os
url = f'https://serpapi.com/account?api_key={os.environ[\"SERPAPI_KEY\"]}'
with urllib.request.urlopen(url, timeout=10) as r: print(r.read().decode())
"
```
If `/account` also 401s, the key is expired/revoked — ask the user to refresh it. Don't retry the flights search; it will fail the same way.
