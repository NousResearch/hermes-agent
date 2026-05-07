# Hawaii Price Tracker — Implementation Notes

## What was built

- **State file**: `/opt/data/hawaii-price-tracker/state.json` — per-window prices per island
- **Script**: `/opt/data/scripts/hawaii-price-checker.py` — core logic
- **Wrapper**: `/opt/data/scripts/hawaii-price-checker-wrapper.py` — loads tokens from `/opt/data/.env.tokens` before execing the script
- **Cron**: Job ID `0189c547e497`, fires daily at 21:00 UTC (2 PM PT), `deliver: telegram`

## Token requirements

`SERPAPI_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_HOME_CHANNEL` — all read from `/opt/data/.env.tokens` by the wrapper. Cron agent does NOT have the container's env vars; the wrapper is mandatory.

## Trip parameters

- Cabin: `cabin=first` (Gordon wants first class only)
- Trip length: 7 days (return = dep + 7 days)
- Class: United only (filters to `if "United" in airline`)
- Departure windows: now +4w, +8w, +12w — displayed independently, no aggregate low

## Report format (June 2026 onward)

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

On SerpAPI error, each failed window shows `⚠️` and a summary line is prepended:
```
⚠️ _SerpAPI error(s): OGG 4w: HTTP 401 Unauthorized; KOA 8w: HTTP 429 rate limit_
```

## Correct SerpAPI Google Flights pattern

```python
params = {
    "engine": "google_flights",
    "departure_id": "SFO",
    "arrival_id": "OGG",
    "outbound_date": "2026-06-20",
    "return_date": "2026-06-27",   # 7-day trip
    "adults": "1",
    "currency": "USD",
    "hl": "en",
    "type": "1",      # round-trip
    "cabin": "first", # Gordon's requirement
    "api_key": SERPAPI_KEY,
}
url = f"https://serpapi.com/search?{urllib.parse.urlencode(params)}"
req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/1.0"})
```

Common mistakes:
- **Wrong endpoint**: `https://serpapi.com/flights` → 404. Use `/search`.
- **Missing User-Agent**: Cloudflare may 403 → always include `User-Agent` header.
- **Missing return_date**: Without it, `type` defaults to one-way (`type=2`), giving ~half the price. First-class one-way KOA = $179; round-trip = $407. Always include `return_date`.
- **Wrong type code**: `type=1` = round-trip, `type=2` = one-way.
- **Adding .astimezone() to date-only strings**: `datetime.fromisoformat(d).astimezone()` applies a UTC↔local shift that can nudge the date by a day. Use plain `datetime.fromisoformat(d)` when `d` is a bare date string like `"2026-06-05"`.

## Extracting United flights from response

```python
for flight in data.get("best_flights", []) + data.get("other_flights", []):
    price = flight.get("price")
    for leg in flight.get("flights", []):   # "flights" not "legs"
        airline = leg.get("airline", "")
        if "United" in airline:
            results.append({"price": int(price), ...})
            break
```

The flight segments live in `flight["flights"]` (list), not `flight["legs"]`.

## Tracking state

`state.json` stores per-island per-window data (no all-time low):
```json
{
  "OGG": {
    "updated": "2026-05-07T12:00:00-07:00",
    "windows": [
      {"weeks": 4, "dep": "2026-06-05", "price": 455, "error": null},
      {"weeks": 8, "dep": "2026-07-03", "price": 460, "error": null},
      {"weeks": 12, "dep": "2026-07-31", "price": null, "error": "HTTP 429 rate limit"}
    ]
  }
}
```

## Islands tracked

- OGG — Maui (Kahului)
- HNL — Honolulu/Oahu
- KOA — Kona
- LIH — Kauai

## Key implementation decisions (June 2026)

- **No all-time low tracking**: Gordon remembers the all-time lows himself; showing them created noise. Per-window prices only.
- **Independent windows**: Each window is searched separately; the report shows all three, not just the cheapest across all three.
- **State dir auto-created**: `_save_state` calls `os.makedirs(dirname, exist_ok=True)` so the script degrades gracefully if the state dir is deleted.
- **Errors surfaced inline**: `_search_flights` re-raises exceptions; `_window_prices` catches and returns the exception as the flight sentinel; `main()` inserts a ⚠️ warning line into the report. Errors are NOT silently swallowed.
- **_send_telegram removed**: The Hermes cron prompt handles delivery; the function was dead code.

## Bugs historically encountered

- **One-way vs round-trip (May 2026)**: Original implementation omitted `return_date`, so `type=2` (one-way) was used by default. First-class one-way KOA = $179; round-trip = $407. Fixed by always passing `return_date` and `type=1`.
- **dep_date null on first save (May 2026)**: Strict `<` comparison meant when `today_price == all_time_low`, `all_time_low_dep` was never written, leaving it null. (Pre-all-time-low-tracking-removal era.)
- **Timezone shift on date-only strings (June 2026)**: `datetime.fromisoformat(d).astimezone()` on a bare `"YYYY-MM-DD"` date string causes a UTC↔local shift that can advance the date by a day. Fixed by removing `.astimezone()` on date-only strings.
