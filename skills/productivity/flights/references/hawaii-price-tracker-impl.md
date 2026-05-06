# Hawaii Price Tracker — Implementation Notes

## What was built

- **State file**: `/opt/data/hawaii-price-tracker/state.json` — lowest seen price per island
- **Script**: `/opt/data/scripts/hawaii-price-checker.py` — core logic
- **Wrapper**: `/opt/data/scripts/hawaii-price-checker-wrapper.py` — loads tokens from `/opt/data/.env.tokens` before execing the script
- **Cron**: Job ID `0189c547e497`, fires daily at 21:00 UTC (2 PM PT), `deliver: telegram`

## Token requirements

`SERPAPI_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_HOME_CHANNEL` — all read from `/opt/data/.env.tokens` by the wrapper. Cron agent does NOT have the container's env vars; the wrapper is mandatory.

## Trip parameters

- Cabin: `cabin=first` (Gordon wants first class only)
- Trip length: 7 days (return = dep + 7 days)
- Class: United only (filters to `if "United" in airline`)
- Price windows: 3 departure dates — now +30d, +45d, +60d. Cheapest across all three wins.

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

`state.json` stores per-island:
- `lowest_price` — all-time low
- `best_flight` — {price, airline, flight_number} for that low
- `dep_date` — ISO date string of the best departure (e.g. "2026-06-05")
- `updated` — ISO timestamp

All-time low only updates on a *strictly lower* price (`<`). The all-time low date is the date that achieved that low.

## Islands tracked

- OGG — Maui (Kahului)
- HNL — Honolulu/Oahu
- KOA — Kona
- LIH — Kauai

## Bugs encountered

- **One-way vs round-trip (May 2026)**: Original implementation omitted `return_date`, so `type=2` (one-way) was used by default. First-class one-way KOA = $179; round-trip = $407. Fixed by always passing `return_date` and `type=1`.
- **dep_date null on first save (May 2026)**: Strict `<` comparison meant when `today_price == all_time_low`, `all_time_low_dep` was never written, leaving it null. Pre-existing state was patched manually to set `dep_date: "2026-06-05"` for all islands.
