# Hawaii Price Tracker — Implementation Notes

## What was built

- **State file**: `/opt/data/hawaii-price-tracker/state.json` — lowest seen price per island
- **Script**: `/opt/data/scripts/hawaii-price-checker.py` — core logic
- **Wrapper**: `/opt/data/scripts/hawaii-price-checker-wrapper.py` — loads tokens from `/opt/data/.env.tokens` before execing the script
- **Cron**: Job ID `0189c547e497`, fires daily at 21:00 UTC (2 PM PT), `deliver: telegram`

## Token requirements

`SERPAPI_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_HOME_CHANNEL` — all read from `/opt/data/.env.tokens` by the wrapper. Cron agent does NOT have the container's env vars; the wrapper is mandatory.

## Correct SerpAPI Google Flights pattern

```python
# RIGHT — matches flights_client.py exactly
params = {
    "engine": "google_flights",
    "departure_id": "SFO",
    "arrival_id": "OGG",
    "outbound_date": "2026-06-20",
    "adults": "1",
    "currency": "USD",
    "hl": "en",
    "api_key": SERPAPI_KEY,
}
# For round-trip:  params["return_date"] = "2026-06-27"; params["type"] = "1"
# For one-way:  params["type"] = "2"

url = f"https://serpapi.com/search?{urllib.parse.urlencode(params)}"
req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/1.0"})
```

Common mistakes:
- **Wrong endpoint**: `https://serpapi.com/flights` → 404. Use `/search`.
- **Missing User-Agent**: Cloudflare may 403 → always include `User-Agent` header.
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

## Price windows

3 departure dates: now +30d, +45d, +60d. Best price across all three wins.

## Islands tracked

- OGG — Maui (Kahului)
- HNL — Honolulu/Oahu
- KOA — Kona
- LIH — Kauai
