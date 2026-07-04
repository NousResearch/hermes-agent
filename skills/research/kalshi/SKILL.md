---
name: kalshi
description: "Query Kalshi: markets, prices, orderbooks, candlesticks."
version: 1.0.0
author: Benedict Anokye-Davies, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Kalshi, Prediction-Markets, Market-Data, Trading, CFTC]
    category: research
    related_skills: [polymarket]
    homepage: https://kalshi.com
---

# Kalshi — Prediction Market Data

Query prediction market data from Kalshi (the CFTC-regulated US prediction
market exchange) using their public REST APIs. All endpoints used here are
read-only and require zero authentication.

See `references/api-endpoints.md` for the full endpoint reference with curl
examples.

## When to Use

- User asks about prediction markets, betting odds, or event probabilities
- User asks about Kalshi specifically
- User wants market prices, orderbook depth, candlesticks, or recent public
  trades
- User wants to monitor or track Kalshi market movements

For non-US prediction markets or crypto-settled markets, use the `polymarket`
skill instead.

## Key Concepts

- **Series** group related **Events** (e.g. series `KXNBAGAME` covers every
  NBA game; an event is one specific game).
- **Events** contain one or more **Markets**.
- **Markets** are binary Yes/No contracts. Prices are in **dollars**, range
  $0.00–$1.00, and **are probabilities**: `yes_bid_dollars=0.64` means the
  best bid implies a 64% probability of Yes.
- **Tickers** are uppercase strings: a market ticker looks like
  `KXNBAGAME-26JUN03NYKSAS-SAS` (series-event-side). The event ticker is
  the prefix without the trailing side: `KXNBAGAME-26JUN03NYKSAS`. The
  series ticker is the leading token: `KXNBAGAME`.
- Volume and open interest are in **contracts** (`_fp` suffix means
  fractional contracts as a string).
- Many price fields ship in pairs: dollar-denominated (`*_dollars`) and
  cent-denominated (`*_cents`). Prefer the dollar fields; they are strings
  to avoid float drift.

## One Public API

Kalshi exposes a single REST surface for read-only data:

- **Trade API v2** at `api.elections.kalshi.com/trade-api/v2`

All endpoints used in this skill are GET, return JSON, and need no auth.
Trading endpoints (`POST /portfolio/orders`, etc.) require an RSA-signed
API key and are out of scope for this skill — use the `kalshi-python`
SDK directly for those.

## Typical Workflow

When a user asks about Kalshi odds:

1. **Search** for relevant events using `GET /events?status=open` (or
   filter by `series_ticker` if you know the topic).
2. **Inspect** specific markets via `GET /markets/{ticker}` to read the
   resolution rules and current prices.
3. **Present** the market title, current price as a percentage, the
   bid/ask spread, and 24h volume.
4. **Deep dive** if asked — pull `/orderbook` for depth, `/markets/trades`
   for the most recent public fills, or `/candlesticks` for OHLC history.

## Presenting Results

Convert prices from dollars to percentages for readability:

- `yes_bid_dollars=0.64`, `yes_ask_dollars=0.65` → "Yes 64.0%–65.0%"
- Show the market title, the implied probability, and the 24h volume
- When showing a spread, always show both sides; thin markets can have
  20-cent spreads where the midpoint is misleading
- Convert UTC timestamps to the user's timezone before quoting; never
  paste raw UTC into a chat reply

Example:
`"Will San Antonio beat NY (Game 1)?" — Yes 64.0%–65.0% (vol 24h: 4,213, OI: 639k)`

## Parsing Numeric Fields

Prices and quantities are returned as **strings** to preserve precision.
Always cast explicitly:

```python
yes_price = float(market["yes_bid_dollars"])
volume_24h = float(market["volume_24h_fp"])
```

The `_fp` suffix indicates a fractional value; treat it as `float`.
The `_dollars` suffix is dollar-denominated to four decimal places.

Candlestick `end_period_ts` values are UTC unix seconds — convert to the
user's local timezone before presenting times.

## Rate Limits

Public read endpoints are generous and rarely an issue for human-paced
queries. Authenticated trading endpoints have stricter limits documented
on the Kalshi developer portal — irrelevant for this skill.

If you hit a 429 the helper script backs off exponentially (1s, 2s, 4s)
and retries up to three times.

## Limitations

- This skill is read-only — it does not place trades. Trading requires an
  RSA-signed API key, KYC'd Kalshi account, and US residency.
- Kalshi's public API requires no auth for reads, but the CDN occasionally
  serves a 403 to non-browser User-Agents — the helper script sets a
  descriptive UA that has been working as of June 2026.
- Some markets (e.g. weather, climate) settle on external data feeds; the
  resolution rules are in the `rules_primary` and `rules_secondary` fields
  on the market object — read them before quoting probabilities as fact.
- The candlesticks endpoint accepts `period_interval` of `60` or `1440`
  per Kalshi's official docs. A `1`-minute interval works empirically but
  is undocumented and may be removed without notice.
- Geographic restrictions apply to trading; read-only data is globally
  accessible.

## How to Run

Invoke through the `terminal` tool. Once installed:

```
SCRIPT=~/.hermes/skills/research/kalshi/scripts/kalshi.py
python3 $SCRIPT events --status open --limit 5
python3 $SCRIPT markets --event-ticker KXNBAGAME-26JUN03NYKSAS
python3 $SCRIPT market KXNBAGAME-26JUN03NYKSAS-SAS
python3 $SCRIPT orderbook KXNBAGAME-26JUN03NYKSAS-SAS --depth 5
python3 $SCRIPT recent-trades KXNBAGAME-26JUN03NYKSAS-SAS --limit 5
python3 $SCRIPT candles KXNBAGAME KXNBAGAME-26JUN03NYKSAS-SAS --hours 1
python3 $SCRIPT series KXNBAGAME
```

List endpoints (`events`, `markets`, `recent-trades`) accept `--cursor` to
fetch the next page when the response includes a non-empty cursor.

Python stdlib only — no API key, no pip installs.
