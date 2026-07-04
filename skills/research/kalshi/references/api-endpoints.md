# Kalshi API Endpoints Reference

All endpoints below are public REST (GET), return JSON, and need no
authentication. Base URL: `https://api.elections.kalshi.com/trade-api/v2`.

For trading endpoints (orders, portfolio, fills) see the official Kalshi
developer docs at https://trading-api.readme.io/ — those require an RSA
API key and are out of scope for this skill.

## Exchange Status

```
GET /exchange/status
```

Response:
```json
{"exchange_active": true, "trading_active": true}
```

Use this as a cheap health check before a batch of queries.

## Series

A **series** groups related events. e.g. `KXNBAGAME` covers every NBA game;
`KXPRESIDENT` covers each presidential election cycle.

### Get a Series

```
GET /series/{series_ticker}
```

Response includes the series title, category, contract terms URL, fee model
(`flat` or `quadratic`), and settlement source list.

### List Series

```
GET /series?limit=N&category=Politics
```

Common categories: `Politics`, `Economics`, `Sports`, `Climate`, `Crypto`,
`World`, `Companies`, `Entertainment`.

## Events

An **event** is one specific occurrence inside a series. e.g. event
`KXNBAGAME-26JUN03NYKSAS` is the NY at SA NBA game on 2026-06-03.

### Get an Event

```
GET /events/{event_ticker}
GET /events/{event_ticker}?with_nested_markets=true
```

Set `with_nested_markets=true` to inline the markets array in the response
(saves a second round-trip).

### List Events

```
GET /events?limit=N&status=open&series_ticker=KXNBAGAME&with_nested_markets=true
```

Parameters:
- `limit` — max results, default 100, max 200
- `cursor` — pagination cursor returned in the previous response
- `status` — `open` (active), `closed` (resolved), `settled`, `unopened`
- `series_ticker` — filter to one series
- `with_nested_markets` — true/false; inline markets

Response shape:
```json
{
  "events": [...],
  "cursor": "CgYI8Mzn4AsSDEtYTkVXUE9QRS03MA"
}
```

When `cursor` is non-empty, more results are available. Pass it back as
`cursor=...` to fetch the next page.

## Markets

A **market** is one binary Yes/No contract. e.g. `KXNBAGAME-26JUN03NYKSAS-SAS`
is "San Antonio wins Game 1".

### Get a Market

```
GET /markets/{ticker}
```

Response: a `market` object with the full set of fields below.

Key market fields:
- `ticker` — unique market identifier
- `event_ticker` — parent event
- `title` / `subtitle` / `yes_sub_title` / `no_sub_title` — display strings
- `status` — `active`, `closed`, `settled`, `unopened`
- `market_type` — usually `binary`
- `yes_bid_dollars` / `yes_ask_dollars` — best bid/ask for Yes side ($0–$1)
- `no_bid_dollars` / `no_ask_dollars` — best bid/ask for No side
- `last_price_dollars` — last traded price
- `previous_price_dollars` — previous tick's last price
- `volume_fp` / `volume_24h_fp` — total / 24h volume in contracts (string)
- `open_interest_fp` — total contracts outstanding (string)
- `liquidity_dollars` — quoted liquidity in dollars
- `rules_primary` / `rules_secondary` — resolution rules text
- `expected_expiration_time` / `expiration_time` / `close_time` — UTC ISO 8601
- `result` — empty until settled; one of `yes`, `no`, `void`

All `*_dollars` and `*_fp` fields are JSON strings to preserve precision.
Cast with `float()` in Python.

### List Markets

```
GET /markets?status=open&event_ticker=KXNBAGAME-26JUN03NYKSAS&limit=N
```

Parameters:
- `status`, `cursor`, `limit` — same as events
- `event_ticker` — filter to one event
- `series_ticker` — filter to one series
- `tickers` — comma-separated list of explicit market tickers

## Orderbook

```
GET /markets/{ticker}/orderbook
GET /markets/{ticker}/orderbook?depth=5
```

Response:
```json
{
  "orderbook_fp": {
    "yes_dollars": [["0.6400", "5912.30"], ["0.6300", "12010.55"], ...],
    "no_dollars":  [["0.0100", "6313778.79"], ["0.0200", "516913.49"], ...]
  }
}
```

Each row is `[price_dollars, size_contracts]`, both as strings. The Yes
side is sorted descending by price (best bid first); the No side is sorted
ascending (best bid first from the No taker's perspective).

To compute the Yes-side best ask from the No-side book: `1 - best_no_bid`.

## Recent Trades

```
GET /markets/trades?ticker={ticker}&limit=N
GET /markets/trades?ticker={ticker}&limit=N&min_ts=UNIX&max_ts=UNIX
```

Parameters:
- `ticker` — required; market ticker
- `limit` — max trades returned, default 100
- `cursor` — pagination
- `min_ts` / `max_ts` — UNIX seconds; filter window

Response: `{"trades": [...], "cursor": "..."}`

Each trade includes `trade_id`, `created_time`, `yes_price_dollars`,
`no_price_dollars`, `count_fp` (contracts), `taker_side` (`yes`/`no`), and
`taker_book_side` (`bid`/`ask` — which side the taker hit).

### Candlesticks (OHLC History)

```
GET /series/{series_ticker}/markets/{market_ticker}/candlesticks
    ?start_ts=UNIX&end_ts=UNIX&period_interval=N
```

Parameters:
- `start_ts` / `end_ts` — UNIX seconds (UTC); range is required
- `period_interval` — minutes per candle: `60` (hourly) or `1440` (daily)
  per the official Kalshi docs. A `1`-minute interval works empirically on
  the public endpoint but is undocumented and may be removed without notice.

Response:
```json
{
  "candlesticks": [
    {
      "end_period_ts": 1780337400,
      "open_interest_fp": "589552.69",
      "volume_fp": "242.62",
      "price":   {"open_dollars": "0.64", "high_dollars": "0.64", "low_dollars": "0.64", "close_dollars": "0.64", "mean_dollars": "0.64", "previous_dollars": "0.64"},
      "yes_ask": {"open_dollars": "0.64", "high_dollars": "0.64", "low_dollars": "0.64", "close_dollars": "0.64"},
      "yes_bid": {"open_dollars": "0.63", "high_dollars": "0.63", "low_dollars": "0.63", "close_dollars": "0.63"}
    }
  ]
}
```

Note the path: `series/{series_ticker}/markets/{market_ticker}` — not
`/markets/{ticker}/candlesticks`. The series ticker is the prefix of the
market ticker (e.g. `KXNBAGAME` for `KXNBAGAME-26JUN03NYKSAS-SAS`).

## Pagination Pattern

Every list endpoint returns a `cursor` field. When non-empty, more results
are available — pass it back as `cursor=...` to get the next page. Stop
when the response returns an empty cursor.

## Errors

The API returns standard HTTP status codes:
- `200` — success
- `400` — bad request (check parameter names and types)
- `404` — ticker not found (typo or wrong path; double-check series vs.
  market ticker)
- `429` — rate limited (back off and retry)
- `5xx` — Kalshi-side issue; retry with exponential backoff
