# Quotient API Reference (Skill-Focused)

Runtime + discovery base: `https://q-api.quotient.social`

For canonical schema details, use `https://q-api.quotient.social/openapi.json`.

## Access and Authorization

- Monetized requests can be authorized with either:
  - `x-quotient-api-key`, or
  - x402 challenge/settle flow (`PAYMENT-REQUIRED` -> `PAYMENT-SIGNATURE`).
- API keys are created from the account/developer area after signup/login at `https://dev.quotient.social` (email or Google).
- New accounts include free credits for initial API usage.

## Pagination Contract

- `cursor` is opaque; pass it exactly as returned.
- Cursor values are bound to endpoint + sort + active filters.
- Reusing a cursor with different filters/sort returns `422 invalid_cursor`.
- Loop until `has_more` is false.

## Endpoints

### GET /api/v1/markets

List covered markets with forecast status.

Common params:
- `topic`
- `max_forecast_age` (hours, default `48`)
- `sort` (`updated_desc`, `volume_desc`, `signal_count_desc`)
- `cursor`
- `limit` (`1-50`, default `20`)

### GET /api/v1/markets/mispriced

Markets where Quotient odds diverge from market odds.

Common params:
- `min_spread` (`0-1`, default `0.05`)
- `max_forecast_age` (hours, default `48`)
- `sort` (`spread_desc`, `spread_asc`, `volume_desc`, `updated_desc`)
- `cursor`
- `limit` (`1-50`, default `20`)

### GET /api/v1/markets/{slug}/intelligence

Full intelligence briefing for one market:
- `quotient_odds`, `market_odds`
- `bluf`
- `key_drivers` with citations
- `signals`
- `sentiment`
### GET /api/v1/markets/{slug}/signals

Paginated signals for one market.

Common params:
- `cursor`
- `limit` (`1-50`, default `10`)

### GET /api/v1/markets/lookup

Batch lookup by identifier.

Use exactly one:
- `slugs` (comma-separated, max 10), or
- `condition_ids` (comma-separated, max 10)

## Common Error Codes

- `401 invalid_api_key`
- `401 gateway_required`
- `402 payment_required`
- `403 insufficient_credits`
- `404 invalid_market` / `404 not_found`
- `422 invalid_request`
- `422 invalid_cursor`
- `429 rate_limited`
