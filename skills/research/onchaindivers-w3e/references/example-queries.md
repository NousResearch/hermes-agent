# OnchainDivers / W3E Example Queries

All examples below are **sanitized**. Replace placeholders with your own env vars at runtime. Do not paste real credentials into repo files, chat logs, or commits.

## Bearer API pattern

```bash
export ONCHAINDIVERS_API_KEY='***'
```

### Polymarket: access smoke test

```bash
curl -sS \
  -H "Authorization: Bearer $ONCHAINDIVERS_API_KEY" \
  -H 'Content-Type: application/json' \
  https://db-access-test.onchaindivers.com/v1/polymarket/query \
  -d '{"query":"SELECT count() AS rows FROM raw_event_meta"}'
```

### Solana: access smoke test

```bash
curl -sS \
  -H "Authorization: Bearer $ONCHAINDIVERS_API_KEY" \
  -H 'Content-Type: application/json' \
  https://db-access-test.onchaindivers.com/v1/solana/query \
  -d '{"query":"SELECT count() AS rows FROM solana_blocks"}'
```

### Hyperliquid: access smoke test

```bash
curl -sS \
  -H "Authorization: Bearer $ONCHAINDIVERS_API_KEY" \
  -H 'Content-Type: application/json' \
  https://db-access-test.onchaindivers.com/v1/hyperliquid/query \
  -d '{"query":"SELECT count() AS rows FROM agg_fulfilled_order"}'
```

## Direct ClickHouse HTTP pattern

Use readonly users only.

```bash
export W3E_HL_CH_USER='<readonly-user>'
export W3E_HL_CH_PASS='<readonly-pass>'
export W3E_HL_CH_HOST='example.host'
export W3E_HL_CH_PORT='28123'
```

```bash
curl -sS \
  --user "$W3E_HL_CH_USER:$W3E_HL_CH_PASS" \
  "http://$W3E_HL_CH_HOST:$W3E_HL_CH_PORT/?database=hyperliquid&default_format=JSONEachRow" \
  --data-binary 'SELECT count() AS rows FROM agg_fulfilled_order'
```

## Schema discovery

If supported by the endpoint, try:

```sql
SELECT name
FROM system.tables
WHERE database = currentDatabase()
ORDER BY name
```

If that fails through bearer mode, fall back to known-good tables and docs.

## Polymarket: discover BTC monthly candidates

```sql
SELECT
  event_id,
  market_id,
  event_title,
  market_question,
  slug,
  end_date,
  active,
  closed
FROM raw_market_meta
WHERE lower(event_title) LIKE '%bitcoin%'
   OR lower(event_title) LIKE '%btc%'
   OR lower(market_question) LIKE '%bitcoin%'
   OR lower(market_question) LIKE '%btc%'
ORDER BY end_date DESC
LIMIT 200
```

Then narrow the candidate set by month wording before touching the fill table.

## Polymarket: pull fills after market discovery

The exact join keys can vary across warehouse revisions, so inspect columns first if needed. Keep the first fill query narrow:

```sql
SELECT *
FROM polymarket_order_filled_v3
WHERE market_id IN ('<candidate-market-id-1>', '<candidate-market-id-2>')
ORDER BY timestamp DESC
LIMIT 200
```



## Hyperliquid: funding or fill research

Start with cheap samples:

```sql
SELECT *
FROM dxn_funding
ORDER BY time DESC
LIMIT 50
```

```sql
SELECT *
FROM agg_fulfilled_order
ORDER BY time DESC
LIMIT 50
```

## Solana: general block-level sanity check

```sql
SELECT *
FROM solana_blocks
ORDER BY slot DESC
LIMIT 10
```

## Notes

- Prefer `SELECT` / `WITH` only
- Use `LIMIT` aggressively during discovery
- Record schema mismatches between docs and live tables in your notes
- Keep analysis grounded in real fills and metadata, not just conceptual narratives
