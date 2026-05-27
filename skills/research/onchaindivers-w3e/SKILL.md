---
name: onchaindivers-w3e
description: "Use when querying OnchainDivers / W3E hosted warehouse data for Polymarket, Hyperliquid, or Solana research without exposing credentials."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [research, clickhouse, onchaindivers, w3e, polymarket, hyperliquid, solana]
    related_skills: [polymarket, arxiv]
---

# OnchainDivers / W3E Research Access

## Overview

Use this skill when the user wants practical research on **Polymarket**, **Hyperliquid**, or **Solana** using the hosted OnchainDivers / W3E data warehouse instead of only public REST endpoints.

This workflow is optimized for:
- quick access validation,
- safe handling of bearer keys and readonly ClickHouse credentials,
- schema discovery,
- writing real SQL against the warehouse,
- and staying honest when docs/examples diverge from the actual tables.

The main lesson: **treat the docs as a map, not as ground truth**. Verify real table names and row availability with live queries before building analysis on top.

## When to Use

- User already has OnchainDivers / W3E access and wants live data checks
- User wants research on Polymarket fills, market metadata, or event metadata
- User wants Hyperliquid fills / funding / wallet state from a warehouse
- User wants Solana research from curated tables instead of raw RPC scraping
- You need repeatable SQL-backed evidence, not hand-wavy market commentary

Do **not** use this skill when:
- public Polymarket REST data is enough,
- the user wants trading execution,
- or credentials are unavailable and no public fallback exists.

## Safe Credential Handling

Never commit or print real secrets into repo files, chat summaries, skills, or Git history.

Preferred pattern:

```bash
export ONCHAINDIVERS_API_KEY='<redacted>'
export W3E_HL_CH_USER='<redacted>'
export W3E_HL_CH_PASS='<redacted>'
```

Use placeholders in docs and examples only.

For bearer API calls, prefer ephemeral shell variables over inline literals:

```bash
curl -sS \
  -H "Authorization: Bearer $ONCHAINDIVERS_API_KEY" \
  -H 'Content-Type: application/json' \
  https://db-access-test.onchaindivers.com/v1/polymarket/query \
  -d '{"query":"SELECT 1"}'
```

For direct ClickHouse HTTP access, use readonly users only.

## Practical Workflow

### 1) Confirm access first

Do not assume stored creds are valid just because local code mentions W3E.

Start with tiny verification queries:

```sql
SELECT count() AS rows FROM raw_event_meta
SELECT count() AS rows FROM raw_market_meta
SELECT count() AS rows FROM polymarket_order_filled_v3
```

For Hyperliquid:

```sql
SELECT count() AS rows FROM agg_fulfilled_order
```

For Solana:

```sql
SELECT count() AS rows FROM solana_blocks
```

If auth is flaky, re-check whether you are using the intended endpoint, header format, and freshest key.

### 2) Discover schema from the live system

Docs may show simplified names such as `markets`, `tokens`, `fills`, or `fundings`, while the live warehouse can expose different physical tables.

Prefer:

```sql
SELECT name
FROM system.tables
WHERE database = currentDatabase()
ORDER BY name
```

If that fails through the bearer endpoint, fall back to:
- docs pages for conceptual orientation,
- known good tables from prior validation,
- or direct ClickHouse access where available.

### 3) Use the warehouse in three layers

For Polymarket, the most useful pattern is:
- `raw_event_meta` → event-level framing
- `raw_market_meta` → market-level metadata
- `polymarket_order_filled_v3` → real fills / executed prices

This is the right path when validating claims about overpriced or underpriced volatility, monthly BTC structures, or entry timing.

### 4) Build from metadata to fills

Do not start with abstract strategy talk. First identify the actual markets, then join to fills.

Good sequence:
1. find candidate BTC monthly markets in metadata,
2. inspect titles / slugs / end dates,
3. only then pull fills for those markets,
4. compare executed prices and realized outcomes.

## Known Reality-Check on Table Names

Based on practical use, these names are the reliable starting points.

### Polymarket

Use these first:
- `raw_event_meta`
- `raw_market_meta`
- `polymarket_order_filled_v3`

### Hyperliquid

Useful starting points:
- `agg_fulfilled_order`
- `raw_node_fills_by_block`
- `view_perpetual_wallet`
- `view_wallet_position`
- `dxn_funding`
- `perp_asset_meta`
- `perp_asset_stats`
- `spot_asset_meta`
- `spot_pair_meta`

### Solana

Useful starting points:
- `solana_blocks`
- `tx_timestamps`
- `jito_tips`
- `pumpfun_token_creation`
- `pumpfun_all_swaps`
- `pumpswap_all_swaps`
- `raydium_all_swaps`
- `meteora_swaps`
- `meteora_dynamic_bonding_swaps`
- `pfamm_migrations`
- `max_caps`

## First Polymarket Query Pattern

When the user wants **monthly BTC markets**, start with metadata discovery, not fills.

Example shape:

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

Then narrow to monthly-expiry style wording such as:
- `end of month`
- `by may 31`
- `in may`
- `monthly`
- or explicit month/year references in titles and slugs.

After that, pull fills from `polymarket_order_filled_v3` only for the shortlisted markets.

## Query Style Guidelines

- Prefer `SELECT` and `WITH` only
- Keep first queries cheap and inspectable
- Use `LIMIT` during discovery
- Add explicit `ORDER BY` so outputs are reproducible
- Save intermediate candidate IDs before running heavy fill queries
- When comparing strategies, use **executed fill prices**, not just quoted probabilities

## Practical PM Microstructure Notes

These points came from practical trading discussion and are worth preserving as hypotheses / implementation constraints, not as proven alpha.

### 1) Entry quality can destroy most of the edge

A common failure mode is assuming the theoretical edge survives market-order entry. In practice, shallow books can move the fill from something like `0.70` to `0.73`, which can compress a nominal `~5%` edge down to `~1–2%`.

Implication: when evaluating a PM strategy, model **actual reachable fills**, not just top-of-book snapshots or event probabilities.

### 2) Marketable limit orders are often better than pure market orders

For these setups, speed is often less important than preserving price.

Useful execution pattern:
- place a **marketable limit** at the worst acceptable price,
- let the instantly available size fill,
- leave the rest resting in the book,
- monitor whether the original alpha still exists,
- cancel and unwind filled inventory if the delta / thesis moves away.

This is especially relevant around balanced prices like `0.45–0.55`, where preserving a few cents matters a lot.

### 3) Partial-maker execution can help fees and realized entry

A marketable limit can behave better than a blunt market order because:
- part of the order may execute immediately,
- part may rest as maker,
- effective fees can be lower,
- realized entry may stay closer to the intended threshold.

### 4) Book-update streams may contain duplicates

When reconstructing order-book or event streams, expect duplicate updates by hash and/or server timestamp. Deduplication is part of the research problem, not an optional cleanup step.

### 5) Monitoring across many BTC expiries/time windows increases opportunity count

One research idea worth checking empirically: monitoring BTC markets across many expiries / time windows can produce a larger event set, making systematic placement through the book more realistic than judging the opportunity from one isolated market.

Do **not** treat this as validated alpha without fill-level evidence.

### 6) Narrow-loss-range structures must be tested against real fills

A proposed framing from the notes: structure straddle / volatility exposure so that loss is concentrated in a narrow central band, then rebalance the rest. This kind of idea must be tested against:
- actual PM fills,
- available size,
- duplicate-book noise,
- and real execution latency.

If you cannot show the realized fills, you do not yet know whether the structure works.

## Common Pitfalls

1. **Trusting docs over reality.**
   If docs say `fills` but the live table is `agg_fulfilled_order`, trust the live system.

2. **Assuming a local codebase means current access works.**
   Old helper scripts can survive long after creds rotate.

3. **Leaking keys in shell history or commits.**
   Use env vars and placeholders only.

4. **Jumping directly into strategy conclusions.**
   First prove the market set and entry prices you are talking about actually existed.

5. **Using abstract volatility narratives instead of actual fill data.**
   For Polymarket research, the fill table is where the truth starts.

## Verification Checklist

- [ ] Access verified with a tiny live query
- [ ] No secrets printed or committed
- [ ] Physical table names confirmed from live system or previously verified evidence
- [ ] Metadata query run before fill query
- [ ] Analysis grounded in actual rows, not only docs or intuition
- [ ] If docs and schema disagree, the discrepancy is called out explicitly

## Supporting Reference

See `references/example-queries.md` for sanitized curl + SQL snippets covering Polymarket, Hyperliquid, and Solana.