---
name: hyperliquid
description: Query Hyperliquid market and account data - perp dexs, perp/spot market contexts, candles, funding history, L2 books, perp state, spot balances, fills, historical orders, trade review, and normalized market-data export. Uses the public info endpoint only and needs no API key.
version: 0.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Hyperliquid, Blockchain, Crypto, Trading, Perpetuals, Spot, DeFi]
    related_skills: []
---

# Hyperliquid Skill

Query Hyperliquid market data and user account history through the public
`/info` endpoint.

12 commands: dexs, perp markets, spot markets, candle history, funding history,
L2 books, perp state, spot balances, fills, historical orders, trade review,
and normalized market-data export.

No API key needed. Uses only Python standard library (`urllib`, `json`,
`argparse`).

---

## When to Use

- User asks for Hyperliquid perp or spot market data
- User wants historical candles for a Hyperliquid market
- User wants current funding, open interest, or 24h notional volume
- User wants to inspect an address's perp positions, spot balances, fills, or historical orders
- User wants a post-trade review using fills plus surrounding market context
- User wants to inspect builder-deployed perp dexs or HIP-3 markets

---

## Prerequisites

The helper script uses only Python standard library.
No external packages or API keys are required.
It automatically reads `~/.hermes/.env` for `HYPERLIQUID_API_URL` and
`HYPERLIQUID_USER_ADDRESS`. A project `.env` in the current working directory
is treated as a dev fallback when present.

Default API base:

```bash
https://api.hyperliquid.xyz
```

Optional testnet or custom override:

```bash
export HYPERLIQUID_API_URL="https://api.hyperliquid-testnet.xyz"
# or save it in ~/.hermes/.env
```

Optional default account address:

```bash
export HYPERLIQUID_USER_ADDRESS="0x0000000000000000000000000000000000000000"
# or save it in ~/.hermes/.env
```

Helper script path:

```bash
~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py
```

---

## Quick Reference

```bash
python3 hyperliquid_client.py dexs
python3 hyperliquid_client.py markets [--dex DEX] [--limit N] [--sort volume|oi|funding_abs|change_abs|name]
python3 hyperliquid_client.py spots [--limit N]
python3 hyperliquid_client.py candles <coin> [--interval 1h] [--hours 24] [--limit N]
python3 hyperliquid_client.py funding <coin> [--hours 72] [--limit N]
python3 hyperliquid_client.py l2 <coin> [--levels N]
python3 hyperliquid_client.py state [address] [--dex DEX]
python3 hyperliquid_client.py spot-balances [address] [--limit N]
python3 hyperliquid_client.py fills [address] [--hours N] [--limit N] [--aggregate-by-time]
python3 hyperliquid_client.py orders [address] [--limit N]
python3 hyperliquid_client.py review [address] [--coin COIN] [--hours N] [--fills N]
python3 hyperliquid_client.py export <coin> [--interval 1h] [--hours N] [--output PATH]
```

Add `--json` to any command for structured output.
For `state`, `spot-balances`, `fills`, `orders`, and `review`, the address is optional if `HYPERLIQUID_USER_ADDRESS` is set.

---

## Procedure

### 0. Setup Check

```bash
python3 --version

# Optional: switch to testnet
export HYPERLIQUID_API_URL="https://api.hyperliquid-testnet.xyz"

# Optional: set a default address for account-level commands
export HYPERLIQUID_USER_ADDRESS="0x0000000000000000000000000000000000000000"

# Confirm connectivity by listing top perp markets
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 5
```

### 1. Discover DEXs and Markets

Use `dexs` to inspect the first perp dex plus any builder-deployed perp dexs.
Use `markets` to inspect mark price, change, funding, open interest, and 24h
notional volume. Use `spots` for spot pairs.

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py dexs

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 15 --sort volume

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --dex mydex --limit 15

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spots --limit 15
```

Tips:
- `--dex` is only for perp endpoints; omit it for the first perp dex.
- Spot pairs may appear as `PURR/USDC` or internal aliases like `@107`.
- For HIP-3 markets, coin strings may include a dex prefix such as `mydex:BTC`.

### 2. Pull Historical Market Data

Use `candles` for OHLCV snapshots and `funding` for historical funding data.
This is the best starting point for backtest prototypes and trade review.

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  candles BTC --interval 1h --hours 72 --limit 48

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  funding BTC --hours 168 --limit 30
```

Notes:
- The info endpoint paginates time-range endpoints. If you need more than one
  response window, repeat the query with a later `startTime`.
- This helper is for interactive inspection. If you later build a real
  backtester, store the returned data in local files or a database.

### 3. Inspect Live Microstructure

Use `l2` to inspect the current order book around a market.

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  l2 BTC --levels 10
```

This is useful when the user asks:
- whether the book looks thin
- where near-term liquidity sits
- whether a large order may move the market

### 4. Review a User's Account State

Use `state` for perp positions and `spot-balances` for spot inventory.

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  state 0x0000000000000000000000000000000000000000

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  state

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  spot-balances
```

Use these when the user asks:
- "How are my positions?"
- "What am I holding?"
- "How much is withdrawable?"

### 5. Review Fills and Orders

Use `fills` and `orders` for recent execution history.

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  fills 0x0000000000000000000000000000000000000000 --hours 72 --limit 25

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  orders --limit 25
```

### 6. Generate A Lightweight Trade Review

Use `review` to combine recent fills with candle and funding context for each
traded coin.

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review 0x0000000000000000000000000000000000000000 --hours 72 --fills 50

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  review --coin BTC --hours 168
```

The review reports:
- realized PnL, fees, and net after fees
- win/loss counts
- coin-by-coin breakdowns
- market trend and average funding for each traded perp
- heuristics like fee drag, concentration, and counter-trend losses

Use it as a first-pass reviewer, not a final judge. It works best when paired
with the raw `fills`, `orders`, `candles`, and `funding` commands.

For deeper post-trade review:
1. Start with `review` to identify problem coins or windows.
2. Pull recent fills for the address.
3. Pull recent orders for the same period.
4. Pull `candles` and `funding` for each traded coin over the relevant window.
5. Judge decision quality separately from outcome quality.

Suggested review format:
- thesis at entry
- market context
- execution quality
- sizing quality
- exit quality
- what to repeat
- what to stop doing

### 7. Export A Reusable Market Dataset

Use `export` to write normalized candles plus funding history to a JSON file.
This is the clean handoff point for a future local backtester.

```bash
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 1h --hours 168 --output ./btc-1h-7d.json

python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  export BTC --interval 15m --hours 72 --end-time-ms 1760000000000
```

The export file contains:
- schema version
- source metadata
- exact time window
- normalized candle rows
- normalized funding rows
- summary stats like price change and average funding

Use `--end-time-ms` when you want reproducible windows for comparisons,
debugging, or future backtests.

---

## Pitfalls

- Public info endpoints are rate-limited. Large historical queries can require
  multiple calls and may only return a capped window of rows.
- `fills --hours ...` uses `userFillsByTime`, which only exposes a recent
  rolling history window.
- `historicalOrders` returns the most recent orders only; it is not a full
  archive export.
- The `review` command is heuristic. It cannot reconstruct exact intent, order
  placement quality, or true slippage from fills alone.
- The `export` command writes a normalized dataset contract, not a full
  backtest engine. You still need your own fill/slippage model later.
- Spot aliases like `@107` are valid market identifiers even if the app UI
  shows a friendlier name.
- Order-book data from `l2` is a point-in-time snapshot, not a time series.
- Candle/funding history is useful for review and prototyping, but it is not a
  full execution simulator. Be conservative about slippage assumptions.

---

## Verification

```bash
# Should print top Hyperliquid perp markets by 24h notional volume
python3 ~/.hermes/skills/blockchain/hyperliquid/scripts/hyperliquid_client.py \
  markets --limit 5
```
