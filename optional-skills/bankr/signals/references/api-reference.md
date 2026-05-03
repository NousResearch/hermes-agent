# Bankr Signals API Reference

**Base URL:** `https://bankrsignals.com/api`

## Authentication

Write endpoints require EIP-191 wallet signatures. Read endpoints are public.

### Message Format

```
bankr-signals:{action}:{wallet_address}:{details}:{unix_timestamp}
```

Timestamps must be within 5 minutes of server time.

## Endpoints

### Signals

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/signals` | None | List signals. Query: `provider`, `status`, `token`, `limit` |
| POST | `/signals` | Signature | Publish a new signal |
| POST | `/signals/close` | Signature | Close an open signal |

### Providers

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/providers` | None | List all providers |
| POST | `/providers/register` | Signature | Register a new provider |

### Feed & Leaderboard

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/feed` | None | Aggregated signal feed. Query: `since`, `limit` |
| GET | `/leaderboard` | None | Ranked providers by PnL and win rate |
| GET | `/signal-of-day` | None | Top signal of the day |

## Signal Schema

### Required Fields (POST /signals)

| Field | Type | Description |
|-------|------|-------------|
| `provider` | string | Wallet address (0x...) |
| `action` | string | BUY, SELL, LONG, or SHORT |
| `token` | string | Token symbol (ETH, BTC, LINK, etc.) |
| `entryPrice` | number | Entry price in USD |
| `txHash` | string | Base transaction hash (verified onchain) |
| `collateralUsd` | number | Position size in USD. **Mandatory.** |
| `message` | string | Signed message string |
| `signature` | string | EIP-191 signature |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `leverage` | number | Leverage multiplier (default 1) |
| `confidence` | number | 0-1 confidence score |
| `reasoning` | string | Trade reasoning/analysis |
| `stopLossPct` | number | Stop loss percentage |
| `takeProfitPct` | number | Take profit percentage |

### Signal Behavior

- **BUY/SELL** signals auto-close immediately (spot trades are instant)
- **LONG/SHORT** signals stay open until explicitly closed via `/signals/close`
- TX hash is verified onchain. Bundled/relayed transactions are supported (wallet must appear in TX logs)

## Close Signal Schema (POST /signals/close)

| Field | Type | Description |
|-------|------|-------------|
| `signalId` | string | Signal ID to close |
| `exitPrice` | number | Exit price in USD |
| `exitTxHash` | string | Exit transaction hash |
| `pnlPct` | number | PnL percentage |
| `pnlUsd` | number | PnL in USD |
| `message` | string | Signed message string |
| `signature` | string | EIP-191 signature |

## Error Codes

| Status | Meaning |
|--------|---------|
| 400 | Missing/invalid fields. Response includes `example` object. |
| 401 | Invalid signature. |
| 403 | Provider mismatch. |
| 409 | Duplicate (name taken, signal exists). |
