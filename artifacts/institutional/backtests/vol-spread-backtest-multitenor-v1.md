# BTC Vol Spread Backtest — Multi-Tenor Research Pack

---



---

**Evidence status:** `SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE`

---



---

Research evidence only. All modeled results use screen-only historical marks and synthetic fills; not executable economics.

---



---

# BTC Vol Spread Backtest — 1d-spread-threshold-5

**Evidence status:** `SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE`

This is research evidence, not executable economics. The engine uses historical monitor snapshots, synthetic fills, and fixed cost assumptions; it does not prove that any spread was tradable.

## Controls

- No future leakage: the strategy receives only prior snapshots at each decision point.
- Chronology is enforced; out-of-order snapshots fail instead of being silently sorted.
- Synthetic fills are explicitly labeled and are not quotes, RFQs, or executions.

## Summary

- Tenor: `1d`
- Snapshot count: 19
- Trade count: 17
- Gross PnL: $743,350.00
- Max drawdown: $27,650.00
- Win rate: 64.7%
- Minimum evidence gate: insufficient-history

## Assumptions

- execution: screen-only synthetic fills; not executable economics
- anti_lookahead: DecisionFrame excludes next_* future fields; ExecutionResolver resolves synthetic exits after decision time
- source: point-in-time historical spread snapshots
- spread_field: spread_1d_vol_pts
- threshold_vol_pts: 5.0
- notional_vega: 10000.0
- cost_per_trade: 250.0

## Trades

| # | Side | Entry | Exit | Net PnL | Signal |
|---:|---|---:|---:|---:|---|
| 1 | short_spread | 5.71 | -11.97 | $176,550.00 | IBIT rich / Deribit cheap |
| 2 | long_spread | -11.97 | -14.71 | $-27,650.00 | IBIT cheap / Deribit rich |
| 3 | long_spread | -14.71 | -11.52 | $31,650.00 | IBIT cheap / Deribit rich |
| 4 | long_spread | -11.52 | -12.99 | $-14,950.00 | IBIT cheap / Deribit rich |
| 5 | long_spread | -12.99 | -13.13 | $-1,650.00 | IBIT cheap / Deribit rich |
| 6 | long_spread | -13.13 | -12.86 | $2,450.00 | IBIT cheap / Deribit rich |
| 7 | long_spread | -12.86 | -7.86 | $49,750.00 | IBIT cheap / Deribit rich |
| 8 | long_spread | -7.86 | 13.83 | $216,650.00 | IBIT cheap / Deribit rich |
| 9 | short_spread | 13.83 | 13.15 | $6,550.00 | IBIT rich / Deribit cheap |
| 10 | short_spread | 13.15 | 6.43 | $66,950.00 | IBIT rich / Deribit cheap |
| 11 | short_spread | 6.43 | 6.43 | $-250.00 | IBIT rich / Deribit cheap |
| 12 | short_spread | 6.43 | 6.36 | $450.00 | IBIT rich / Deribit cheap |
| 13 | short_spread | 6.36 | 8.96 | $-26,250.00 | IBIT rich / Deribit cheap |
| 14 | short_spread | 8.96 | -16.97 | $259,050.00 | IBIT rich / Deribit cheap |
| 15 | long_spread | -16.97 | -15.16 | $17,850.00 | IBIT cheap / Deribit rich |
| 16 | long_spread | -15.16 | -17.69 | $-25,550.00 | IBIT cheap / Deribit rich |
| 17 | long_spread | -17.69 | -16.49 | $11,750.00 | IBIT cheap / Deribit rich |


---

# BTC Vol Spread Backtest — 7d-spread-threshold-5

**Evidence status:** `SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE`

This is research evidence, not executable economics. The engine uses historical monitor snapshots, synthetic fills, and fixed cost assumptions; it does not prove that any spread was tradable.

## Controls

- No future leakage: the strategy receives only prior snapshots at each decision point.
- Chronology is enforced; out-of-order snapshots fail instead of being silently sorted.
- Synthetic fills are explicitly labeled and are not quotes, RFQs, or executions.

## Summary

- Tenor: `7d`
- Snapshot count: 19
- Trade count: 2
- Gross PnL: $54,000.00
- Max drawdown: $0.00
- Win rate: 100.0%
- Minimum evidence gate: insufficient-history

## Assumptions

- execution: screen-only synthetic fills; not executable economics
- anti_lookahead: DecisionFrame excludes next_* future fields; ExecutionResolver resolves synthetic exits after decision time
- source: point-in-time historical spread snapshots
- spread_field: spread_7d_vol_pts
- threshold_vol_pts: 5.0
- notional_vega: 10000.0
- cost_per_trade: 250.0

## Trades

| # | Side | Entry | Exit | Net PnL | Signal |
|---:|---|---:|---:|---:|---|
| 1 | short_spread | 7.81 | 7.74 | $450.00 | IBIT rich / Deribit cheap |
| 2 | short_spread | 7.74 | 2.36 | $53,550.00 | IBIT rich / Deribit cheap |


---

# BTC Vol Spread Backtest — 30d-spread-threshold-5

**Evidence status:** `SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE`

This is research evidence, not executable economics. The engine uses historical monitor snapshots, synthetic fills, and fixed cost assumptions; it does not prove that any spread was tradable.

## Controls

- No future leakage: the strategy receives only prior snapshots at each decision point.
- Chronology is enforced; out-of-order snapshots fail instead of being silently sorted.
- Synthetic fills are explicitly labeled and are not quotes, RFQs, or executions.

## Summary

- Tenor: `30d`
- Snapshot count: 19
- Trade count: 2
- Gross PnL: $48,300.00
- Max drawdown: $150.00
- Win rate: 50.0%
- Minimum evidence gate: insufficient-history

## Assumptions

- execution: screen-only synthetic fills; not executable economics
- anti_lookahead: DecisionFrame excludes next_* future fields; ExecutionResolver resolves synthetic exits after decision time
- source: point-in-time historical spread snapshots
- spread_field: spread_30d_vol_pts
- threshold_vol_pts: 5.0
- notional_vega: 10000.0
- cost_per_trade: 250.0

## Trades

| # | Side | Entry | Exit | Net PnL | Signal |
|---:|---|---:|---:|---:|---|
| 1 | short_spread | 5.87 | 5.86 | $-150.00 | IBIT rich / Deribit cheap |
| 2 | short_spread | 5.86 | 0.99 | $48,450.00 | IBIT rich / Deribit cheap |


---

## Robustness Metrics

- Sample gate pass count: 0 / 3
- Sample gate ready: False
- Minimum gates: 30 snapshots and 20 synthetic trades per tenor
- Max observed sample: 19 snapshots / 17 synthetic trades
- Tenors with synthetic trades: 1d, 7d, 30d
- Best gross PnL tenor: 1d
- Worst gross PnL tenor: 30d
- Cost sensitivity: $250.00 explicit cost + 0.00 vol pts slippage = $250.00 effective cost/trade
- Control: Synthetic fills only; cost sensitivity is illustrative and not executable economics.
