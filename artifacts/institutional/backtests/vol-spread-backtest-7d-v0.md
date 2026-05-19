# BTC Vol Spread Backtest — 7d-spread-threshold-5

**Evidence status:** `SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE`

This is research evidence, not executable economics. The engine uses historical monitor snapshots, synthetic fills, and fixed cost assumptions; it does not prove that any spread was tradable.

## Controls

- No future leakage: the strategy receives only prior snapshots at each decision point.
- Chronology is enforced; out-of-order snapshots fail instead of being silently sorted.
- Synthetic fills are explicitly labeled and are not quotes, RFQs, or executions.

## Summary

- Tenor: `7d`
- Snapshot count: 14
- Trade count: 2
- Gross PnL: $54,000.00
- Max drawdown: $0.00
- Win rate: 100.0%

## Assumptions

- execution: screen-only synthetic fills; not executable economics
- source: monitor-derived historical spread snapshots
- threshold_vol_pts: 5.0
- notional_vega: 10000.0
- cost_per_trade: 250.0

## Trades

| # | Side | Entry | Exit | Net PnL | Signal |
|---:|---|---:|---:|---:|---|
| 1 | short_spread | 7.81 | 7.74 | $450.00 | IBIT rich / Deribit cheap |
| 2 | short_spread | 7.74 | 2.36 | $53,550.00 | IBIT rich / Deribit cheap |
