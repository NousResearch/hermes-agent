# BTC Vol Desk Monitor — 2026-05-18 16:30:34 CDT

**Run ID:** `btcvol-20260518-163034`

**Evidence standard:** all public/API screen outputs are `screen-only` unless linked to quote or trade records.

## BTC / ETF Reference

- BTC spot/reference: 77,457.06
- BTC per ETF share: 0.000567911859

## Deribit ATM IV Term Structure

| Expiry | DTE | Symbol | IV mark |
|---|---:|---|---:|
| 2026-05-19 | 1 | BTC-19MAY26-77000-P | 27.64% |
| 2026-05-20 | 2 | BTC-20MAY26-77000-P | 32.42% |
| 2026-05-21 | 3 | BTC-21MAY26-77000-P | 34.06% |
| 2026-05-22 | 4 | BTC-22MAY26-77000-C | 35.22% |
| 2026-05-29 | 11 | BTC-29MAY26-77000-P | 37.25% |
| 2026-06-05 | 18 | BTC-5JUN26-77000-C | 37.46% |
| 2026-06-26 | 39 | BTC-26JUN26-78000-P | 37.67% |
| 2026-07-31 | 74 | BTC-31JUL26-77000-C | 38.72% |
| 2026-09-25 | 130 | BTC-25SEP26-78000-C | 40.42% |
| 2026-12-25 | 221 | BTC-25DEC26-78000-P | 43.79% |
| 2027-03-26 | 312 | BTC-26MAR27-78000-P | 44.92% |

## IBIT / ETF ATM IV Term Structure

| Expiry | DTE | Symbol | IV mark |
|---|---:|---|---:|
| 2026-05-18 | 0 | IBIT-2026-05-18-43.5-C | 11.15% |
| 2026-05-20 | 2 | IBIT-2026-05-20-43.5-C | 40.81% |
| 2026-05-22 | 4 | IBIT-2026-05-22-43.5-C | 40.68% |
| 2026-05-26 | 8 | IBIT-2026-05-26-43.5-C | 35.74% |
| 2026-05-27 | 9 | IBIT-2026-05-27-43.5-C | 36.36% |
| 2026-05-29 | 11 | IBIT-2026-05-29-43.5-C | 38.52% |

## CME / Databento BTC Options Status

**Evidence label:** Databento CME rows are `screen_only_not_executable`; they are licensed vendor marks, not executable quotes.

| Expiry | DTE | Symbol | Bid | Ask | Confidence |
|---|---:|---|---:|---:|---|
| 2026-05-22 | 4 | UD:B4: VT 2614591 | 9.00 | 45.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P55000 | 2.00 | 70.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P55500 | 4.00 | 70.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P56000 | 6.00 | 75.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P56500 | 8.00 | 75.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P57000 | 11.00 | 80.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P57500 | 13.00 | 80.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P58000 | 16.00 | 85.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P58500 | 20.00 | 90.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P59000 | 23.00 | 90.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P59500 | 25.00 | 95.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P60000 | 30.00 | 100.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P60500 | 35.00 | 105.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P61000 | 40.00 | 115.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P61500 | 45.00 | 120.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P62000 | 55.00 | 125.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P62500 | 60.00 | 135.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P63000 | 70.00 | 140.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P63500 | 80.00 | 150.00 | screen_only_not_executable |
| 2026-05-29 | 11 | BTCK6 P64000 | 85.00 | 160.00 | screen_only_not_executable |

## Dislocation Board

| Candidate | Gross IV diff | Confidence | Next action |
|---|---:|---|---|
| IBIT 0D ATM vs Deribit 1D ATM | -16.49 vol pts | screen-only | quote review |
| IBIT 2D ATM vs Deribit 2D ATM | 8.39 vol pts | screen-only | quote review |
| IBIT 4D ATM vs Deribit 4D ATM | 5.46 vol pts | screen-only | quote review |
| IBIT 8D ATM vs Deribit 11D ATM | -1.51 vol pts | screen-only | watch |
| IBIT 9D ATM vs Deribit 11D ATM | -0.89 vol pts | screen-only | watch |
| IBIT 11D ATM vs Deribit 11D ATM | 1.27 vol pts | screen-only | watch |

## Quality Warnings

- Deribit normalized rows: 888
- iShares BTC/share current fetch or parse failed: iShares holdings endpoint returned HTML instead of holdings CSV
- iShares BTC/share fallback used latest cached official CSV: artifacts/institutional/data/raw/btcvol-20260515-183316/ishares_ibit_holdings.csv
- IBIT holdings source is stale-cache; use for continuity only until current iShares CSV recovers
- IBIT BTC/share independent market-implied cross-check diff: 1.04%
- IBIT normalized rows: 402
- CME Databento normalized rows: 1708
- CME Databento BBO rows: 5000
