# Licensed Historical Source Acquisition Package v1

**Evidence status:** `SCREEN-ONLY · LICENSED SOURCE ACQUISITION PACKAGE · NOT EXECUTABLE`

This package removes ambiguity about what is still needed. It does not promote current public/screen captures to licensed historical readiness.

## Required source groups

### IBIT options history
- Required fields: `available_ts, expiration, strike, option_type, bid, ask, volume, open_interest, source_ref`
- Accepted formats: `csv`, `jsonl`, `parquet`
- Provider targets: OPRA historical options via broker/vendor export; ORATS/OptionMetrics/Polygon paid historical options if license permits replay/diligence use
- Status: `not-procured-for-licensed-historical-readiness`
- Next action: Request/export licensed historical file, then run build-source-intake-entry and validate-source-intake.

### Deribit options history
- Required fields: `available_ts, instrument_name, underlying_price, bid_iv, ask_iv, mark_iv, open_interest, source_ref`
- Accepted formats: `csv`, `jsonl`, `parquet`
- Provider targets: Deribit historical/replay export with timestamped option book/IV fields; Kaiko/Amberdata/Tardis licensed Deribit options history
- Status: `not-procured-for-licensed-historical-readiness`
- Next action: Request/export licensed historical file, then run build-source-intake-entry and validate-source-intake.

### CME Bitcoin options history
- Required fields: `available_ts, symbol, expiration, strike, option_type, bid, ask, settlement, source_ref`
- Accepted formats: `csv`, `jsonl`, `parquet`
- Provider targets: Databento CME options history export; CME DataMine/broker historical CME Bitcoin options export
- Status: `not-procured-for-licensed-historical-readiness`
- Next action: Request/export licensed historical file, then run build-source-intake-entry and validate-source-intake.

### BTC reference history
- Required fields: `available_ts, btc_usd, venue_or_index, source_ref`
- Accepted formats: `csv`, `jsonl`, `parquet`
- Provider targets: CF Benchmarks/CME CF BRR/BRTI licensed reference history; Kaiko/Coin Metrics/Bloomberg BTC reference index export
- Status: `not-procured-for-licensed-historical-readiness`
- Next action: Request/export licensed historical file, then run build-source-intake-entry and validate-source-intake.

### IBIT holdings history
- Required fields: `available_ts, btc_per_share, shares_outstanding, fund_assets, source_ref`
- Accepted formats: `csv`, `jsonl`, `parquet`
- Provider targets: iShares daily holdings/NAV archive under approved use; Bloomberg/refinitiv/FactSet ETF holdings history export
- Status: `not-procured-for-licensed-historical-readiness`
- Next action: Request/export licensed historical file, then run build-source-intake-entry and validate-source-intake.

### Rates and fee curves
- Required fields: `available_ts, tenor, rate, borrow_or_fee, source_ref`
- Accepted formats: `csv`, `jsonl`, `parquet`
- Provider targets: FRED/SOFR/Treasury time series with usage terms archived; Broker borrow/financing/fee curve export when available
- Status: `not-procured-for-licensed-historical-readiness`
- Next action: Request/export licensed historical file, then run build-source-intake-entry and validate-source-intake.

## Operator commands after a licensed file is obtained

```bash
PYTHONPATH=. python3 -m institutional_btc_vol.cli build-source-intake-entry <raw-file> --source-group "<source group>" --provenance "<provider/license>" --license-label "<license label>" --source-ref "<vendor export id>" --output artifacts/institutional/historical/source_entries/<entry>.json
PYTHONPATH=. python3 -m institutional_btc_vol.cli validate-source-intake artifacts/institutional/historical/source_intake_manifest.json --output artifacts/institutional/historical/source-intake-validation-current.md
```

**Current truth:** current captures are available for internal diligence, but licensed historical readiness stays blocked until these files are procured and validated.
