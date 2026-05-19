# Source Intake Blocker Audit — Current Captures

- Run audited: `btcvol-20260517-162152`
- Audited at UTC: `2026-05-18T17:10:12.616640+00:00`
- Control: `SCREEN-ONLY · SOURCE AVAILABILITY AUDIT · NOT EXECUTABLE`

## Finding

The production/source-intake blocker list is stale relative to current monitor artifacts. It is reading `artifacts/institutional/historical/source_intake_manifest.json`, which intentionally contains only two fixture historical sources. The repo actually has current live/source captures for most groups under `artifacts/institutional/data/raw/` and `normalized/`, but those captures are not yet converted into the strict historical licensed-source intake manifest.

## Current local availability

- **IBIT options history**: found — `artifacts/institutional/data/raw/btcvol-20260517-162152/nasdaq_ibit_option_chain.json`
  - sha256: `fdff135345c245b05548dc0ad44ca56183a3636d3615e647e40ccbfaa241b83c`; bytes: `186563`
- **Deribit options history**: found — `artifacts/institutional/data/raw/btcvol-20260517-162152/deribit_book_summary_btc_options.json`
  - sha256: `fbb4621fb847f8e84c898d6ccd77e3a044fae9141e2b6ad118ff1b080833f738`; bytes: `533720`
- **CME Bitcoin options history raw BBO**: found — `artifacts/institutional/data/raw/btcvol-20260517-162152/databento_cme_btc_options_bbo_1m.csv`
  - sha256: `4a57b6cf64294627276c09abcb8cf9785039235de6c8d9bf7900f96ff69d618e`; bytes: `646595`
- **CME Bitcoin options history normalized**: found — `artifacts/institutional/data/normalized/btcvol-20260517-162152/databento_cme_btc_options_bbo.csv`
  - sha256: `c09e7e8df608abb8bbf76fc6fe6197ae8a0fdc68039cbfca782c0fc2e8515f6c`; bytes: `274518`
- **BTC reference history**: found — `artifacts/institutional/historical/source_audit/btc_reference_from_run_manifest.jsonl`
  - sha256: `b46634d69256a4a6a313d564241603cb42c85e9ec237d95e825e766666e5273e`; bytes: `321`
- **IBIT holdings history official attempt**: found — `artifacts/institutional/data/raw/btcvol-20260517-162152/ishares_ibit_holdings.csv`
  - sha256: `1b838d2c121f21f5a2ed8c43dab0841bc45a486a2bb8f756cdc27c637a2f5169`; bytes: `8539634`
- **IBIT holdings history cached valid CSV**: found — `artifacts/institutional/data/raw/btcvol-20260517-162152/ishares_ibit_holdings_fallback_cached.csv`
  - sha256: `f7ba3f217ef58e76cd72e248c61d25c9646325252663b9f9eaeee392188ffeef`; bytes: `4749`
- **Rates and fee curves**: found — `artifacts/institutional/historical/source_audit/fred_rates_sofr_dgs1_dgs3mo_current.csv`
  - sha256: `f863aa00472ca8378e06a3f8dc080606e2240c529ad2c968ba3b648e90b410b2`; bytes: `353760`

## Fixability classification

- **Can fix now locally:** BTC reference file, rates/fee public reference file, manifest generation from existing captures, better report language that separates “data exists” from “readiness gate passed”.
- **Can partially fix from existing artifacts:** CME has Databento captured files in the latest run, but current environment has no `DATABENTO_API_KEY`, so it can be documented/replayed from existing raw files but not refreshed right now.
- **Cannot honestly clear as investment/client readiness yet:** IBIT and Deribit are public/screen sources, and the source-intake gate says licensed/replay-ready historical coverage. Public screen captures can support internal diligence but should not be promoted to licensed historical readiness.
- **Still externally gated:** license scope approval for redistribution/use, enough history for sample gates, and two-counterparty quote evidence.

## Rates proof

Fetched FRED public CSV `SOFR,DGS1,DGS3MO` into the audit folder. Latest non-empty tail:

- `2026-05-11,3.60,3.79,3.70`
- `2026-05-12,3.60,3.80,3.70`
- `2026-05-13,3.59,3.79,3.69`
- `2026-05-14,3.56,3.79,3.69`
- `2026-05-15,3.55,,`
