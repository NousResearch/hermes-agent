# Yahoo Finance 429 Rate Limit — Production Pipeline Blocked (2026-05-19)

## Incident Summary

2026-05-19: Yahoo Finance API returned HTTP 429 for all tickers, blocking `collect_market_data.py` and the daily production pipeline. Market data cached from 2026-05-15 (4 days stale) failed the 3-day staleness gate in `run_daily_production.py`, aborting the screen.

## Evidence

- `logs/daily_production_2026-05-19.log` line 2816: `Market data staleness gate: FAIL — market_data.json is 4d stale (collected=2026-05-15, as_of=2026-05-19, max=3d).`
- 1144 CTgov trial changes detected before the abort — data processing won't be wasted when the pipeline re-runs.
- `collect_market_data.py` (both normal run and `--force-refresh`) showed all tickers returning `429 Client Error`

## Diagnosis Steps

1. **Check cron log**: `logs/cron.log` — empty means wrapper-level PASS but pipeline-level abort (wrapper only captures top-level pass/fail, not step-level gates).

2. **Check daily production log**: `logs/daily_production_YYYY-MM-DD.log` — tail reveals the blocking gate. 2824 lines for this failure.

3. **Check market_data.json age**: `stat production_data/market_data.json` — modification date shows last successful collection.

4. **Verify Yahoo connectivity**:
   ```bash
   curl -s -o /dev/null -w "%{http_code}" \
     "https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=financialData&corsDomain=finance.yahoo.com"
   ```
   Returns 429 when rate limited, 200 when healthy.

5. **Check yfinance version**: v0.2.35 (not outdated — 429 is server-side, not library issue).

6. **Check prior successful run**: `logs/daily_production_2026-05-18.log` (240739 lines) shows run succeeded with market_data aged 3d — barely passed gate. Evidence that 3d staleness was the limit.

## What Did NOT Work

- `collect_market_data.py` without flags: fell back to cached data
- `collect_market_data.py --force-refresh`: same failure — network test at script start catches the 429 before force-override takes effect
- Price refresh via yfinance in `run_daily_production.py` Step 1: silent failure (0 rows appended, pipeline continues but trades on stale prices)
- No proxy or alternate data source was configured

## Pipeline Resilience Assessment

| Component | Status During 429 | Impact |
|-----------|------------------|--------|
| Price refresh | Silent failure (0 rows, non-fatal) | Stale prices in screen |
| Market data collect | Fallback to cached (non-fatal to collect) | Fatal at upstream gate |
| Staleness gate (3d) | HARD fail | Pipeline blocked |
| Screen step | Never reached | Aborted |
| CTgov trial changes | Captured before abort | Not wasted |

## Key Observation

The daily production pipeline's 3-day market data staleness gate is a HARD stop. There is no soft-warning grace period, no operator-override flag, no config-based extension mechanism — the gate is hardcoded as `MAX_MARKET_DATA_AGE_DAYS = 3` in `run_daily_production.py`. If Yahoo goes down for a multi-day stretch, the pipeline remains blocked until the data is refreshed or the gate is manually extended.
