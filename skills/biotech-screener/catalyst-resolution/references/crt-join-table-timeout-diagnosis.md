# CRT Join Table Timeout Diagnosis

Last updated: 2026-05-19
Source session: Full fleet triage + 1C CRT join table investigation

## Overview

The CRT options join table (`output/catalyst_ev/crt_options_join.json`) is built by step 5k.21c of the daily production pipeline. It joins CRT resolution records with option-state snapshot rankings and price history. This is a RESEARCH artifact — no portfolio selection depends on it.

The CRT tracker (step 5m) runs independently and does NOT depend on the join table.

## Architecture

```
run_daily_production.py
├── Step 5k.21c (line 5537): subprocess call to scripts/research/build_crt_options_join.py
│   └── 120s timeout — subprocess can time out without blocking pipeline
│   └── Writes: output/catalyst_ev/crt_options_join.json
│   └── Loads: price_history.csv (~491K lines via _load_prices())
│   └── Reads: CRT resolution records (~40-50) + snapshot ranking directories
├── Step 5m (line 5834): tools.catalyst_resolution_tracker.run_crt()
    └── Direct function call (not subprocess)
    └── Writes: per-ticker resolution files, watchlist_current.json
```

## Diagnosis Steps

### 1. Check join table timestamp
```bash
stat output/catalyst_ev/crt_options_join.json
```
If timestamp > 72h old (excluding weekends), the join table is stale.

### 2. Search production logs for timeout
```bash
grep -i "CRT join table" logs/daily_production_2026-05-*.log
```
Timeout signature: `WARNING - CRT join table refresh failed: Command [...] timed out after 120 seconds`

### 3. Check if CRT tracker (step 5m) ran independently
```bash
grep "CRT →" logs/daily_production_2026-05-*.log
```
If tracker shows results (e.g., "CRT → 9 watchlist, 2 new resolutions") but join table is stale, the issue is isolated to step 5k.21c.

### 4. Check for pipeline abortion before CRT steps
```bash
grep -n "Aborting before screen run" logs/daily_production_2026-05-*.log
```
Market data staleness (4+ days) causes the pipeline to abort at step 3, before ever reaching CRT steps.

### 5. Check price_history.csv size
```bash
wc -l production_data/price_history.csv
```
Currently ~491K lines. The `_load_prices()` function reads the entire CSV into memory each run. Large size + slow I/O (WSL cold-start) is the primary timeout contributor.

## Root Cause Analysis

The join table stopped updating because of TWO distinct events:

**Primary: May 18 timeout** — Step 5k.21c exceeded 120s on May 18. The script took >2min to load and process 491K-line price CSV + 48 resolution records against snapshot directories.

**Secondary: May 19 pipeline abortion** — Market data was 4 days stale (collected May 15), so production never reached step 5k. Even if the timeout were fixed, the join table wouldn't have refreshed on May 19.

## Remediation Options

All blocked during architecture freezes. Requires operator approval.

1. **Increase timeout** — bump 120s → 300s in `run_daily_production.py` line ~5545
2. **Optimize `build_crt_options_join.py`** — use indexed price access instead of full CSV load; or add early-exit if no new resolutions since last run
3. **Both** — safest

## Historical Note

The May 18 timeout was the FIRST time this step had failed in production. The May 15 run completed in ~2min with the same 48 resolution records and 491K-line price CSV. The May 18 failure was likely a transient slowdown (WSL post-weekend cold I/O, snapshot directory growth).
