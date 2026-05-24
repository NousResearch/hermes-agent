# 13F Institutional Summary Pipeline Architecture

## Artifact Flow

```
warm_13f_cache.py
  └─> data/caches/sec_13f/PIT/{YYYY-MM-DD}/  ── PIT cache (per-manager CSVs)

run_screen.py (line ~5096)
  ├─ build_institutional_summary(as_of_date, ...)
  │   └─ reads from PIT cache ── aggregates manager 13F filings
  └─ writes data/snapshots/{date}/institutional_summary.json
```

## Canonical Source Confusion

There are **two** copies of `institutional_summary.json` with different semantics:

| Location | Purpose | Auto-refreshed? |
|---|---|---|
| `data/snapshots/{date}/institutional_summary.json` | Per-run snapshot artifact. Written daily by `run_screen.py` before cohort validation. Each dated snapshot has its own `cache_as_of_date`. | YES — daily by pipeline |
| `production_data/institutional_summary.json` | **Canonical** copy read by `check_13f_cohort_quarantine.py` for the G2 (institutional data freshness) gate and cohort Jaccard comparison. | **NO — no promotion step exists** |

## The Gap

The daily production pipeline writes fresh institutional summary data to the snapshot directory, but **no mechanism copies it to `production_data/`**. The `production_data/institutional_summary.json` was last refreshed on 2026-04-13 via a manual commit (`d241a14f`). This causes a **false G2 failure** in cohort quarantine: the quarantine validator reads the stale canonical copy, detects 46/48 filings filed vs cache_as_of_date=2026-04-13, and blocks the Jaccard comparison.

## Symptoms

- `production_data/institutional_summary.json` has `cache_as_of_date` = `2026-04-13`
- Most recent snapshot `institutional_summary.json` has `cache_as_of_date` = `2026-05-19` (or current date)
- `check_13f_cohort_quarantine.py --pre-date YYYY-MM-DD` fails G2 with "quarantine verdict stands (G2 producer freshness fail)"
- All PIT cache directories exist through current date — the producer is healthy

## Diagnosis Steps

1. Check canonical copy:
   ```bash
   python3 -c "import json; d=json.load(open('production_data/institutional_summary.json')); print(d.get('cache_as_of_date','MISSING'))"
   ```

2. Check most recent snapshot:
   ```bash
   ls -lt data/snapshots/ | head -3
   python3 -c "import json; d=json.load(open(f'data/snapshots/$(ls -t data/snapshots/ | head -1)/institutional_summary.json')); print(d.get('cache_as_of_date','MISSING'))"
   ```

3. Check PIT cache age:
   ```bash
   ls -d data/caches/sec_13f/PIT/*/ | tail -3
   ```

4. Compare dates: if snapshot and PIT are current but production_data is stale, the gap is at the promotion step.

## Minimum Viable Fix

```bash
# Copy today's snapshot to canonical production_data
cp data/snapshots/YYYY-MM-DD/institutional_summary.json production_data/institutional_summary.json
```

Then re-run cohort quarantine validation.

### After-Propagation Nuance

After propagation, the cohort quarantine G2 check will NOT necessarily show `PASS`. It will show one of two states:

| State | Meaning | Next Step |
|---|---|---|
| `REFRESH_NOT_LANDED — prior_date N, latest_snapshot M` | No actual 13F EDGAR refresh has run since propagation. The G2 check evaluates `prior_date` from the **delta JSON** (`data/13f_filing_deltas/latest.json`), which advances only when `prep_13f_refresh.py` + `warm_13f_cache.py` complete a full refresh cycle. The stale-G2 gate is resolved; the tool is correctly waiting for new data. | Wait for next scheduled 13F refresh cycle. No manual action needed. |
| `PASS — G2 producer freshness check passed` | Either (a) a real 13F refresh landed since propagation, advancing `prior_date`, or (b) the refresh cycle ran on the same day, making the propagated summary eligible. | Cohort quarantine lifts; KG launch can proceed. |

The fix is: **propagation alone does not advance `prior_date`**. It only fixes the file-freshness gap that was causing a false G2 failure. The remaining `REFRESH_NOT_LANDED` is correct behavior — the quarantine validator is now working as designed.

### Check Actual G2 Status After Propagation

```bash
python3 tools/check_13f_cohort_quarantine.py --pre-date YYYY-MM-DD --post-date YYYY-MM-DD
# Look for:
# - "G1: PASS" (snapshot exists for post_date)
# - "G2: REFRESH_NOT_LANDED" (correct — waiting for real refresh)
#   OR
# - "G2: PASS" (refresh landed — proceed)
```

## Long-Term Options

- Add a promotion step to `run_daily_production.py` (e.g., after the screen step, copy `institutional_summary.json` to `production_data/`)
- Make `check_13f_cohort_quarantine.py` read from the most recent snapshot dir instead of `production_data/`
- Move the canonical copy to a symlink pattern: `production_data/institutional_summary.json -> ../data/snapshots/latest/institutional_summary.json`

## Related Files

- `tools/warm_13f_cache.py` — PIT cache warmer
- `institutional_summary.py` — `build_institutional_summary()` implementation
- `run_screen.py` — lines ~5096 (call to build) and ~7270 (write to snapshot)
- `tools/check_13f_cohort_quarantine.py` — quarantine validator (reads production_data copy)
- `tools/prep_13f_refresh.py` — pre-refresh baseline (reads from snapshot, not production_data)
