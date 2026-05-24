# Snapshot Artifact Layout

## Output Location

Every pipeline run writes to `data/snapshots/{YYYY-MM-DD}/`. This is the canonical output. The `production_data/` directory is a separate canonical store that receives promoted copies of select artifacts — it is NOT refreshed every pipeline run.

## Snapshot Directory Contents

| Artifact | File | When Created | Notes |
|----------|------|-------------|-------|
| Rankings (primary) | `rankings.csv` | Step 3 (Screen) | The authoritative ranking output. SHA256 sum at `rankings.csv.sha256`. |
| Decision portfolio | `decision_portfolio.csv` | Step 8 (Action Packet) | Position-level decisions |
| Audit report | `audit/` directory | Step 4 (Audit) | Per-instrument audit data |
| Gates results | `gates/` directory | Step 5 (Gates) | Gate pass/fail per check |
| Drift report | `drift_report.json` | Step 7 (Drift Report) | Cross-snapshot feature deltas |
| Action packet | `action_packet.json` | Step 8 | Trade-ready decisions |
| Shadow portfolio | `shadow_portfolio.json` | Step 9 | Shadow portfolio at snapshot |
| Trade plan | `trade_plan.json` | Step 10 | Execution plan |
| Portfolio report | `portfolio_report.json` | Step 11 | P&L and risk metrics |
| Readiness scorecard | `readiness_scorecard.json` | Step 12 | Pipeline health |
| Institutional summary | `institutional_summary.json` | Step 3 | 13F aggregate (two-copy gap — see `references/13f-institutional-summary-pipeline.md`) |

## How to verify a pipeline run

```bash
# Check if today's snapshot exists and how big
du -sh data/snapshots/$(date +%Y-%m-%d)/

# Check the primary ranking artifact
ls -lh data/snapshots/$(date +%Y-%m-%d)/rankings.csv
stat data/snapshots/$(date +%Y-%m-%d)/rankings.csv

# Count ranked instruments
head -1 data/snapshots/$(date +%Y-%m-%d)/rankings.csv
wc -l data/snapshots/$(date +%Y-%m-%d)/rankings.csv

# Check run log for errors
cat production_data/run_log_$(date +%Y-%m-%d).json | python3 -c "import sys,json; d=json.load(sys.stdin); print('Errors:', len(d.get('errors',[])), 'Warnings:', len(d.get('warnings',[])))"

# List most recent snapshots (default: last 5)
ls -lt data/snapshots/ | head -5
```

## Naming Convention History

| Version | Ranking Artifact Name | Used Until |
|---------|----------------------|------------|
| v1 (legacy) | `snapshot_top30.json` | ~2026-02-07 |
| v2 (current) | `rankings.csv` | Present |

If an agent or operator searches for `snapshot_top30.json` and finds nothing, look for `rankings.csv` instead. The legacy name no longer exists in the pipeline output.

## What production_data/ Contains (and doesn't)

`production_data/` is NOT refreshed every pipeline run. It receives only specific promoted artifacts:

- `universe.json` — refreshed by universe_maintenance cron (10:00 ET)
- `institutional_summary.json` — **NOT auto-refreshed** (manual promotion needed)
- `run_log_YYYY-MM-DD.json` — written by the crontab wrapper after each pipeline run
- `market_data.json` — refreshed by collect_market_data.py; stale detection at Step 5

Do NOT check `production_data/` for snapshot artifacts — check `data/snapshots/{date}/` instead.
