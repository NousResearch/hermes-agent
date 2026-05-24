# Production Run WARN State Recording Template

## When to use this

When `run_daily_production.py` exits with code 2 (WARN) — core pipeline completed but one or more non-fatal steps failed or timed out.

## State Recording Template

```
State:         COMPLETED_WARN
Exit:          2 (non-fatal)
WARN source:   [step name / error message]
Tracking:      Single occurrence / 2+ consecutive / N of last M
Non-blocking:  Yes — [what completed successfully: screen, gates, rankings, snapshot, trade plan, etc.]
Patch/live:    [temp patches in effect, e.g. market_data_max_age_days=5 — revert when Yahoo 429 clears]
```

## Follow-up Items

When recording a WARN state, include these follow-up items based on the WARN source:

### Herald classify timeout
1. Verify classification retries on the next supervisor run (non-blocking)
2. Track recurrence — 3+ consecutive → ops issue
3. If downgrading: increase timeout or switch to non-grok fallback

### Market data staleness patch
1. Note the patch as TEMPORARY with a revert trigger condition
2. Examples of trigger conditions:
   - "Revert when Yahoo/yfinance 429 clears and normal market-data freshness resumes"
   - "Revert after 5 consecutive clean PASS runs"

### General
1. Continue monitoring posture (no new implementation)
2. Re-check at next cron cycle
3. If same WARN recurrs 3+ times, escalate to operator with a specific remediation proposal

## Example (2026-05-19)

```
State:         COMPLETED_WARN
Exit:          2 (non-fatal)
WARN source:   Herald classify_press_releases.py --use-grok timed out at 300s
Tracking:      Single occurrence (first observed)
Non-blocking:  Yes — screen, gates, rankings (842KB), snapshot, trade plan, portfolio report, readiness scorecard all completed
Patch/live:    market_data_max_age_days=5 (TEMPORARY — revert when Yahoo 429 clears)
```
