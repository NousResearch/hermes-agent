# A6 Restore Drill Incident Playbook

Date: 2026-03-21
Status: active (Wave 8 A6 Ops Reliability Lane)

## 0) Trigger
This drill should be executed weekly, or when `SLO-4_RESTORE_DRILL_FRESHNESS` falls out of budget, or before a major Wave upgrade rollout.

## 1) Procedure
1. Verify the primary `chk_latest` path exists.
2. Snapshot the current `memory/` directory via `cp -r memory/ memory_backup/`.
3. Locate the last known good rollback point:
   `bash ops/openclaw/continuity/history.sh --source-preset control-plane --limit 5`
4. Attempt a dry-run checkpoint rollback (does not apply by default):
   `bash ops/openclaw/continuity/verify_then_resume.sh --checkpoint <chk_id> --run-rollback`
5. If the rollback script outputs success without throwing fatal errors, the drill is considered passed.
6. Record the test run output for compliance.
7. Write/update the canonical latest evidence artifact at:
   - `state/continuity/latest/restore_drill_latest.json`
   - minimum fields: `schema`, `drilled_at`, `status`, `drill_ref`, `notes`

Example:

```json
{
  "schema": "clawd.restore_drill.evidence.v1",
  "drilled_at": "2026-04-01T15:25:00Z",
  "status": "pass",
  "drill_ref": "reports/restore_drill_2026-04-01.md",
  "notes": "Checkpoint rollback dry-run completed without fatal errors."
}
```

## 2) Automation (EX-05.1)
Restore-drill freshness is now automated through the continuity watchdog path:

- refresher command:
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh restore-drill-refresh --json`
- watchdog integration:
  - `ops/openclaw/run_no_nudge_continuity_watchdog.sh` invokes the refresher and reruns `continuity_now` when a refresh occurs.
- cadence semantics:
  - refresh threshold defaults to `OPENCLAW_NO_NUDGE_RESTORE_DRILL_REFRESH_AFTER_SEC=518400` (6 days)
  - refresher is fail-closed on status truthfulness: if evidence status is missing or not `pass`, it reruns the bounded drill even when age is still fresh.
  - SLO freshness budget remains `OPENCLAW_NO_NUDGE_RESTORE_DRILL_MAX_AGE_SEC=604800` (7 days)

Automation writes both:
- latest evidence: `state/continuity/latest/restore_drill_latest.json`
- run report: `reports/restore_drill_auto_*.md`

The drill remains bounded (dry-run rollback via `verify_then_resume.sh --run-rollback`) and fail-closed: pass/fail is recorded in evidence, while freshness stays continuously updated without manual babysitting.
