# FTD V2 Codex Fix Report

Date: 2026-05-14

## Changed Files

- `/Users/johngalt/.hermes/scripts/ftd_watchdog.py`
  - Enforced `max_sprints_without_benjamin_review` during `READY_NEXT_SPRINT`.
  - Incremented `sprints_since_benjamin_checkpoint` for the completed sprint before cap comparison.
  - Set `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` with a precise cap summary/reason and paused/notified without creating a sprint or spawning a PM runner.
  - Changed `spawning_started_at` to use `now_iso()`.
  - Rechecked fresh ACTIVE liveness after acquiring `project_lock` before setting dead-runner `ERROR`.
  - Rechecked stale `SPAWNING_NEXT_SPRINT` state inside the lock before recovery/error handling.
  - Moved Kanban sprint creation out of the READY_NEXT_SPRINT lock section; PM runner spawn remains in the final locked commit section to avoid writing stale ACTIVE PID state over a concurrent transition.

- `/Users/johngalt/.hermes/scripts/ftd_lib.py`
  - Added `save=False` support to `create_next_sprint()` so watchdog can perform the external Kanban create outside the project lock and commit the resulting sprint evidence after rechecking state.

- `/Users/johngalt/.hermes/scripts/ftd_status.py`
  - Fixed JSON status watchdog probing to honor `--watchdog` instead of always forcing cron probing.

- `/Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py`
  - Added focused tests for checkpoint cap pause/no-spawn, current `spawning_started_at`, ACTIVE dead-runner recheck, `ftd_status.py --watchdog`, stale spawning ERROR, and missing cron recreation.
  - Extended READY_NEXT_SPRINT spawn coverage to assert checkpoint counter increment.

## Verification

Passed:

```bash
source venv/bin/activate && python3 -m pytest /Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py -q
# 20 passed in 0.22s
```

Passed:

```bash
python3 -m py_compile /Users/johngalt/.hermes/scripts/ftd_*.py
```

## Deferred Items

- PM runner spawn still happens while holding `project_lock`. I left this in place because `spawn_pm_runner()` currently performs the Popen and ACTIVE PID state write as one operation; splitting it cleanly needs a separate prepare/commit API to avoid clobbering concurrent operator transitions. The slower Kanban create call is now outside the lock.
