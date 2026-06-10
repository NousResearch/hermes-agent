# FTD V2 Control Plane Implementation Report

Date: 2026-05-14

## Changed files

- `/Users/johngalt/.hermes/scripts/ftd_lib.py`
- `/Users/johngalt/.hermes/scripts/ftd_start.py`
- `/Users/johngalt/.hermes/scripts/ftd_stop.py`
- `/Users/johngalt/.hermes/scripts/ftd_status.py`
- `/Users/johngalt/.hermes/scripts/ftd_set_state.py`
- `/Users/johngalt/.hermes/scripts/ftd_pm_runner_wrapper.py`
- `/Users/johngalt/.hermes/scripts/ftd_watchdog.py`
- `/Users/johngalt/.hermes/scripts/ftd_liveness_check.py`
- `/Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py`
- `/Users/johngalt/.hermes/hermes-agent/docs/ftd-v2-codex-implementation-report-2026-05-14.md`

## Implemented

- Added V2 state migration with `schema_version: 2`, `repo_realpath`, and legacy `active_pm_runner_pid` copied to `active_pm_wrapper_pid` without deleting legacy fields.
- Added explicit PM runner fields: `active_pm_wrapper_pid`, `active_pm_child_pid`, and `active_pm_run_id`.
- Updated generated default repo config to avoid nonexistent `galtcode`, `galtresearch`, and `galtops` profiles.
- Added guarded project-board dispatch: only `ACTIVE` projects with live PM child/wrapper processes dispatch.
- Added project-scoped watchdog script with `--project-id` / `--repo`; no global `--all` behavior.
- Added liveness script and `sprint_task_control_plane_problem(state, task)`.
- Updated start to write a 300-second `STARTING` reservation and refuse duplicate active/starting starts.
- Updated stop to terminate child process group first, then wrapper process group, with SIGKILL fallback and guarded PID/run cleanup.
- Updated wrapper to record child/wrapper/run metadata, set `ERROR` whenever the child exits while the project remains `ACTIVE`, and avoid clearing a newer run.
- Updated set-state so it no longer clears active PID fields.
- Expanded status JSON/human fields for schema, run ID, child/wrapper PID liveness, last exit, and watchdog status.
- Expanded focused control-plane tests.

## Verification

- `python3 -m pytest /Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py -q`
  - Failed under system Python: `No module named pytest`.
- `source venv/bin/activate && python3 -m pytest /Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py -q`
  - Passed: `14 passed in 0.12s`.
- `python3 -m py_compile /Users/johngalt/.hermes/scripts/ftd_*.py`
  - Passed.

## Deferred

- Future statutes/case-law profiles were not implemented.
- `ftd_validate.py`, `ftd_git.py`, `ftd_resources.py`, and `ftd_workers.py` were not added; they remain later-phase work beyond the Phase 1 control-plane floor.
- Watchdog notification delivery remains cron/stdout based; deeper notification de-dup metadata can be extended later.
- Cron verification uses available `hermes cron list` probing and recreates when stored IDs are clearly stale; deeper paused/running status semantics depend on Hermes cron CLI output.
