# P0-T01 Assumptions

- Capture time: 2026-07-29T14:58:45+08:00
- Branch: feat/cron-control-p0
- Worktree: /Users/ryanchao/.hermes/worktrees/cron-control-p0
- Mode: read-only snapshot

## Scope

- The inventory covers the current cron control-plane baseline only.
- The snapshot is limited to five source files:
  - `cron/jobs.py`
  - `cron/executions.py`
  - `cron/delivery_watchdog.py`
  - `cron/provider_recovery.py`
  - `cron/scheduler.py`

## Exclusions

- No live runtime state under `~/.hermes/cron/` was read or modified.
- No secrets, full prompts, or full model outputs were recorded.
- No production behavior was changed.

## Follow-up Rule

- Any new cron-control source file added before P0-T02 should be appended to
  both the inventory and the hash manifest.
