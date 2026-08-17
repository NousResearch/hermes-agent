# Cron Doctor Spec

## Problem

Scheduled jobs can silently degrade when a script is moved, a workdir disappears,
a provider run fails, or delivery starts failing. `hermes cron list` shows some of
this inline, but there is no compact read-only health check that can be run from a
terminal, cron job, or CI-style smoke check.

## Goal

Add `hermes cron doctor` as a read-only diagnostic command that summarizes cron
job health and exits non-zero when actionable issues are found.

## Non-goals

- Do not mutate jobs or auto-repair state.
- Do not start/stop the gateway.
- Do not inspect secrets or print credentials.

## Acceptance criteria

- `hermes cron doctor` returns `0` and prints a healthy message when active jobs
  have no detected issues.
- It returns `1` and prints grouped job-level issues when any active job has:
  - last run failure (`last_status` not `ok`),
  - last delivery failure,
  - no `next_run_at` while still active,
  - `no_agent` enabled without a script,
  - script path missing/outside `HERMES_HOME/scripts`, or
  - configured workdir path missing.
- Parser, command dispatch, and focused tests cover the new subcommand.
