# Cron Guide

This directory owns scheduled jobs and the scheduler tick loop.

## Files

- `cron/jobs.py`: job storage, schedule parsing, output storage, job mutation.
- `cron/scheduler.py`: due-job execution, delivery, locking, cron agent setup.
- `tools/cronjob_tools.py`: agent-facing cron management tool.
- `hermes_cli/main.py` and `hermes_cli/cron.py`: user-facing CLI routing.

## Schedules

Supported forms:

- duration strings such as `30m`, `2h`, `1d`
- "every" phrases such as `every 2h` or `every monday 9am`
- five-field cron expressions
- ISO timestamps for one-shot jobs

Per-job fields can include skills, model/provider overrides, pre-run scripts,
`context_from`, `workdir`, repeat count, and delivery targets.

## Execution Rules

Cron jobs run as fresh agent sessions. Prompts must be self-contained. Cron
agents pass `skip_memory=True`; do not rely on user memory providers inside
scheduled jobs.

The scheduler uses a file lock under `get_hermes_home() / "cron"` to avoid
duplicate ticks across processes.

Cron delivery output is not appended to the target chat session. It is delivered
with a header/footer frame from the cron session.

## Hardening

Keep these invariants:

- runaway sessions are interrupted by the scheduler timeout path,
- catchup and grace windows avoid stale job storms,
- dangerous command handling follows cron approval config,
- assembled cron prompts are scanned for injection patterns,
- cron scripts are bounded by configured script timeout,
- failures are delivered clearly when a delivery target exists.

## Skills

Jobs can attach one or more skills. The scheduler loads those skills before the
prompt and bumps usage so curator sees active skills.

If curator rewrites or archives agent-created skills, cron job skill references
may need repair through the existing rewrite helper in `cron/jobs.py`.
