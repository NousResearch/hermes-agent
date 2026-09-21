# Cron Setup — 2026-05-28

Session where three governance crons were created from scratch.

## Crons Created

| Job ID | Name | Schedule | Type | Delivery |
|--------|------|----------|------|----------|
| `572a4aae2e55` | denji-worker-failure-analysis | Mon 09:00 (weekly) | no_agent script | discord |
| `bbae1ceab5d2` | denji-self-eval-reminder | Fri 10:00 (weekly) | no_agent script | discord |
| `f230f9fefc1f` | denji-quarterly-audit | 1st Mon of Mar/Jun/Sep/Dec 09:00 | no_agent script | discord |

## Scripts

All three scripts were pre-existing at `~/.hermes/governance/scripts/` and had to be copied to `~/.hermes/scripts/` because the `cronjob` tool only accepts relative paths under that directory.

- `worker-failure-analysis.py` — scans all Kanban DBs for blocked/crashed/terminal-drift tasks, forced-skill gaps, integrity failures. Prints `[SILENT]` on clean runs.
- `self-eval-reminder.py` — prints a Discord-formatted reminder about the weekly self-eval protocol.
- `quarterly-audit.py` — generates system health report: profile count, kanban status distribution, cron health, WFA trend.

## Key Lesson

The WFA script was already running hourly (via a different mechanism — likely a pre-existing cron under the default profile) producing `worker-failure-analysis-*.json` files in the logboard. The new `denji-worker-failure-analysis` cron replaces ad-hoc manual runs with a scheduled weekly cadence on Monday mornings.

## Follow-up

Profile Change Ledger entry added: 28/05/26, follow-up 11/06/26. Check that all three crons fire correctly on their next scheduled run.