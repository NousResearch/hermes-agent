# Cron inventory receipt (M5)

> Read-only. Jobs stored in `~/.hermes/profiles/auto-coder/cron/jobs.json`.  
> **No create/update/pause/remove performed.**

## Host profile

All jobs below run under scheduler context for **`auto-coder`** (`HERMES_HOME=/home/khall/.hermes/profiles/auto-coder`).

## Jobs

| job_id | name | schedule | mode | deliver | last_status | mutation_risk | notes |
|--------|------|----------|------|---------|-------------|---------------|-------|
| `35ea2ada2a15` | nightly-hermes-dream-retrospective | `30 2 * * *` | agent + script | origin | ok | **medium** | May patch skills/memory; prompt forbids gateway restart, cron recursion, config mutation |
| `ff1a46693ee7` | daily-hermes-ai-edge-briefing | `0 8 * * *` | agent + script | origin | ok | **medium-high** | Web/browser/terminal; read-only Hermes probes in prompt; skill patches possible |
| `2c29486eb2a9` | Hermes portable backup sync | `0 3 * * *` | **no_agent** script | origin | ok | **medium** | `hermes-portable-backup-cron.sh` — filesystem copy/sync side effects |

## Deliver target redaction

All jobs: `deliver: origin` (session home channel — no raw chat id in this receipt).

## Toolsets (enabled per job)

- **Nightly:** terminal, file, session_search, skills, memory  
- **Daily:** web, browser, terminal, file, skills, code_execution, memory, session_search  
- **Backup:** n/a (script only)

## Risk classes

- **Low:** none currently scheduled as pure read-only no-agent watchers.
- **Medium:** nightly retrospective, backup script (bounded by script + prompt).
- **High:** none if prompts are honored; daily briefing has broadest tool surface — monitor for prompt drift.

## DWF / Kanban

Operator policy: **no Kanban dispatch** from cron unless prompts explicitly add it (current nightly prompt forbids Kanban).

## Verification

```bash
HERMES_HOME=~/.hermes/profiles/auto-coder hermes cron list
```