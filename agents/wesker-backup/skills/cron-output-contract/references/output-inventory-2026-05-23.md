# Cron Output Inventory — 23/05/26 (Session Outcome)

Ground-truth snapshot after Sahil's output consistency audit. Generated 23/05/26. All changes applied in-session.

## Changes Applied This Session

| Action | Item | Reason |
|--------|------|--------|
| REMOVED | `kanban-team-flow-smoke` | 0-signal output, replaced by quality gate + kanban digest |
| REMOVED | `safety-belt-cleanup-evaluation` | One-shot, already ran |
| ABSORBED into heartbeat | `memory_watchdog.sh` → `memory_health_finding()` | Same domain, similar cadence |
| ABSORBED into heartbeat | `cron-gap-monitor.sh` → `cron_gap_finding()` | Same domain, similar cadence |
| ABSORBED into heartbeat | `services_health_watchdog.py` → `duplicate_gateway_finding()` | Same domain, similar cadence |
| REVIVED | `system-report-daily` at 07:00 to `#governance` | No morning overview existed |
| REVIVED | `calendar-brief-daily` at 07:05 to `#calendar` | Dead since 19/05 |
| REVIVED | `kanban-daily-digest` at 00:00 to `#governance` | End-of-day gap |
| REVIVED | `content-engine-daily` at 09:00 to `#content` | Dead since 10/05 |
| REVIVED | `token-health-weekly` Mon 08:30 to `#governance` | Never wired |
| REVIVED | `hermes-drift-weekly` Mon 08:45 to `#governance` | Never wired |
| REVIVED | `prompt-optimizer-weekly` Sun 20:00 to `#governance` | Never wired |
| FIXED | `token_health_wrapper.sh` — Telegram HTML->Discord-safe | `<b>` tags rendered as raw text |

## Current Active Cron Outputs (17 jobs, all Discord-only)

Telegram adapter disconnected 19/05/26 18:48. `TELEGRAM_BOT_TOKEN` env var renamed to `_DISABLED`. Zero Telegram output.

| # | Cron Name | Schedule | Delivers To | Agent? |
|---|-----------|----------|-------------|--------|
| 1 | MrHermagi Daily Lesson | 07:00 daily | `#mr-hermagi` thread | LLM |
| 2 | system-report-daily | 07:00 daily | `#governance` | no_agent |
| 3 | calendar-brief-daily | 07:05 daily | `#calendar` (ID:1506022733287391282) | no_agent |
| 4 | github-radar-merged | 08:00 daily | `#research-digest` | LLM+script |
| 5 | mailbox-cleaner-main | 08:00 daily | `#calendar` (ID:1506022733287391282) | LLM |
| 6 | mailbox-cleaner-jobhunt | 08:05 daily | `#job-hunt` | LLM |
| 7 | token-health-weekly | Mon 08:30 | `#governance` | no_agent |
| 8 | hermes-drift-weekly | Mon 08:45 | `#governance` | LLM |
| 9 | content-engine-daily | 09:00 daily | `#content` | no_agent |
| 10 | mailbox-cleaner-urgent-detector | 09-20 hourly | `#job-hunt` | LLM |
| 11 | signals-daily-scan | 10:00 daily | `#signals` (ID:1507448574844207186) | LLM |
| 12 | kensei-triage-processor | Every 30m | `#governance` | LLM |
| 13 | kensei-heartbeat-audit | Every 2h | `#governance` | LLM+script |
| 14 | kensei-quality-gate | 09/13/17/21 | `#governance` | no_agent |
| 15 | knowledge-weekly-digest | Mon 09:00 | `#knowledge` (ID:1507448575796187278) | LLM |
| 16 | prompt-optimizer-weekly | Sun 20:00 | `#governance` | LLM |
| 17 | kanban-daily-digest | 00:00 daily | `#governance` | no_agent |

### Discord channel map
- `#governance` — 7 crons (triage, heartbeat, quality-gate, system-report, kanban-digest, drift, prompt-optimizer, token-health) at mixed cadences
- `#job-hunt` — mailbox jobhunt + urgent detector
- `#research-digest` — GitRadar only
- `#calendar` (ID:1506022733287391282) — mailbox main + calendar brief
- `#signals` (ID:1507448574844207186) — signals scan
- `#knowledge` (ID:1507448575796187278) — weekly digest
- `#content` — content engine daily
- `#mr-hermagi` thread (1507357967731916942:1507388984135909378) — daily lesson

### Profile cron files
All three active profiles (octacon, gojo, wesker) have empty cron/jobs.json.

## Gaps Resolved

| Gap | Fixed By |
|---|---|
| No morning pulse | System report 07:00 + calendar brief 07:05 + consistent morning cadence |
| No end-of-day kanban | `kanban-daily-digest` at 00:00 to `#governance` |
| No blocker pushes | 3 new probes in heartbeat-audit (memory, cron-gap, duplicate gateway) — fires kanban tasks automatically |
| Dead content pipeline | `content-engine-daily` revived at 09:00 to `#content` |
| Telegram HTML in Discord | `token_health_wrapper.sh` fixed — `<b>` → plain text |
