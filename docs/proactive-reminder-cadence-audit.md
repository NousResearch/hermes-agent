# Proactive Reminder Cadence Audit

## Problem

Joe has recurring personal maintenance tasks (for example fridge cleaning monthly and air-conditioner cleaning every 3 months). These reminders often live in Markdown notes and are easy to miss unless an agent manually rereads them.

## Goal

Add a small, offline helper that turns structured Markdown checklist rows into a deterministic morning audit of due and upcoming reminders. The helper must be safe for cron use: read local files/stdin, produce Markdown or JSON, and never send messages or mutate external services.

## Input contract

The helper scans Markdown checklist rows that contain these fields separated by `|`:

```markdown
- [ ] Clean fridge | cadence: every 1 month | last: 2026-05-01 | note: Wipe shelves
- [x] Air-conditioner cleaning | cadence: every 3 months | last: 2026-03-10
```

Required fields:
- Task title in the checklist text.
- `cadence: every N day(s)/week(s)/month(s)/year(s)`.
- `last: YYYY-MM-DD`.

Optional fields:
- `note: ...` for human context.
- `owner: ...` for routing context.

Rows outside this shape are ignored so nearby instructions/templates do not become reminders.

## Output contract

Default Markdown output:
- `## Due now` for overdue/today reminders.
- `## Upcoming` for reminders due within `--lookahead-days`.
- `## Clear` when nothing is due/upcoming.

Each reminder shows title, due date, overdue/upcoming age, cadence, source file, and optional note/owner. `--format json` emits the normalized records for downstream cron/reporting.

## Verification plan

- Unit tests cover valid parsing, ignoring adjacent guidance/template bullets, due/upcoming classification, month-end calendar handling, Markdown output, JSON output, and stdin fallback.
- Smoke check runs the script against a temporary sample note with a fixed `--today` date.
