# Nightly Meeting Prep Brief Helper

## Goal

Add a small local-first helper script that turns a human-maintained JSON/YAML meeting queue into a compact Markdown prep brief for upcoming meetings. This is aimed at Joe's personal operating system: reduce morning context-switching by surfacing who he is meeting, why it matters, what to prepare, and stale/missing prep fields.

## Non-goals

- No calendar API access.
- No emails/messages sent.
- No live deployment or background service.
- No mutation of source files.

## Input contract

`scripts/meeting_prep_brief.py` accepts one or more local files containing either:

- a top-level list of meeting records, or
- a mapping with a `meetings` list.

Supported file types: `.json`, `.yaml`, `.yml`.

Meeting fields are intentionally flexible for hand-written notes:

- Required-ish: `title` or `name`; `date` or `datetime`.
- Optional: `time`, `timezone`, `attendees`, `goals`, `prep`, `questions`, `materials`, `notes`, `link`, `source`, `status`.
- `attendees`, `goals`, `prep`, `questions`, and `materials` may be strings or lists.

## Output contract

Default output is Markdown:

- `# Meeting Prep Brief`
- one section per upcoming meeting, sorted by date/time
- explicit missing-prep warnings for records without goals/prep/questions/materials
- a `## Quick Actions` section with deterministic next actions

If no active meetings are due in the window and there are no data-hygiene quick actions, output exactly `[SILENT]`.

`--json` emits structured JSON with `generated_for`, `window_days`, `meetings`, and `quick_actions`.

## Selection logic

- Include meetings whose date/datetime is between `--today` and `--today + --days` inclusive.
- Default `--days`: 7.
- Skip statuses: `done`, `cancelled`, `canceled`, `archived`.
- If a record has no parseable date, exclude it from the main brief but include a quick-action warning.

## Verification plan

- Unit tests for JSON input, YAML input when PyYAML is available, status/date filtering, flexible list parsing, missing-prep warnings, JSON output, and exact `[SILENT]` behavior.
- Smoke check with a temporary sample file.
