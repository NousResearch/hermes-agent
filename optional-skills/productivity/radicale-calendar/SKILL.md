---
name: radicale-calendar
description: Manage appointments on a self-hosted Radicale calendar.
version: 1.0.0
author: joost (KoenradusXLVIII), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: RADICALE_URL
    prompt: Base URL of your Radicale server (e.g. https://calendar.example.com/)
    help: "The CalDAV root URL Radicale serves from - same one you'd enter in any CalDAV client."
  - name: RADICALE_USERNAME
    prompt: Radicale account username
  - name: RADICALE_PASSWORD
    prompt: Radicale account password
    help: "Basic auth password (or an app password, if Radicale sits behind a reverse proxy that enforces one)."
metadata:
  hermes:
    tags: [Calendar, CalDAV, Radicale, Scheduling]
    related_skills: [google-workspace]
    homepage: https://radicale.org
---

# Radicale Calendar

Check, create, reschedule, and delete appointments on a self-hosted
[Radicale](https://radicale.org) CalDAV calendar via a small CLI wrapper
(`scripts/radicale_cli.py`) around the `caldav` Python library. Talks
directly to your own Radicale server - no OAuth flow, just a URL and
basic auth.

## When to Use

- "What's on my calendar [today / this week / on X date]?"
- "Do I have anything on [date/time]?" (check for conflicts before agreeing to something)
- "Add an appointment for..." / "Schedule a meeting..."
- "Move my [X] appointment to..." / "Reschedule..."
- "Cancel my [X] appointment"

## Prerequisites

Requires the `caldav` package. Standard installs get it via `hermes-agent[caldav]`
(`uv sync --extra caldav` in a dev checkout); `pip install caldav==3.2.1` works
too if running this script outside the main Hermes environment.

Set `RADICALE_URL` / `RADICALE_USERNAME` / `RADICALE_PASSWORD` (see frontmatter
above - Hermes prompts for these securely on first load if any are missing,
storing them in `~/.hermes/.env`, which is what this script reads from).

## How to Run

Invoke the CLI with the `terminal` tool:

```bash
python scripts/radicale_cli.py <command> [args]
```

All commands return JSON on stdout. Parse it, don't eyeball raw text.

## Quick Reference

| Command | Purpose |
|---|---|
| `list-calendars` | List available calendars |
| `list-events [--start] [--end]` | List events (defaults to next 7 days) |
| `create-event --summary --start --end [--location] [--description] [--all-day] [--recur yearly]` | Create an event |
| `reschedule-event --uid --start --end [--all-day]` | Change when an event happens |
| `update-event --uid [--summary] [--location] [--description]` | Change fields other than the time |
| `delete-event --uid` | Delete an event |

## Procedure

### List events

```bash
python scripts/radicale_cli.py list-events
python scripts/radicale_cli.py list-events --start 2026-08-12T00:00:00 --end 2026-08-19T00:00:00
```

Returns `[{uid, summary, start, end, location, description, calendar}]`.

### Create an event

```bash
python scripts/radicale_cli.py create-event \
  --summary "Dentist" --start 2026-08-15T14:00:00 --end 2026-08-15T15:00:00 \
  --location "Downtown Clinic" --description "Check-up"
```

Returns `{status: "created", uid, summary, start, end, ...}` - the `uid` is
needed for rescheduling/updating/deleting later.

### Create an all-day / recurring event (e.g. a birthday)

```bash
python scripts/radicale_cli.py create-event \
  --summary "Someone's Birthday" --start 1990-04-13 --all-day --recur yearly
```

`--all-day` takes a bare `YYYY-MM-DD` and produces a whole-day event
(`DTSTART;VALUE=DATE`), not a timed one. `--end` defaults to the day after
`--start` if omitted (iCal's DTEND is exclusive). `--recur yearly` adds
`RRULE:FREQ=YEARLY` - `list-events` expands it into future years without
re-creating the event annually.

### Reschedule an event

```bash
python scripts/radicale_cli.py reschedule-event \
  --uid <uid> --start 2026-08-16T14:00:00 --end 2026-08-16T15:00:00
```

Named `reschedule-event`, not `move-event`, deliberately - other CalDAV
tools use "move" to mean transferring an event to a *different calendar*.
This changes date/time only.

### Update other fields

```bash
python scripts/radicale_cli.py update-event --uid <uid> --location "New location"
```

Pass an empty string (`--location ""`) to clear a field entirely.

### Delete an event

```bash
python scripts/radicale_cli.py delete-event --uid <uid>
```

## Pitfalls

- **Never create, reschedule, or delete an event without confirming with
  the user first.** Show what will change (summary, date/time, location)
  and get explicit confirmation. Listing/checking needs no confirmation.
- **Always resolve an explicit ISO 8601 date/time** before calling
  anything - if the user gives a time without a date, check today's date
  rather than guessing.
- **Get `uid` from `list-events` or `create-event`'s own output** - never
  guess or invent one.
- **`--all-day` on `reschedule-event` matters** - omitting it on an
  existing all-day event's reschedule assigns a `datetime` where a `date`
  is expected, silently turning it into a timed event.
- **Only one calendar exists** on a fresh Radicale instance, and commands
  default to it. Once more than one exists, check `list-calendars` first
  rather than assuming which one is meant.

## Verification

After any create/reschedule/update, re-run `list-events` (or search by the
returned `uid`) and confirm the summary/start/end/location match what was
requested before telling the user it succeeded.
