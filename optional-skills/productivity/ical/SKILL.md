---
name: ical
description: Parse and generate RFC 5545 iCalendar files.
version: 1.0.0
author: Thomas Bale (TumCucTom)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [calendar, ical, ics, rfc5545, events, meetings, scheduling]
    related_skills: [google-workspace, notion, linear]
    category: productivity
---

# iCalendar

Read and write `.ics` (iCalendar / RFC 5545) files. Generate meeting invites, parse calendar exports, convert between JSON and the iCalendar format. Stdlib only — no CalDAV, no OAuth, no network required.

Use this when the user wants a calendar artefact they can email, drop into Apple Calendar / Google Calendar / Outlook, or feed to another tool. For live integration with a hosted calendar (Google, iCloud, Fastmail), reach for a CalDAV or Google Calendar API client instead — those need credentials and are out of scope for this skill.

## When to Use

- User asks to create a meeting invite / `.ics` file / calendar event
- User has a `.ics` file and wants the events extracted as JSON or a table
- User wants to convert between calendar formats (e.g. JSON → iCal)
- User says "RFC 5545", "iCalendar", or "VEVENT"
- User wants a quick way to dump a list of meetings to share with someone

Don't use for:
- Syncing with a live hosted calendar (Google, iCloud, Office 365) — needs OAuth / CalDAV
- Building a calendar UI — this is a file-format skill, not a frontend
- Recurring events with complex RRULE exceptions — supported but not the focus; sanity-check generated output
- Time-zone math beyond "use this TZID and let the client resolve it" — the parser stores local times as written and lets the user interpret

## Prerequisites

Nothing beyond Python 3.11+ (the skill uses `datetime` and `uuid` from stdlib). To *use* the generated files, the recipient needs a calendar client that reads `.ics` — every modern mail client and calendar app does.

```bash
python --version
```

## How to Run

The skill ships a single script that doubles as a library and a CLI. The agent drives it through the `terminal` tool, or imports the parse/generate functions directly.

- **Parse an .ics into JSON**: `python scripts/ical.py read meeting.ics --json`
- **Generate an .ics from a JSON event list**: `python scripts/ical.py write events.json invite.ics`
- **Pretty-print events as a table**: `python scripts/ical.py read meeting.ics`
- **Validate an .ics file** (parse + report errors, no output): `python scripts/ical.py validate meeting.ics`
- **Generate a one-off invite from CLI flags**: `python scripts/ical.py quick "Lunch with Alex" "2026-06-15T12:30" "2026-06-15T13:30" --location "Cafe Nord" > invite.ics`

For programmatic use inside another Python script, import `parse_ics` and `generate_ics` from `scripts/ical.py`.

## Quick Reference

| Command | What it does |
|---|---|
| `ical.py read <file>` | Print events as a human-readable table |
| `ical.py read <file> --json` | Print events as JSON |
| `ical.py write <events.json> <out.ics>` | Generate an .ics from a JSON event list |
| `ical.py validate <file>` | Parse the file, report errors, exit 0 on success |
| `ical.py quick <summary> <start> <end> [flags]` | Generate a one-event .ics on stdout |

The JSON event schema is:

```json
{
  "summary": "Lunch with Alex",
  "start": "2026-06-15T12:30:00",
  "end":   "2026-06-15T13:30:00",
  "description": "Catch up on the project",
  "location": "Cafe Nord",
  "uid": "optional-stable-id@example.com",
  "organizer": "optional-name <email>"
}
```

`start` and `end` are local ISO-8601 datetimes. Add a `Z` for UTC. Trailing `Z` is preserved on output so the calendar client can render in the recipient's local time.

## Procedure

### 1. Decide direction

- **Generating** (the user wants an invite): collect event fields (summary, start, end, location, description). If any are missing, ask. Defaults: empty description, no location, no organizer.
- **Parsing** (the user has an .ics file): run `read <file> --json` and inspect the output. If the file came from a real calendar client, expect trailing `Z` on timestamps and escaped commas/semicolons in text fields.

### 2. Generate

```bash
cat > /tmp/lunch.json <<'JSON'
[
  {
    "summary": "Lunch with Alex",
    "start": "2026-06-15T12:30:00",
    "end":   "2026-06-15T13:30:00",
    "location": "Cafe Nord"
  }
]
JSON
python scripts/ical.py write /tmp/lunch.json /tmp/lunch.ics
```

Verify the output by re-reading:

```bash
python scripts/ical.py read /tmp/lunch.ics
```

### 3. Parse

```bash
python scripts/ical.py read path/to/calendar.ics --json
```

The output is a JSON array of event objects. Each has `summary`, `start`, `end`, plus any optional fields that were present (`description`, `location`, `uid`, `organizer`).

### 4. Hand off

For an invite: email the `.ics` as an attachment, or include its contents inline (`text/calendar; method=REQUEST` for proper meeting-request semantics — the script doesn't set the METHOD property, so add it manually if the recipient expects a real "invite" UI).

For parsed output: hand the JSON to the user as-is, or pipe into another tool.

## Pitfalls

- **Time zones.** RFC 5545 lets `DTSTART` carry a TZID reference, a UTC `Z` suffix, or a floating local time. The parser stores whatever it sees — no conversion. The generator writes whatever the user passed. If the user passes a naive `2026-06-15T12:30:00`, the output is a floating local time. If they pass `2026-06-15T12:30:00Z`, it's UTC. Calendar clients honour the type, so this matters: a `12:30` without a zone will render as 12:30 *in the recipient's local time*, not 12:30 UTC. Confirm with the user which they want.
- **Line folding.** RFC 5545 wraps long lines at 75 octets with a CRLF + space continuation. The parser handles unfolding. The generator folds lines longer than 75 octets. Don't pre-fold in Python; let the generator do it.
- **Escaping.** Commas, semicolons, backslashes, and newlines in text fields (`SUMMARY`, `DESCRIPTION`, `LOCATION`) need escaping. The generator handles commas, semicolons, and backslashes; the parser unescapes them. Newlines in `DESCRIPTION` should be encoded as `\n` (literal backslash-n) in the input JSON — the generator converts to RFC 5545's two-character sequence.
- **`UID` uniqueness.** Every `VEVENT` needs a stable `UID` so updates / cancellations can target it. The generator auto-fills a `uuid4@<host>` UID if the JSON omits one. If the user has a stable identifier (e.g. database row id + their domain), use that instead so future updates find the event.
- **Line endings.** RFC 5545 mandates CRLF. The generator writes CRLF; the parser accepts CRLF, LF, or mixed. Don't re-encode the output to LF — some clients tolerate it but the spec is clear.
- **Recurring events (`RRULE`).** Supported in the parser (returned as a raw `RRULE` string in the JSON). The generator does **not** expand recurrences — pass a single event for the first occurrence and add `rrule` to the JSON if the client should expand. Full EXDATE / RDATE handling is not implemented.
- **No METHOD property.** Generated files are bare `VEVENT`s inside a `VCALENDAR` without a `METHOD`. For an email invite, the sending client usually adds `METHOD:REQUEST`. If the user needs a proper meeting request, they can prepend the property themselves or use a dedicated SMTP-meeting tool.
- **Validation is shallow.** `validate` only confirms the file parses and contains at least one `VEVENT`. It does not check the spec — a `DTEND` before `DTSTART` will pass validation but render badly in clients.

## Verification

A single command proves the skill is wired up:

```bash
python scripts/ical.py quick "smoke test" "2026-06-15T12:00:00" "2026-06-15T13:00:00" | python scripts/ical.py read -
```

Expected: the second command prints a table with one row whose summary is `smoke test` and times are `2026-06-15 12:00:00` → `2026-06-15 13:00:00`. If either command fails with a traceback, the install is broken — re-check that the script is on the `PATH` (or call it with the explicit `python` prefix).
