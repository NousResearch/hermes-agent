---
name: apple-reminders
description: "Apple Reminders via remindctl: add, list, complete."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Reminders, tasks, todo, macOS, Apple]
prerequisites:
  commands: [remindctl]
---

# Apple Reminders

Use `remindctl` to manage Apple Reminders directly from the terminal. Tasks sync across all Apple devices via iCloud.

## Prerequisites

- **macOS** with Reminders.app
- Install: `brew install steipete/tap/remindctl`
- Check access: `remindctl status` (want "Full access")
- Request access: `remindctl authorize` (a human must click Allow in the macOS prompt)
- Diagnose: `remindctl doctor`

Access is granted per calling app. A Hermes process launched differently from the terminal where it was granted (e.g. the gateway, behind launchd/osascript) may need its own grant. If `remindctl status` is not "Full access", tell the user instead of retrying.

## When to Use

- User mentions "reminder" or "Reminders app"
- Creating personal to-dos with due dates that sync to iOS
- Managing Apple Reminders lists
- User wants tasks to appear on their iPhone/iPad

## When NOT to Use

- Scheduling agent alerts → use the cronjob tool instead
- Calendar events → use Apple Calendar or Google Calendar
- Project task management → use GitHub Issues, Notion, etc.
- If user says "remind me" but means an agent alert → clarify first

## Running Unattended

With no human at the keyboard, a command that prompts will hang.

- Pass `--no-input` on every write (`add`, `edit`, `complete`, `delete`, `list --create/--delete/--rename`). Every command accepts it; it disables interactive prompts.
- `delete` and `list <name> --delete` delete immediately under `--no-input` or with no TTY; the CLI will not confirm (remindctl only prompts on a TTY). Always confirm with the user first. For a reminder, preview with `remindctl delete <id> --dry-run`, then run with `--force --no-input`. `list` has no `--dry-run`: preview a list with `remindctl list <name> --json`, confirm, then `remindctl list <name> --delete --force --no-input`.
- `complete` and `delete` accept `-n`/`--dry-run` to preview.
- Pass reminder IDs, not indexes. Commands take an "Index or ID prefix", and indexes shift as reminders change. Read the `id` field (a UUID) from `--json` output and pass it, or a unique prefix of it.
- Use `--json` and parse it; don't scrape the default output.

## Quick Reference

### View Reminders

```bash
remindctl                    # Today's reminders
remindctl today              # Today
remindctl tomorrow           # Tomorrow
remindctl week               # This week
remindctl overdue            # Past due
remindctl all                # Everything
remindctl 2026-01-04         # Specific date
remindctl show open --list Work --json
```

The bare forms are shortcuts for `remindctl show [today|tomorrow|week|overdue|upcoming|open|completed|all|<date>] [--list L]`.

### Search / Inspect

```bash
remindctl search "milk" --json                       # add --list L, --completed
remindctl info <id> --json                           # one reminder
```

### Manage Lists

```bash
remindctl list               # List all lists
remindctl list Work          # Show specific list
remindctl list Projects --create --no-input          # Create list
remindctl list Work --delete --force --no-input      # Delete list (confirm with user first)
```

### Create Reminders

```bash
remindctl add "Buy milk" --no-input
remindctl add --title "Call mom" --list Personal --due tomorrow --no-input
remindctl add --title "Meeting prep" --due "2026-02-15 09:00" --no-input
```

Extras for `add` and `edit`: `-r`/`--repeat`, `-p`/`--priority`, `--url`, `--location` with `--radius` and `--leaving`.

### Due Time vs Alarm / Early Nudge

`--due` and `--alarm` are different fields:

- `--due` sets the reminder's due date/time.
- `--alarm` sets the EventKit alarm/notification trigger. Timed due reminders may default to an alarm at the due time, but pass `--alarm` explicitly when the user asks for an earlier nudge.

For a reminder due at 2:00 PM with a notification 30 minutes earlier:

```bash
remindctl add --title "Hairdresser" --due "2026-05-15 14:00" --alarm "2026-05-15 13:30" --no-input
```

### Edit

```bash
remindctl edit <id> --due "2026-05-15 14:00" --alarm "2026-05-15 13:30" --no-input
```

Extras: `--clear-due`, `--clear-alarm`, `--complete`/`--incomplete`, `-l`/`--list` to move to another list.

The Reminders UI may show or group the item by the alarm time because that is when the notification fires. Verify with JSON instead of assuming the due time moved:

```bash
remindctl info <id> --json
```

Expected shape:

- `dueDate`: actual due time
- `alarmDate`: notification / early nudge time

Apple's public `EKReminder` docs list only reminder-specific properties. Alarm support comes from inherited `EKCalendarItem` behavior exposed by remindctl's `--alarm` flag.

### Complete / Delete

```bash
remindctl complete <id> [<id> ...] --no-input     # Complete by ID
remindctl delete <id> --force --no-input          # Delete by ID (confirm with user first)
remindctl delete <id> --dry-run                   # Preview
```

### Output Formats

```bash
remindctl today --json       # JSON for scripting
remindctl today --plain      # TSV format
remindctl today --quiet      # Counts only
```

## JSON Shape

- `add`, `info`, `edit` return one object; `show`, `list`, `complete` return arrays; `delete` returns `{"deleted": N}`.
- Reminder fields: `id`, `title`, `notes`, `listID`, `listName`, `dueDate`, `alarmDate`, `dueDateIsAllDay`, `isCompleted`, `completionDate`, `priority`, `creationDate`, `lastModifiedDate`.
- `--due`/`--alarm` times without a zone are local time; JSON dates are UTC (14:00 EST comes back as `19:00:00Z`).

## Date Formats

Accepted by `--due` and date filters:
- `today`, `tomorrow`, `yesterday`
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:mm`
- ISO 8601 (`2026-01-04T12:34:56Z`)

## Rules

1. When user says "remind me", clarify: Apple Reminders (syncs to phone) vs agent cronjob alert
2. Always confirm reminder content and due date before creating
3. Use `--json` for programmatic parsing, and IDs from it for `edit`/`complete`/`delete`
4. Pass `--no-input` on every write. Deletes run immediately with no CLI confirmation, so confirm with the user first (preview a reminder with `delete --dry-run`, a list with `list <name> --json`), then use `--force --no-input`
