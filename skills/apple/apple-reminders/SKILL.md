---
name: apple-reminders
description: "Use when managing Apple Reminders with remindctl."
version: 1.0.1
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
- Grant Reminders permission when prompted
- Check: `remindctl status` / Request: `remindctl authorize`

## When to Use

- User mentions "reminder" or "Reminders app"
- Creating personal to-dos with due dates that sync to iOS
- Managing Apple Reminders lists
- User wants tasks to appear on their iPhone/iPad

## When NOT to Use

- Scheduling agent alerts → use the cronjob tool instead
- Calendar events → use the user's canonical calendar service.
- Project task management → use GitHub Issues, Notion, etc.
- If user says "remind me" but means an agent alert → clarify first

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
```

### Manage Lists

```bash
remindctl list               # List all lists
remindctl list Work          # Show specific list
remindctl list Projects --create    # Create list
remindctl list Work --delete        # Delete list
```

### Create Reminders

```bash
remindctl add "Buy milk"
remindctl add --title "Call mom" --list Personal --due tomorrow
remindctl add --title "Meeting prep" --due "2026-02-15 09:00"
```

### Due Time vs Alarm / Early Nudge

`--due` and `--alarm` are different fields:

- `--due` sets the reminder's due date/time.
- `--alarm` sets the EventKit alarm/notification trigger. Timed due reminders may default to an alarm at the due time, but pass `--alarm` explicitly when the user asks for an earlier nudge.

For a reminder due at 2:00 PM with a notification 30 minutes earlier:

```bash
remindctl add --title "Hairdresser" --due "2026-05-15 14:00" --alarm "2026-05-15 13:30"
```

To edit an existing reminder:

```bash
remindctl edit 87354 --due "2026-05-15 14:00" --alarm "2026-05-15 13:30"
```

The Reminders UI may show or group the item by the alarm time because that is when the notification fires. Verify with JSON instead of assuming the due time moved:

```bash
remindctl today --json
```

Expected shape:

- `dueDate`: actual due time
- `alarmDate`: notification / early nudge time

Apple's public `EKReminder` docs list only reminder-specific properties. Alarm support comes from inherited `EKCalendarItem` behavior exposed by remindctl's `--alarm` flag.

### Complete / Delete

```bash
remindctl complete 1 2 3          # Complete by ID
remindctl delete 4A83 --force     # Delete by ID
```

### Output Formats

```bash
remindctl today --json       # JSON for scripting
remindctl today --plain      # TSV format
remindctl today --quiet      # Counts only
```

## Date Formats

Accepted by `--due` and date filters:
- `today`, `tomorrow`, `yesterday`
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:mm`
- ISO 8601 (`2026-01-04T12:34:56Z`)

## Rules

1. When "remind me" is genuinely ambiguous between Apple Reminders and an agent-delivered alert, infer from context or clarify only if the destination materially changes the outcome.
2. A direct user instruction authorizes the specified reminder mutation; do not ask for duplicate confirmation. Ask only for a missing binding field that cannot be safely inferred.
3. Use `--json` for programmatic parsing. For add/edit, capture the returned reminder ID, then run `remindctl info <id> --json` and compare title, list, `dueDate`, `alarmDate`, recurrence, notes, URL, and priority wherever applicable before reporting success or retrying.
4. For complete/delete, capture the prior ID and list, execute once, then re-list the affected reminder or list with JSON and verify the requested state or absence. Reconcile an uncertain first attempt before retrying.
5. For list create/delete, re-list all lists and verify exact presence or absence. Report partial bulk outcomes explicitly rather than claiming total success.
