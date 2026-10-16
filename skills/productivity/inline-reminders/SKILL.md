---
name: inline-reminders
description: "Natural-language reminders: 'remind me Tuesday 8am to call the plumber'. Capture, list, and cancel one-shot reminders."
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Reminders, Alerts, Scheduling, Productivity]
    related_skills: []
---

# Inline Reminders

Capture, list, and cancel one-shot reminders spoken in natural language.
Reminders fire into the chat where they were captured (fallback: home channel)
via a no_agent cron poller that costs zero tokens on quiet runs.

This is a transactional queue, not a cron registry — one-shot reminders add
themselves to a queue file and expire on fire.  No cron-registration cleanup.

## When to Use

- "Remind me Tuesday 8am to call the plumber"
- "Remind me in 20 minutes to check the oven"
- "Remind me tomorrow 9am to send the report"
- "What reminders do I have this week?" / "What's due?"
- "Cancel the reminder about the plumber" / "Forget the reminder about X"
- "Every Tuesday at 8am, remind me to take out the trash" (recurring → repeat-flag on the queue entry; the poller re-arms it after each fire)

Don't use for: alarms with exact-second precision (latency bound = poll
cadence, ~1 min).  Don't use for: recurrence beyond daily/weekly (e.g. "every
2 hours") — use a cron job directly for those.

## How It Works

Three layers, all local — no LLM in the wait/fire path:

1. **Capture** (this skill, agent turn): parse the user's "remind me" request,
   compute `due_at`, rephrase the message relative to fire time, write to the
   queue file, confirm.
2. **Wait** (no_agent cron poller): `reminder_poller.py` runs every ~1 min via
   the existing no_agent cron mechanism.  Empty stdout = silent.  Zero tokens
   on idle.
3. **Fire**: poller sees `due_at` passed → prints the message to stdout →
   scheduler delivers verbatim via the job's configured delivery channel
   (v1: home channel — the entry's `origin` is recorded for future per-chat
   routing) → entry marked `fired` and moved to the append-only fired log;
   recurring entries are re-armed for their next occurrence on the same poll.

## Procedure

> Windows note: the shell snippets below use `python -c` with nested quotes,
> which can mangle through cmd on this box.  If a snippet misbehaves, write
> the call to a small temp `.py` probe and run that instead.

### 1. Parse the request

Extract two things from the user's message:
- **when**: the time expression ("Tuesday 8am", "in 20 minutes", "tomorrow 9am", "today 6pm")
- **what**: the reminder message ("call the plumber", "check the oven")

If the user says "every <interval>" (e.g. "every Tuesday 8am"), this is a
**recurring** reminder — see step 5 (graduate to cron job).

### 2. Compute due_at

Run the queue module's parser to convert the time expression to a
timezone-aware datetime:

```bash
python -c "
import sys; sys.path.insert(0, r'<HERMES_AGENT_ROOT>')
from cron.reminder_queue import parse_when
from hermes_time import now
print(parse_when('<when>', now()).isoformat())
"
```

Replace `<HERMES_AGENT_ROOT>` with the hermes-agent root (the directory
containing `hermes_constants.py`).  The parser handles:
- Relative: "in 20 minutes", "in 2 hours", "in 3 days"
- Absolute: "tuesday 8am", "tue 8am", "tomorrow 8am", "today 6pm"
- ISO-8601: "2026-08-28 08:00"

The timezone is the configured Hermes timezone (SAST, UTC+2 on this host —
no DST, so "Tuesday 8am" is unambiguous).

### 3. Rephrase the message relative to fire time

If the user said "remind me tomorrow to call the plumber" and the reminder
fires tomorrow, the delivered message should say "call the plumber" (not
"tomorrow, call the plumber" — at fire time it IS today).  Rephrase:
- "tomorrow" → "today" (if firing same-day or next-day relative to capture)
- "next week" → "this week"
- Remove redundant time words — the delivery context has the fire time.

### 4. Write to the queue and confirm

Use the terminal tool to run:

```bash
python -c "
import sys; sys.path.insert(0, r'<HERMES_AGENT_ROOT>')
from cron.reminder_queue import add_reminder
from datetime import datetime
entry = add_reminder(
    due_at=datetime.fromisoformat('<due_at_iso>'),
    message='<rephrased_message>',
    origin={'platform': '<platform>', 'chat_id': '<chat_id>', 'thread_id': '<thread_id>'},
)
print(f'ID: {entry[\"id\"]}')
"
```

Origin: record the current chat's platform and chat_id so the reminder
fires back into THIS conversation.  If you can't determine the origin
(unknown platform/chat), pass `origin=None` — the poller will deliver to
the home channel fallback.

Confirm to the user: "✅ Set for <local time>.  ID: <id>"

Format the time in local time (the configured timezone), e.g. "Tue 08:00"
or "2026-08-28 08:00".

### 5. Recurring reminders → repeat-flag on the queue entry

If the user says "every Tuesday 8am" or "every day at 6pm", this is a
recurring cadence.  It still lives in the queue — as an entry with a
structured `recurring` rule.  After the entry fires, the poller computes the
next occurrence deterministically (no LLM) and re-adds the entry with the
same message, origin and rule.  Every occurrence lands in `fired.log` as its
own record, and the reminder never stops until cancelled.

Build the rule at capture:

- "every Tuesday 8am" →
  `recurring={"kind": "weekly", "weekday": 1, "time": "08:00"}`,
  first `due_at` = `parse_when("tuesday 8am")` (the parser already rolls to
  next week if today's occurrence has passed).
- "every day at 6pm" →
  `recurring={"kind": "daily", "time": "18:00"}`,
  first `due_at` = `parse_when("today 6pm")`; if that is already in the past,
  use `parse_when("tomorrow 6pm")`.

Confirm to the user: "✅ Every Tue 08:00 (first: 2026-08-31 08:00)."

Do NOT create a cron job for recurring reminders — the poller already is the
scheduler, and queue entries stay on the drift-immune no-agent path (no model
resolution, no pinning, zero tokens).  Only the poller itself is a registered
cron job (see Setup).  Catch-up for a missed occurrence follows the fire-late
default; the next occurrence then re-anchors from the poll tick.

### 6. List pending reminders

When the user asks "what's due?" or "what reminders do I have?":

```bash
python -c "
import sys; sys.path.insert(0, r'<HERMES_AGENT_ROOT>')
from cron.reminder_queue import list_pending
import json
for e in list_pending():
    print(f'{e[\"id\"]}  {e[\"due_at\"]}  {e[\"message\"]}')
"
```

Reply with the list, soonest first.  If empty: "No pending reminders."

### 7. Cancel a reminder

When the user says "cancel the reminder about X" or "forget the reminder
about the plumber":

```bash
python -c "
import sys; sys.path.insert(0, r'<HERMES_AGENT_ROOT>')
from cron.reminder_queue import cancel_by_query
cancelled = cancel_by_query('<query>')
for e in cancelled:
    print(f'Cancelled: {e[\"message\"]}')
"
```

If the user gives an ID ("cancel reminder abc123"), use `cancel_reminder(id)`
instead.  Confirm: "✅ Cancelled: <message>" or "No matching reminder found."

## Catch-up policy

If the machine was asleep at `due_at`, the reminder fires on the next poll
tick (fire-late is the default).  The poller does NOT skip stale reminders —
better late than never.  A skip-if-stale policy is TBD per the issue.

## Verification

After capturing a reminder, verify it landed:

```bash
python -c "
import sys; sys.path.insert(0, r'<HERMES_AGENT_ROOT>')
from cron.reminder_queue import list_pending
for e in list_pending():
    print(f'{e[\"id\"]}  {e[\"due_at\"]}  {e[\"message\"]}')
"
```

You should see the entry you just added.

## Setup (one-time, for the poller)

The poller must be installed as a no_agent cron job to actually fire
reminders.  This is a one-time setup step:

1. Copy the poller script to HERMES_HOME/scripts/:
   ```bash
   cp <HERMES_AGENT_ROOT>/cron/scripts/reminder_poller.py <HERMES_HOME>/scripts/
   ```

2. Create the no_agent cron job:
   ```bash
   hermes cron create '* * * * *' "reminder poller" \
       --name reminder-poller \
       --script reminder_poller.py \
       --no-agent \
       --deliver telegram
   ```

3. Verify: `hermes cron list` — the poller should appear with `no_agent: true`.

The poller is silent when no reminders are due (empty stdout = no delivery),
so it costs zero tokens on idle.
