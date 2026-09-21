---
name: blocked-task-escalator
description: Deploy a no-agent watchdog cron that monitors all kanban boards for tasks blocked >60min and alerts via Telegram. Silent when clean. Reusable for any profile that needs blocked-task visibility.
version: 1.0.0
---

# Blocked Task Escalator

## When to use

- Any time blocked kanban tasks are being missed (no human notification)
- After deploying a new board where blocked tasks would otherwise sit silently
- As standard ops hygiene for any profile with a running gateway

## Implementation

1. Create the watchdog script at `~/.hermes/scripts/blocked-task-escalator.py` (see template below)
2. Deploy as `no_agent=True` cron on the profile that has the running Telegram gateway (usually `default`):
   ```
   cronjob(action='create', name='kensei-blocked-task-escalator', schedule='every 30m',
           script='blocked-task-escalator.py', no_agent=True,
           profile='default', deliver='all')
   ```
3. Verify: run `python3 ~/.hermes/scripts/blocked-task-escalator.py` to see current state

## Script template

```python
#!/usr/bin/env python3
import sqlite3, os, time

HERMES_HOME = "/home/kensei/.hermes"

BOARDS = {
    "default":   f"{HERMES_HOME}/kanban.db",
    "apps":      f"{HERMES_HOME}/kanban/boards/apps/kanban.db",
    "content-lead": f"{HERMES_HOME}/kanban/boards/content-lead/kanban.db",
    "ops":       f"{HERMES_HOME}/kanban/boards/ops/kanban.db",
    "research":  f"{HERMES_HOME}/kanban/boards/research/kanban.db",
}

ESCALATION_AGE = 3600
now = int(time.time())
lines = []
total_blocked = 0

for slug, db_path in BOARDS.items():
    if not os.path.isfile(db_path):
        continue
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, title, assignee, created_at FROM tasks WHERE status = 'blocked' ORDER BY created_at ASC")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        continue
    for row in rows:
        age = now - row["created_at"]
        if age < ESCALATION_AGE:
            continue
        total_blocked += 1
        age_h = age / 3600.0
        lines.append(f"  {row['id']} | {slug:13s} | {(row['assignee'] or 'unassigned'):15s} | {age_h:.1f}h | {row['title'][:80]}")

if not lines:
    exit(0)

header = f"⚠️  {total_blocked} task(s) blocked >60min across kanban boards"
print(header)
print()
print("  ID             | Board         | Assignee        | Age     | Title")
print("  ───────────────┼───────────────┼─────────────────┼─────────┼────────────────────────────────────────────")
for l in lines:
    print(l)
```

## Pitfalls

- Use absolute paths (not `~`) because cron HOME varies by profile (wesker's `~` resolves to `/home/kensei/.hermes/profiles/wesker/home/`)
- Must deploy on the profile with the running gateway, otherwise `deliver='all'` goes nowhere
- Script path for cron's `script=` field resolves relative to the profile's `scripts/` dir — for default profile that's `~/.hermes/scripts/`
- `no_agent=True` means zero token cost per run — the script's stdout IS the delivery message
