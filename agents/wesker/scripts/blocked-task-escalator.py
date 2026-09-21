#!/usr/bin/env python3
"""Blocked-task escalator — scans all boards and alerts on tasks blocked >60min.

Runs as a no_agent=True cron on wesker profile. Output delivered verbatim to #ops.
Silent when nothing to report.

Uses absolute paths (not ~) because wesker's cron HOME is its profile dir.
"""

import sqlite3
import os
import time

HERMES_HOME = "/home/kensei/.hermes"

BOARDS = {
    "default":   f"{HERMES_HOME}/kanban.db",
    "apps":      f"{HERMES_HOME}/kanban/boards/apps/kanban.db",
    "content-lead": f"{HERMES_HOME}/kanban/boards/content-lead/kanban.db",
    "ops":       f"{HERMES_HOME}/kanban/boards/ops/kanban.db",
    "research":  f"{HERMES_HOME}/kanban/boards/research/kanban.db",
}

ESCALATION_AGE = 3600  # 60 minutes in seconds
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
        cur.execute("""
            SELECT id, title, assignee, created_at
            FROM tasks
            WHERE status = 'blocked'
            ORDER BY created_at ASC
        """)
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        continue

    for row in rows:
        age = now - row["created_at"]
        if age < ESCALATION_AGE:
            continue
        total_blocked += 1
        age_h = age / 3600.0
        title = row["title"][:80]
        assignee = row["assignee"] or "unassigned"
        lines.append(f"  {row['id']} | {slug:13s} | {assignee:15s} | {age_h:.1f}h | {title}")

if not lines:
    exit(0)

header = f"⚠️  {total_blocked} task(s) blocked >60min across kanban boards"
print(header)
print()
print("  ID             | Board         | Assignee        | Age     | Title")
print("  ───────────────┼───────────────┼─────────────────┼─────────┼────────────────────────────────────────────")
for l in lines:
    print(l)

print()
print(f"Next check: +30min | Boards scanned: {sum(1 for b in BOARDS.values() if os.path.isfile(b))}")
