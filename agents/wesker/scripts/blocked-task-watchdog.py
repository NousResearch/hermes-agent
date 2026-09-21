#!/usr/bin/env python3
"""Kanban blocked-task watchdog — script-only cron (no_agent=True).

Queries the main kanban DB for tasks blocked >30 min.
Silent when clear; delivers alert text when thresholds breached.

Path: ~/.hermes/profiles/wesker/scripts/blocked-task-watchdog.py
"""

import os
import sqlite3
import sys
import time

# Check multiple possible locations for the kanban DB
_HERMES_HOME = os.environ.get("HERMES_HOME", "")
_POSSIBLE_DBS = [
    "/home/kensei/.hermes/kanban.db",
    os.path.expanduser("~/.hermes/kanban.db"),
    os.path.expanduser("~/.hermes/kanban/kanban.db"),
]
if _HERMES_HOME:
    _POSSIBLE_DBS.insert(0, os.path.join(_HERMES_HOME, "kanban", "kanban.db"))

KANBAN_DB = next((p for p in _POSSIBLE_DBS if os.path.isfile(p)), "")
BLOCKED_AGE_MINUTES = 30  # Alert if blocked longer than this
BLOCKED_AGE_SECONDS = BLOCKED_AGE_MINUTES * 60

if not os.path.isfile(KANBAN_DB):
    print(f"SKIP: kanban DB not found at {KANBAN_DB}")
    sys.exit(0)

try:
    conn = sqlite3.connect(KANBAN_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Find all currently blocked tasks
    cur.execute("""
        SELECT id, title, assignee, status, created_at, started_at
        FROM tasks
        WHERE status = 'blocked'
          AND started_at IS NOT NULL
        ORDER BY started_at ASC
    """)

    blocked = cur.fetchall()
    now = int(time.time())
    alerts = []

    for row in blocked:
        blocked_since = row["started_at"]
        age_seconds = now - blocked_since
        age_minutes = age_seconds // 60

        if age_seconds >= BLOCKED_AGE_SECONDS:
            # Fetch the most recent block event reason
            cur.execute("""
                SELECT payload, created_at
                FROM task_events
                WHERE task_id = ? AND kind = 'blocked'
                ORDER BY created_at DESC
                LIMIT 1
            """, (row["id"],))
            block_event = cur.fetchone()
            reason = ""
            if block_event and block_event["payload"]:
                import json
                try:
                    p = json.loads(block_event["payload"])
                    reason = p.get("reason", "")
                except (json.JSONDecodeError, TypeError):
                    reason = str(block_event["payload"])

            alerts.append({
                "id": row["id"],
                "title": row["title"],
                "assignee": row["assignee"] or "unassigned",
                "age_min": age_minutes,
                "reason": reason,
            })

    conn.close()

    if not alerts:
        sys.exit(0)  # Silent — no news is good news

    # Format alert
    print(f"⛔ KANBAN BLOCKED TASK ALERT ({len(alerts)} task(s) blocked >{BLOCKED_AGE_MINUTES}m)")
    print()
    for a in alerts:
        reason_line = f"  Reason: {a['reason']}" if a['reason'] else ""
        print(f"  * {a['id']} ({a['assignee']}, {a['age_min']}m)")
        print(f"    {a['title']}")
        if reason_line:
            print(reason_line)
        print()

    print("To resolve: hermes kanban unblock <id> (with context) or reassign via dashboard.")

except Exception as e:
    print(f"ERROR in blocked-task-watchdog: {e}")
    sys.exit(0)  # Don't alert on script errors — just log silently
