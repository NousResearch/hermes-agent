#!/usr/bin/env python3
"""Blocked-task escalator — scans boards and alerts on newly stale blockers.

Runs as a no_agent=True cron. Output delivered verbatim to Discord.
Silent when no blocker newly crosses the threshold or changes state.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

HERMES_HOME = Path("/home/kensei/.hermes")
STATE_FILE = HERMES_HOME / "governance" / "blocked-task-escalator-state.json"

BOARDS = {
    "default": HERMES_HOME / "kanban.db",
    "apps": HERMES_HOME / "kanban" / "boards" / "apps" / "kanban.db",
    "content-lead": HERMES_HOME / "kanban" / "boards" / "content-lead" / "kanban.db",
    "ops": HERMES_HOME / "kanban" / "boards" / "ops" / "kanban.db",
    "research": HERMES_HOME / "kanban" / "boards" / "research" / "kanban.db",
}

ESCALATION_AGE = 3600  # 60 minutes since latest blocked event, not task creation
now = int(time.time())


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"blocked": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def latest_blocked_at(conn: sqlite3.Connection, task_id: str, fallback_created_at: int) -> int:
    try:
        row = conn.execute(
            "SELECT MAX(created_at) FROM task_events WHERE task_id = ? AND kind = 'blocked'",
            (task_id,),
        ).fetchone()
        if row and row[0]:
            return int(row[0])
    except Exception:
        pass
    return int(fallback_created_at or now)


def scan_board(slug: str, db_path: Path) -> list[dict]:
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, title, assignee, created_at
        FROM tasks
        WHERE status = 'blocked'
        ORDER BY created_at ASC
        """
    ).fetchall()
    findings = []
    for row in rows:
        blocked_at = latest_blocked_at(conn, row["id"], row["created_at"])
        age = now - blocked_at
        if age < ESCALATION_AGE:
            continue
        findings.append({
            "key": f"{slug}:{row['id']}",
            "id": row["id"],
            "board": slug,
            "assignee": row["assignee"] or "unassigned",
            "age_h": age / 3600.0,
            "blocked_at": blocked_at,
            "title": (row["title"] or "")[:80],
        })
    conn.close()
    return findings


state = load_state()
previous = state.get("blocked", {})
current = {}
new_or_changed = []

for slug, db_path in BOARDS.items():
    try:
        for item in scan_board(slug, db_path):
            current[item["key"]] = item
            old = previous.get(item["key"])
            # Alert once per blocked episode. If the task is unblocked and blocked again,
            # blocked_at changes and it alerts once for the new episode.
            if not old or old.get("blocked_at") != item["blocked_at"]:
                new_or_changed.append(item)
    except Exception:
        # no_agent watchdog: keep scanning other boards; failures should be explicit.
        new_or_changed.append({
            "key": f"{slug}:scan-error",
            "id": "scan-error",
            "board": slug,
            "assignee": "n/a",
            "age_h": 0.0,
            "blocked_at": now,
            "title": f"Failed to scan {db_path}",
        })

save_state({"updated_at": now, "blocked": current})

if not new_or_changed:
    raise SystemExit(0)

print(f"⚠️ Blocked task escalator · {time.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"new stale blockers: {len(new_or_changed)} · current stale blockers: {len(current)} · boards scanned: {sum(1 for b in BOARDS.values() if b.is_file())}")
print()
print("ID         | Board         | Assignee        | Blocked age | Title")
print("───────────┼───────────────┼─────────────────┼─────────────┼────────────────────────────────────────────")
for item in sorted(new_or_changed, key=lambda x: (-x["age_h"], x["board"], x["id"])):
    print(f"{item['id'][:10]:10s} | {item['board'][:13]:13s} | {item['assignee'][:15]:15s} | {item['age_h']:.1f}h        | {item['title']}")
