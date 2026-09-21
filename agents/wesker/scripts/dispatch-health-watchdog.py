#!/usr/bin/env python3
"""Dispatch Health Watchdog — monitors kanban task health across all boards.

Read-only. Queries SQLite kanban DBs and reports:
- Stuck tasks (running >30min without heartbeat)
- Oldest unassigned ready/todo tasks
- Crash/fail events in last 24h
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# The wesker profile sets $HOME to a fake home (/home/kensei/.hermes/profiles/wesker/home).
# Check if the real /home/kensei/.hermes/kanban exists there (it won't in a fake home).
# If not, use the canonical /home/kensei.
_candidate = Path(os.environ.get('HOME', '/home/kensei'))
if not (_candidate / ".hermes" / "kanban").exists():
    _real_home = Path('/home/kensei')
else:
    _real_home = _candidate

NOW = int(datetime.now(timezone.utc).timestamp())
STUCK_THRESHOLD_S = 1800  # 30 minutes
CRASH_WINDOW_S = 86400    # 24 hours

def _kanban_dirs():
    """Yield (board_name, db_path) pairs, deduplicated by real path."""
    seen = set()
    base = _real_home / ".hermes" / "kanban"

    # Central boards: these are canonical
    boards_dir = base / "boards"
    if boards_dir.exists():
        for bdir in sorted(boards_dir.iterdir()):
            bfile = (bdir / "kanban.db").resolve()
            if bfile.exists() and str(bfile) not in seen:
                seen.add(str(bfile))
                yield (bdir.name, bfile)

    # Root kanban.db
    root_db = base / "kanban.db"
    if root_db.exists():
        rp = root_db.resolve()
        if str(rp) not in seen:
            seen.add(str(rp))
            yield ("default", rp)

    # Profile-level boards: only include if NOT a duplicate of central
    profiles_dir = _real_home / ".hermes" / "profiles"
    if profiles_dir.exists():
        for pdir in sorted(profiles_dir.iterdir()):
            if not pdir.is_dir():
                continue
            # Profile root kanban.db
            pboard = (pdir / "kanban.db").resolve()
            if pboard.exists() and str(pboard) not in seen:
                seen.add(str(pboard))
                yield (f"profile-{pdir.name}", pboard)
            # Profile board subdirs
            subboards = pdir / "kanban" / "boards"
            if subboards.exists():
                for bdir in sorted(subboards.iterdir()):
                    bfile = (bdir / "kanban.db").resolve()
                    if bfile.exists() and str(bfile) not in seen:
                        seen.add(str(bfile))
                        yield (f"profile-{pdir.name}-{bdir.name}", bfile)

BOARDS = list(_kanban_dirs())


def db_query(db_path, query, params=()):
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return {"error": str(e)}


report = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "boards": {},
    "stuck_tasks": [],
    "old_unassigned_tasks": [],
    "crashes_24h": [],
    "summary": {},
}

total_stuck = 0
total_old_unassigned = 0
total_crashes = 0
board_counts = {}
crash_dedup = set()

for board_name, db_path in BOARDS:
    if not db_path.exists():
        continue

    board_data = {"stuck": 0, "old_unassigned": 0, "crashes": 0, "total_tasks": 0}

    # Task counts by status
    status_rows = db_query(db_path, "SELECT status, count(*) as cnt FROM tasks GROUP BY status")
    if isinstance(status_rows, list):
        board_data["status_breakdown"] = {r["status"]: r["cnt"] for r in status_rows}
        board_data["total_tasks"] = sum(r["cnt"] for r in status_rows)

    # Stuck tasks: running > 30 min without heartbeat
    stuck = db_query(
        db_path,
        """SELECT id, title, status, assignee, started_at, last_heartbeat_at,
                  round((? - COALESCE(started_at, 0))/60.0, 1) as running_min
           FROM tasks
           WHERE status = 'running'
             AND started_at IS NOT NULL
             AND (? - started_at) > ?""",
        (NOW, NOW, STUCK_THRESHOLD_S)
    )
    if isinstance(stuck, list):
        for t in stuck:
            t["board"] = board_name
            # Check heartbeat staleness
            hb_stale = False
            if t.get("last_heartbeat_at"):
                hb_age = NOW - t["last_heartbeat_at"]
                if hb_age > 600:  # no heartbeat in 10 min
                    hb_stale = True
            t["heartbeat_stale"] = hb_stale
            report["stuck_tasks"].append(t)
            total_stuck += 1
            board_data["stuck"] += 1

    # Oldest ready/todo tasks (unassigned or no movement)
    old_unassigned = db_query(
        db_path,
        """SELECT id, title, status, assignee, priority,
                  round((? - created_at)/3600.0, 1) as age_hours
           FROM tasks
           WHERE (status = 'ready' OR status = 'todo')
           ORDER BY created_at ASC
           LIMIT 10""",
        (NOW,)
    )
    if isinstance(old_unassigned, list):
        for t in old_unassigned:
            t["board"] = board_name
            report["old_unassigned_tasks"].append(t)
            total_old_unassigned += 1
            board_data["old_unassigned"] += 1

    # Crash/fail events in last 24h
    crashes = db_query(
        db_path,
        """SELECT t.id, t.title, r.outcome, r.started_at, r.ended_at, r.error, r.profile
           FROM task_runs r
           JOIN tasks t ON r.task_id = t.id
           WHERE r.outcome IN ('crashed', 'timed_out', 'spawn_failed', 'failed', 'gave_up')
             AND r.ended_at IS NOT NULL
             AND (? - r.ended_at) < ?
           ORDER BY r.ended_at DESC""",
        (NOW, CRASH_WINDOW_S)
    )
    if isinstance(crashes, list):
        for c in crashes:
            c["board"] = board_name
            # Deduplicate by (task_id, run_id) across duplicated boards
            dedup_key = (c["id"], c.get("started_at"), c.get("ended_at"))
            if dedup_key in crash_dedup:
                continue
            crash_dedup.add(dedup_key)
            if c["started_at"]:
                c["run_duration_s"] = c["ended_at"] - c["started_at"] if c["ended_at"] else None
            report["crashes_24h"].append(c)
            total_crashes += 1
            board_data["crashes"] += 1

    board_counts[board_name] = board_data
    report["boards"][board_name] = board_data

# Summary
report["summary"] = {
    "total_stuck": total_stuck,
    "total_old_unassigned": total_old_unassigned,
    "total_crashes_24h": total_crashes,
    "boards_checked": len(board_counts),
    "status": "all_clear" if (total_stuck == 0 and total_crashes == 0 and total_old_unassigned == 0) else "attention_needed",
}

# Also produce a human-readable Telegram-friendly message
lines = []
lines.append(f"**Dispatch Health Watchdog — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}**")
lines.append("")

if total_stuck > 0:
    lines.append(f"**Stuck Tasks ({total_stuck})**")
    for t in report["stuck_tasks"]:
        stale_marker = " - stale heartbeat" if t.get("heartbeat_stale") else ""
        assignee = t.get("assignee", "?") or "?"
        lines.append(f"  * [{t['board']}] `{t['id'][:12]}` {t['title']} — running {t['running_min']}m (assigned: {assignee}){stale_marker}")

if total_old_unassigned > 0:
    lines.append(f"**Oldest Pending Tasks ({total_old_unassigned})**")
    for t in report["old_unassigned_tasks"][:5]:
        assignee = t.get("assignee") or "unassigned"
        lines.append(f"  • [{t['board']}] `{t['id'][:12]}` {t['title']} — {t['age_hours']}h old, assignee: {assignee}")

if total_crashes > 0:
    lines.append(f"**Failures (24h): {total_crashes} events**")
    # Group by task to deduplicate
    crash_by_task = defaultdict(list)
    for c in report["crashes_24h"]:
        crash_by_task[c["id"]].append(c)
    for task_id, events in sorted(crash_by_task.items())[:8]:
        title = events[0]["title"]
        board = events[0]["board"]
        count = len(events)
        latest = events[-1]
        error = latest.get("error") or ""
        lines.append(f"  • [{board}] `{task_id[:12]}` {title} — {count}x crashes, last: {latest['outcome']}")
        if error and len(error) < 100:
            lines.append(f"    └ {error}")

if total_stuck == 0 and total_crashes == 0 and total_old_unassigned == 0:
    lines.append("✅ All clear — no stuck tasks, no recent crashes, no old pending tasks.")

# Board-level status summary
lines.append("")
board_summaries = sorted((n, d["total_tasks"]) for n, d in board_counts.items())
lines.append("**Boards:** " + ", ".join(f"{n}({c} tasks)" for n, c in board_summaries))

report["telegram_message"] = "\n".join(lines)

# Output: with no_agent=true in cron, stdout IS the delivered message.
# Non-empty stdout → delivered. Empty stdout → silent.
if report["summary"]["status"] == "all_clear":
    sys.exit(0)
else:
    print(report["telegram_message"])
    sys.exit(1)
