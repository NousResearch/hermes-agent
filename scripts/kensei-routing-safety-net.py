#!/usr/bin/env python3
"""
Routing safety-net cron (reporting only).

Scans all kanban boards for stalled tasks:
  1. Tasks in ready status for >4h with no pickup -> flag
  2. Tasks blocked for >24h -> flag

NOTE: Auto-assignment was removed — kensei-triage-processor owns triage
routing. This cron only REPORTS stalled work so a human/lead can act.

Runs as no_agent=true — deterministic, zero LLM cost per tick.
Silent when nothing to report.
"""

import os
import sqlite3
import time
from datetime import datetime

HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
STALE_READY_HOURS = 4
STALE_BLOCKED_HOURS = 24

# W1-G (Batch 1): board DB identities resolved via _board_compat so retired
# slugs (default->core, ops->security-ops, content-lead->content) map to the
# current canonical DB path. Semantic board labels (keys) are preserved.
import _board_compat
BOARDS = {slug: str(path) for slug, path in _board_compat.build_board_db_map([
    "default", "ops", "research", "apps", "content-lead",
]).items()}

# Assignees that are real automation profiles (expected to auto-pick work).
KNOWN_BOT_PROFILES = {
    "kensei", "misa-misa", "remii", "wesker", "octacon", "ceecee", "gojo",
    "mrhermagi", "quan", "design-lead", "light", "kensei-review",
    "market-scanner", "denji", "orchestrator", "triage-router",
}
# Assignees that are humans (not expected to "pick up" automatically).
KNOWN_HUMANS = {"sahil"}


def get_boards():
    """Return list of (slug, path) tuples for boards that exist and pass integrity."""
    result = []
    for slug, path in BOARDS.items():
        if os.path.isfile(path):
            try:
                conn = sqlite3.connect(path)
                conn.execute("PRAGMA quick_check")
                conn.close()
                result.append((slug, path))
            except sqlite3.DatabaseError:
                continue
    return result


def staleness_column(conn):
    """Best proxy for 'last activity': updated_at if the board has it, else created_at."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    return "updated_at" if "updated_at" in cols else "created_at"


def query_stale(db_path, status, hours):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        col = staleness_column(conn)
        cutoff = time.time() - (hours * 3600)
        cur = conn.cursor()
        cur.execute(
            f"""SELECT id, title, status, assignee, {col} AS activity_ts
               FROM tasks
               WHERE status = ?
                 AND {col} < ?
               ORDER BY {col} ASC""",
            (status, cutoff),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except sqlite3.DatabaseError:
        return []
    except Exception:
        return []


def fmt_ts(unix_ts):
    """Local server time (BST in summer) — matches Sahil's clock."""
    if not unix_ts:
        return "unknown"
    return datetime.fromtimestamp(unix_ts).strftime("%d %b %H:%M")


def assignee_note(assignee):
    """Human-readable reason a task is stalled."""
    if not assignee or assignee.lower() == "unassigned":
        return "DEAD ASSIGNEE (unassigned) — reassign"
    low = assignee.lower()
    if low in KNOWN_HUMANS:
        return f"awaiting human ({assignee})"
    if low in KNOWN_BOT_PROFILES:
        return f"{assignee} hasn't picked up"
    # Not a known human or bot — dead/unknown assignee that needs reassigning.
    return f"DEAD ASSIGNEE ({assignee}) — reassign"


def main():
    reports = []
    total_stale_ready = 0
    total_stale_blocked = 0

    for board_slug, db_path in get_boards():
        for task in query_stale(db_path, "ready", STALE_READY_HOURS):
            assignee = task.get("assignee") or "unassigned"
            reports.append(
                f"STALE {task['id'][:10]} on {board_slug}: ready since "
                f"{fmt_ts(task['activity_ts'])} → {assignee_note(assignee)}"
            )
            total_stale_ready += 1

        for task in query_stale(db_path, "blocked", STALE_BLOCKED_HOURS):
            assignee = task.get("assignee") or "unassigned"
            reports.append(
                f"STALL {task['id'][:10]} on {board_slug}: blocked since "
                f"{fmt_ts(task['activity_ts'])} → {assignee_note(assignee)}"
            )
            total_stale_blocked += 1

    if not reports:
        return

    lines = ["## Routing Safety-Net Scan", ""]
    lines.append(f"**{total_stale_ready} stale ready, {total_stale_blocked} stale blocked**")
    lines.append("")
    lines.append("```")
    for r in reports:
        lines.append(r)
    lines.append("```")
    lines.append("")
    lines.append("*Reporting only (no auto-assign). Runs every 6h. Silent when nothing to report.*")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
