"""Respawn backoff for rate-limited Kanban runs: the wait grows with each consecutive quota wall.

Split out of ``hermes_cli.kanban_db_dispatch`` (``check_respawn_guard``), which is over its size cap.
"""

from __future__ import annotations

import sqlite3

# 5 -> 15 -> 45 min with the default 300 s cooldown, then it stays at the cap.
BACKOFF_FACTOR = 3
BACKOFF_MAX_STEPS = 2


def latest_run_and_cooldown(conn: sqlite3.Connection, task_id: str, base_cooldown: int):
    """``(latest ended run or None, cooldown seconds)``: *base_cooldown* times
    ``BACKOFF_FACTOR`` per consecutive trailing ``rate_limited`` run beyond the first."""
    recent = conn.execute(
        "SELECT outcome, ended_at, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT ?",
        (task_id, BACKOFF_MAX_STEPS + 1),
    ).fetchall()
    streak = next((i for i, r in enumerate(recent) if r["outcome"] != "rate_limited"), len(recent))
    return (recent[0] if recent else None), base_cooldown * BACKOFF_FACTOR ** max(streak - 1, 0)
