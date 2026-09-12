"""Kanban running-task snapshot for /agents.

The kanban dispatcher spawns worker processes OUTSIDE any agent session
(``hermes -p <profile> chat -q work kanban task <id>``), so neither the CLI
nor the gateway ``/agents`` handler can see them via the process registry or
async-delegation list — the workers are not that session's children. The
board, however, is shared SQLite across profiles BY DESIGN
(``kanban_db.kanban_home``), so the authoritative "what is running" answer is
one read-only query per board.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as _kb


def _read_only_rows(db_path: Path) -> list[sqlite3.Row]:
    """Query running tasks from one board DB without initializing it.

    Existence is checked BEFORE connect (sqlite3 would create an empty file
    otherwise) and no schema SQL is ever executed, so a status read never
    initializes or migrates a board. Plain connect (not ``mode=ro``) because a
    read-only URI cannot read a WAL board with a live writer (needs a
    writable ``-shm``); only SELECTs are issued, and ``no such table`` (fresh,
    schema-less file) is answered with an empty list.
    """
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # name-based access; default is a bare tuple
    try:
        return conn.execute(
            "SELECT id, title, assignee, started_at FROM tasks "
            "WHERE status = 'running' "
            "ORDER BY (started_at IS NULL), started_at ASC, created_at ASC",
        ).fetchall()
    except sqlite3.OperationalError as exc:
        # Fresh install: the DB file exists but has no schema yet.
        if "no such table" in str(exc).lower():
            return []
        raise
    finally:
        conn.close()


def list_running_tasks() -> list[dict[str, Any]]:
    """Running kanban tasks across every board (oldest-started first).

    Returns dicts with ``task_id``, ``title``, ``assignee``, ``board``,
    ``started_at``. Fail-open: unreadable boards are skipped, never raised —
    /agents is a status command and must not crash on a locked or partial
    board. An import failure (kanban module absent in stripped installs)
    surfaces as ``[]`` via the callers' ``_probe``/``_quiet_sync`` wrappers.
    """
    import time

    rows: list[dict[str, Any]] = []
    try:
        boards = [meta.get("slug", _kb.DEFAULT_BOARD)
                  for meta in _kb.list_boards(include_archived=False)]
    except Exception:
        return []
    for board in boards:
        try:
            db_path = _kb.kanban_db_path(board=board)
            for row in _read_only_rows(db_path):
                rows.append({
                    "task_id": row["id"],
                    "title": row["title"] or "",
                    "assignee": row["assignee"],
                    "board": board,
                    "started_at": row["started_at"],
                })
        except Exception:
            continue
    now = int(time.time())
    for row in rows:
        started = row.get("started_at")
        row["elapsed_seconds"] = max(0, now - int(started)) if started else None
    return rows


def running_count() -> Optional[int]:
    """Total running kanban tasks, or ``None`` when the board is unreadable."""
    try:
        return len(list_running_tasks())
    except Exception:
        return None
