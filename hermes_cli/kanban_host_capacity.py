"""Host-level capacity accounting for the kanban dispatcher.

Caps bound the HOST, not one board: every active board's tick sees only its own
DB, so without counting the other boards a derived cap of N is multiplied by the
number of boards. Kept beside the dispatcher rather than inside it to keep that
facade within its size budget.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import Optional


def _kb():
    from hermes_cli import kanban_db
    return kanban_db


def _kbc():
    from hermes_cli import kanban_db_connect
    return kanban_db_connect


def count_running_tasks(conn: sqlite3.Connection) -> int:
    """Number of tasks in ``status='running'``.

    Used by the multi-board sweep to count OTHER boards' workers against the
    host-level budget. Fails open to 0 so a broken board doesn't brick dispatch
    on healthy ones.
    """
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )
    except Exception:
        return 0


def count_running_tasks_other_boards(board: Optional[str] = None) -> int:
    """Total ``running`` tasks across every board EXCEPT ``board``.

    Caps bound the HOST, but each board's tick only sees its own DB; without
    this a derived cap of N gets multiplied by the number of active boards.
    Boards are matched by resolved DB path, so ``HERMES_KANBAN_DB`` (pins every
    board to one file) yields 0. Fails open per board.
    """
    try:
        current_path = str(_kb().kanban_db_path(board=board).expanduser().resolve())
    except Exception:
        current_path = None
    try:
        boards = _kb().list_boards(include_archived=False)
    except Exception:
        return 0
    total = 0
    for meta in boards:
        slug = meta.get("slug") or _kb().DEFAULT_BOARD
        try:
            path = _kb().kanban_db_path(board=slug).expanduser()
            resolved = str(path.resolve())
            if current_path is not None and resolved == current_path:
                continue
            if not path.exists():
                continue
            other = _kbc().connect(board=slug)
            try:
                total += count_running_tasks(other)
            finally:
                with contextlib.suppress(Exception):
                    other.close()
        except Exception:
            continue
    return total
