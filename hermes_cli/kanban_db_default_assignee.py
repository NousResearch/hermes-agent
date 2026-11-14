"""``kanban.default_assignee`` fill-in for unassigned ready cards during a dispatch tick.

Split out of ``hermes_cli.kanban_db_dispatch``; origin-resident helpers are
reached late-bound via ``_kb`` (import-cycle breaking).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_db_dispatch import DispatchResult

# The origin module's logger, so existing kanban log filters keep matching.
_log = logging.getLogger("hermes_cli.kanban_db")


def apply_default_assignee(
    conn: sqlite3.Connection, task_id: str, assignee: str, *, dry_run: bool,
) -> bool:
    """Persist ``kanban.default_assignee`` on an unassigned ready row.

    Mutating the row keeps board state honest: the task is legitimately owned
    by the default, not "unassigned but secretly routed". ``dry_run`` reports
    without writing. Returns False when the write failed or a concurrent
    assign won the row.
    """
    if dry_run:
        return True
    try:
        with _kb.write_txn(conn):
            updated = conn.execute(
                "UPDATE tasks SET assignee = ? WHERE id = ? "
                "AND (assignee IS NULL OR assignee = '')",
                (assignee, task_id),
            )
            if updated.rowcount != 1:
                # The card was assigned after the tick's read; do not log a phantom assignment.
                return False
            _kb._append_event(
                conn, task_id, "assigned",
                {"assignee": assignee, "source": "kanban.default_assignee"},
            )
    except Exception:  # health: allow BLE001 -- per-card tick boundary moved from kanban_db_dispatch; logs exc_info
        _log.debug(
            "kanban dispatch: failed to apply default_assignee=%r to task %s",
            assignee, task_id, exc_info=True,
        )
        return False
    return True


def fill_default_assignee(
    conn: sqlite3.Connection, task_id: str, default_assignee: Optional[str],
    result: DispatchResult, *, dry_run: bool,
) -> Optional[str]:
    """Assignee to dispatch an unassigned ready card as, or None to skip it this tick.

    Honours ``kanban.default_assignee`` so an unassigned task doesn't park in
    'ready' forever. A card a concurrent assign claimed first is deferred to the
    next tick (which gates its real assignee), not bucketed as unassigned.
    """
    if not default_assignee:
        result.skipped_unassigned.append(task_id)
        return None
    if not apply_default_assignee(conn, task_id, default_assignee, dry_run=dry_run):
        current = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if current is None or not current["assignee"]:
            result.skipped_unassigned.append(task_id)
        return None
    result.auto_assigned_default.append(task_id)
    return default_assignee


# Late-bound origin namespace (see module docstring).
from hermes_cli import kanban_db as _kb
