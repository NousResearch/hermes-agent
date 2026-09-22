"""Mirror coordinator task outcomes onto the Mac-local Kanban board."""

from __future__ import annotations

import logging

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_log = logging.getLogger(__name__)


def mirror_fleet_task(task_id: str, status: str, *, result: str | None = None, error: str | None = None) -> bool:
    """Apply a remote terminal state to its reserved local fleet card.

    This runs only on the coordinator host. The Windows runner never opens or
    copies the Mac Kanban database.
    """
    with kbc.connect_closing() as connection:
        task = kb.get_task(connection, task_id)
        if task is None or not kb.is_federated_task(task):
            return False
        if status == "completed":
            return kb.complete_task(
                connection,
                task_id,
                result=(result or "Completed by a federated Hermes runner"),
                force=True,
            )
        if status == "pending" and task.status in {"blocked", "scheduled"}:
            return kb.unblock_task(connection, task_id)
        if status in {"failed", "cancelled"} and task.status in {"ready", "running"}:
            return kb.block_task(
                connection,
                task_id,
                reason=error or f"Federated runner reported {status}",
                kind="transient" if status == "failed" else "needs_input",
            )
    return False

