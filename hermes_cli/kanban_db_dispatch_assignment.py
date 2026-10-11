"""Default-assignee persistence for dispatcher admission."""
import sqlite3


def apply_default_assignee(conn: sqlite3.Connection, task_id: str, assignee: str, *, dry_run: bool) -> bool:
    """Persist the default assignee, keeping dry runs read-only."""
    from hermes_cli import kanban_db as kb
    if dry_run:
        return True
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET assignee = ? WHERE id = ? "
                "AND (assignee IS NULL OR assignee = '')",
                (assignee, task_id),
            )
            kb._append_event(
                conn, task_id, "assigned",
                {"assignee": assignee, "source": "kanban.default_assignee"},
            )
    except (sqlite3.Error, ValueError, TypeError):
        kb._log.debug(
            "kanban dispatch: failed to apply default_assignee=%r to task %s",
            assignee, task_id, exc_info=True,
        )
        return False
    return True
