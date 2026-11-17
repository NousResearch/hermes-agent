"""Read-only provenance projections for native Kanban surfaces."""
from __future__ import annotations

import sqlite3


def task_summary(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT p.table_name, p.disposition, p.source_row_sha256, i.source_snapshot_sha256 "
            "FROM recovery_row_provenance p JOIN recovery_imports i ON i.import_id=p.import_id "
            "WHERE (p.table_name='tasks' AND p.target_pk=?) "
            "OR (p.table_name='task_comments' AND p.target_pk IN (SELECT CAST(id AS TEXT) FROM task_comments WHERE task_id=?)) "
            "OR (p.table_name='task_events' AND p.target_pk IN (SELECT CAST(id AS TEXT) FROM task_events WHERE task_id=?)) "
            "OR (p.table_name='task_runs' AND p.target_pk IN (SELECT CAST(id AS TEXT) FROM task_runs WHERE task_id=?)) "
            "OR (p.table_name='task_attachments' AND p.target_pk IN (SELECT CAST(id AS TEXT) FROM task_attachments WHERE task_id=?)) "
            "ORDER BY p.table_name, p.target_pk", (task_id,) * 5).fetchall()
    except sqlite3.Error:
        return []
    return [dict(row) for row in rows]
