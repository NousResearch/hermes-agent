"""SQLite schema ownership for the hosted Group Chat driver."""

from __future__ import annotations

import sqlite3


_LEASE_COLUMNS = frozenset({
    "room_id",
    "gateway_id",
    "authority_epoch",
    "process_generation",
    "lease_generation",
    "expires_at",
    "acquired_at",
    "updated_at",
    "released_at",
})


_TASK_COLUMNS = frozenset({
    "room_id",
    "task_id",
    "thread_id",
    "turn_id",
    "source_event_seq",
    "payload_json",
    "payload_digest",
    "status",
    "execution_generation",
    "cancel_generation",
    "run_gateway_id",
    "run_process_generation",
    "run_lease_generation",
    "cancel_id",
    "settlement_id",
    "settlement_status",
    "result_json",
    "created_at",
    "updated_at",
    "started_at",
    "terminal_at",
    "indeterminate_at",
})


_TASK_COLUMN_ORDER = (
    "room_id",
    "task_id",
    "thread_id",
    "turn_id",
    "source_event_seq",
    "payload_json",
    "payload_digest",
    "status",
    "execution_generation",
    "cancel_generation",
    "run_gateway_id",
    "run_process_generation",
    "run_lease_generation",
    "cancel_id",
    "settlement_id",
    "settlement_status",
    "result_json",
    "created_at",
    "updated_at",
    "started_at",
    "terminal_at",
    "indeterminate_at",
)


class DriverStateError(ValueError):
    """Base class for invalid or conflicting driver-state operations."""


def _create_task_table(
    conn: sqlite3.Connection, table: str = "hosted_room_driver_tasks"
) -> None:
    if table not in {"hosted_room_driver_tasks", "hosted_room_driver_tasks_next"}:
        raise DriverStateError("invalid hosted-room task table name")
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {table} (
            room_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            source_event_seq INTEGER NOT NULL CHECK (source_event_seq >= 1),
            payload_json TEXT NOT NULL,
            payload_digest TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN (
                    'queued', 'running', 'settled', 'failed',
                    'cancelled', 'indeterminate', 'deferred', 'stopping'
                )
            ),
            execution_generation INTEGER NOT NULL DEFAULT 0
                CHECK (execution_generation >= 0),
            cancel_generation INTEGER NOT NULL DEFAULT 0
                CHECK (cancel_generation >= 0),
            run_gateway_id TEXT,
            run_process_generation TEXT,
            run_lease_generation INTEGER,
            cancel_id TEXT,
            settlement_id TEXT,
            settlement_status TEXT,
            result_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            started_at REAL,
            terminal_at REAL,
            indeterminate_at REAL,
            PRIMARY KEY (room_id, task_id),
            UNIQUE (room_id, thread_id, turn_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
        )"""
    )


def _initialize_retry_receipt_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_retry_receipts (
            retry_id TEXT PRIMARY KEY,
            room_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            source_execution_generation INTEGER NOT NULL
                CHECK (source_execution_generation >= 0),
            created_at REAL NOT NULL,
            FOREIGN KEY (room_id, task_id)
                REFERENCES hosted_room_driver_tasks(room_id, task_id)
                ON DELETE CASCADE
        )"""
    )


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_driver_leases (
            room_id TEXT PRIMARY KEY,
            gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            process_generation TEXT NOT NULL,
            lease_generation INTEGER NOT NULL CHECK (lease_generation >= 1),
            expires_at REAL NOT NULL,
            acquired_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            released_at REAL,
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
        )"""
    )
    _create_task_table(conn)
    _initialize_retry_receipt_table(conn)
    _validate_schema(conn)
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_driver_tasks_status
           ON hosted_room_driver_tasks(
               room_id, status, source_event_seq, created_at, task_id
           )"""
    )


def _validate_schema(conn: sqlite3.Connection) -> None:
    lease_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_driver_leases)")
    )
    task_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_driver_tasks)")
    )
    if lease_columns != _LEASE_COLUMNS or task_columns != _TASK_COLUMNS:
        raise DriverStateError(
            "unsupported unpublished hosted-room driver schema; "
            "recreate the driver tables before starting the driver"
        )

    for table in ("hosted_room_driver_leases", "hosted_room_driver_tasks"):
        foreign_keys = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        if not any(
            row[2] == "hosted_rooms" and row[3] == "room_id" and row[4] == "room_id"
            for row in foreign_keys
        ):
            raise DriverStateError(f"{table} is missing its hosted_rooms foreign key")


def _schema_objects_exist(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        """SELECT name FROM sqlite_master
           WHERE type='table' AND name IN (
               'hosted_room_driver_leases', 'hosted_room_driver_tasks'
           )"""
    ).fetchall()
    tables = {row[0] for row in rows}
    if tables != {"hosted_room_driver_leases", "hosted_room_driver_tasks"}:
        return False
    index = conn.execute(
        """SELECT 1 FROM sqlite_master
           WHERE type='index' AND name='idx_hosted_room_driver_tasks_status'"""
    ).fetchone()
    return index is not None


def _task_schema_supports_current_statuses(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        """SELECT sql FROM sqlite_master
           WHERE type='table' AND name='hosted_room_driver_tasks'"""
    ).fetchone()
    sql = str(row[0] or "").lower() if row else ""
    return "'stopping'" in sql and "'deferred'" in sql


def _task_schema_has_legacy_retry_id(conn: sqlite3.Connection) -> bool:
    return any(
        row[1] == "retry_id"
        for row in conn.execute("PRAGMA table_info(hosted_room_driver_tasks)")
    )


def _migrate_task_status_constraint(conn: sqlite3.Connection) -> None:
    """Expand the unpublished task-state CHECK without losing durable work."""
    preserved_receipts = []
    receipt_table = conn.execute(
        """SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='hosted_room_retry_receipts'"""
    ).fetchone()
    if receipt_table is not None:
        preserved_receipts.extend(
            conn.execute(
                """SELECT retry_id, room_id, task_id,
                          source_execution_generation, created_at
                     FROM hosted_room_retry_receipts"""
            ).fetchall()
        )
    if _task_schema_has_legacy_retry_id(conn):
        preserved_receipts.extend(
            conn.execute(
                """SELECT retry_id, room_id, task_id, execution_generation, updated_at
                 FROM hosted_room_driver_tasks
                WHERE retry_id IS NOT NULL AND retry_id != ''"""
            ).fetchall()
        )
    conn.execute("DROP TABLE IF EXISTS hosted_room_retry_receipts")
    conn.execute("DROP INDEX IF EXISTS idx_hosted_room_driver_tasks_status")
    _create_task_table(conn, "hosted_room_driver_tasks_next")
    columns = ", ".join(_TASK_COLUMN_ORDER)
    conn.execute(
        f"""INSERT INTO hosted_room_driver_tasks_next ({columns})
             SELECT {columns} FROM hosted_room_driver_tasks"""
    )
    conn.execute("DROP TABLE hosted_room_driver_tasks")
    conn.execute(
        "ALTER TABLE hosted_room_driver_tasks_next RENAME TO hosted_room_driver_tasks"
    )
    conn.execute(
        """CREATE INDEX idx_hosted_room_driver_tasks_status
           ON hosted_room_driver_tasks(
               room_id, status, source_event_seq, created_at, task_id
           )"""
    )
    _initialize_retry_receipt_table(conn)
    for receipt in preserved_receipts:
        existing = conn.execute(
            "SELECT room_id, task_id FROM hosted_room_retry_receipts WHERE retry_id=?",
            (str(receipt["retry_id"]),),
        ).fetchone()
        if existing is not None and (
            str(existing["room_id"]),
            str(existing["task_id"]),
        ) != (str(receipt["room_id"]), str(receipt["task_id"])):
            raise DriverStateError("draft retry_id is bound to multiple tasks")
        conn.execute(
            """INSERT OR IGNORE INTO hosted_room_retry_receipts(
                   retry_id, room_id, task_id, source_execution_generation, created_at
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                str(receipt["retry_id"]),
                str(receipt["room_id"]),
                str(receipt["task_id"]),
                int(
                    receipt["source_execution_generation"]
                    if "source_execution_generation" in receipt.keys()
                    else receipt["execution_generation"]
                ),
                float(
                    receipt["created_at"]
                    if "created_at" in receipt.keys()
                    else receipt["updated_at"]
                ),
            ),
        )
