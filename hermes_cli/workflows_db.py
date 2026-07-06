"""SQLite persistence for workflow graph definitions and runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from croniter import croniter

from hermes_constants import get_hermes_home
from hermes_cli.workflows_spec import WorkflowSpec


@dataclass(frozen=True)
class WorkflowDefinitionRecord:
    workflow_id: str
    version: int
    name: str
    enabled: bool
    spec: WorkflowSpec
    checksum: str
    created_by: str | None
    created_at: int


@dataclass(frozen=True)
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    version: int
    status: str
    input: dict[str, Any]
    context: dict[str, Any]
    trigger_type: str
    trigger_id: str | None
    created_at: int
    updated_at: int


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workflow_definitions (
    workflow_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    name        TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    spec_json   TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    created_by  TEXT,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (workflow_id, version)
);

CREATE TABLE IF NOT EXISTS workflow_executions (
    execution_id  TEXT PRIMARY KEY,
    workflow_id   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    status        TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    context_json  TEXT NOT NULL,
    trigger_type  TEXT NOT NULL,
    trigger_id    TEXT,
    claim_lock    TEXT,
    claim_expires INTEGER,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    FOREIGN KEY (workflow_id, version)
        REFERENCES workflow_definitions(workflow_id, version)
);

CREATE TABLE IF NOT EXISTS workflow_node_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id  TEXT NOT NULL,
    node_id       TEXT NOT NULL,
    status        TEXT NOT NULL,
    input_json    TEXT,
    output_json   TEXT,
    error         TEXT,
    started_at    INTEGER,
    completed_at  INTEGER,
    wait_until    INTEGER,
    kanban_task_id TEXT,
    FOREIGN KEY (execution_id) REFERENCES workflow_executions(execution_id)
);

CREATE TABLE IF NOT EXISTS workflow_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id  TEXT NOT NULL,
    node_run_id   INTEGER,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    FOREIGN KEY (execution_id) REFERENCES workflow_executions(execution_id),
    FOREIGN KEY (node_run_id) REFERENCES workflow_node_runs(id)
);

CREATE TABLE IF NOT EXISTS workflow_schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    version     INTEGER,
    trigger_id  TEXT,
    schedule    TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    next_run_at INTEGER,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_status
    ON workflow_executions(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_definition
    ON workflow_executions(workflow_id, version);
CREATE INDEX IF NOT EXISTS idx_workflow_events_execution
    ON workflow_events(execution_id, created_at);
CREATE INDEX IF NOT EXISTS idx_workflow_schedules_enabled
    ON workflow_schedules(enabled, next_run_at);
"""


def workflows_db_path() -> Path:
    return get_hermes_home() / "workflows.db"


def _resolve_db_path(db_path: Path | None = None) -> Path:
    return Path(db_path) if db_path is not None else workflows_db_path()


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="workflows.db")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    with contextlib.closing(connect(db_path)) as conn:
        conn.executescript(SCHEMA_SQL)
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(workflow_node_runs)")
        }
        if "wait_until" not in columns:
            conn.execute("ALTER TABLE workflow_node_runs ADD COLUMN wait_until INTEGER")
        if "kanban_task_id" not in columns:
            conn.execute("ALTER TABLE workflow_node_runs ADD COLUMN kanban_task_id TEXT")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_node_runs_kanban_task
                ON workflow_node_runs(kanban_task_id)
        """)


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    else:
        conn.execute("COMMIT")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _spec_json(spec: WorkflowSpec) -> str:
    return _json_dumps(spec.model_dump(mode="json", by_alias=True))


def _checksum(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _schedule_expr(trigger: Any) -> str | None:
    return trigger.cron or trigger.schedule or getattr(trigger, "expr", None)


def _next_cron_run(expr: str, base_ts: int) -> int:
    return int(croniter(expr, base_ts).get_next(float))


def _record_from_row(row: sqlite3.Row) -> WorkflowDefinitionRecord:
    return WorkflowDefinitionRecord(
        workflow_id=row["workflow_id"],
        version=row["version"],
        name=row["name"],
        enabled=bool(row["enabled"]),
        spec=WorkflowSpec.model_validate(json.loads(row["spec_json"])),
        checksum=row["checksum"],
        created_by=row["created_by"],
        created_at=row["created_at"],
    )


def _definition_record(
    conn: sqlite3.Connection,
    workflow_id: str,
    version: int | None = None,
) -> WorkflowDefinitionRecord:
    if version is None:
        row = conn.execute(
            """
            SELECT * FROM workflow_definitions
             WHERE workflow_id = ?
             ORDER BY version DESC
             LIMIT 1
            """,
            (workflow_id,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT * FROM workflow_definitions
             WHERE workflow_id = ? AND version = ?
            """,
            (workflow_id, version),
        ).fetchone()
    if row is None:
        if version is None:
            raise KeyError(f"workflow definition not found: {workflow_id}")
        raise KeyError(f"workflow definition not found: {workflow_id} v{version}")
    return _record_from_row(row)


def deploy_definition(
    conn: sqlite3.Connection,
    spec: WorkflowSpec,
    *,
    created_by: str | None = None,
) -> None:
    raw = _spec_json(spec)
    now = int(time.time())
    with write_txn(conn):
        conn.execute(
            """
            INSERT INTO workflow_definitions (
                workflow_id, version, name, enabled, spec_json, checksum,
                created_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workflow_id, version) DO UPDATE SET
                name = excluded.name,
                enabled = excluded.enabled,
                spec_json = excluded.spec_json,
                checksum = excluded.checksum,
                created_by = excluded.created_by
            """,
            (
                spec.id,
                spec.version,
                spec.name,
                1 if spec.enabled else 0,
                raw,
                _checksum(raw),
                created_by,
                now,
            ),
        )
        conn.execute(
            "DELETE FROM workflow_schedules WHERE workflow_id = ?",
            (spec.id,),
        )
        if spec.enabled:
            for trigger in spec.triggers:
                if trigger.type != "schedule":
                    continue
                expr = _schedule_expr(trigger)
                if not expr:
                    continue
                conn.execute(
                    """
                    INSERT INTO workflow_schedules (
                        workflow_id, version, trigger_id, schedule, enabled,
                        next_run_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        spec.id,
                        spec.version,
                        trigger.id,
                        expr,
                        _next_cron_run(expr, now),
                        now,
                        now,
                    ),
                )


def get_definition(
    conn: sqlite3.Connection,
    workflow_id: str,
    version: int | None = None,
) -> WorkflowSpec:
    return _definition_record(conn, workflow_id, version).spec


def list_definitions(conn: sqlite3.Connection) -> list[WorkflowDefinitionRecord]:
    rows = conn.execute(
        "SELECT * FROM workflow_definitions ORDER BY workflow_id, version"
    ).fetchall()
    return [_record_from_row(row) for row in rows]


def start_execution(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    input_data: dict,
    trigger_type: str,
    trigger_id: str | None = None,
    version: int | None = None,
    now: int | None = None,
) -> str:
    definition = _definition_record(conn, workflow_id, version)
    execution_id = f"wfexec_{secrets.token_hex(8)}"
    created_at = int(time.time()) if now is None else now
    with write_txn(conn):
        conn.execute(
            """
            INSERT INTO workflow_executions (
                execution_id, workflow_id, version, status, input_json,
                context_json, trigger_type, trigger_id, created_at, updated_at
            ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                workflow_id,
                definition.version,
                _json_dumps(input_data),
                _json_dumps({"input": input_data, "node": {}}),
                trigger_type,
                trigger_id,
                created_at,
                created_at,
            ),
        )
    return execution_id


def get_execution(conn: sqlite3.Connection, execution_id: str) -> WorkflowExecution:
    row = conn.execute(
        "SELECT * FROM workflow_executions WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"workflow execution not found: {execution_id}")
    return WorkflowExecution(
        execution_id=row["execution_id"],
        workflow_id=row["workflow_id"],
        version=row["version"],
        status=row["status"],
        input=json.loads(row["input_json"]),
        context=json.loads(row["context_json"]),
        trigger_type=row["trigger_type"],
        trigger_id=row["trigger_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def append_event(
    conn: sqlite3.Connection,
    execution_id: str,
    kind: str,
    payload: dict | None = None,
    node_run_id: int | None = None,
) -> None:
    with write_txn(conn):
        if node_run_id is not None:
            row = conn.execute(
                "SELECT execution_id FROM workflow_node_runs WHERE id = ?",
                (node_run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"workflow node run not found: {node_run_id}")
            if row["execution_id"] != execution_id:
                raise ValueError("node_run_id does not belong to execution")
        conn.execute(
            """
            INSERT INTO workflow_events (
                execution_id, node_run_id, kind, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                node_run_id,
                kind,
                _json_dumps(payload or {}),
                int(time.time()),
            ),
        )
