"""SQLite persistence for workflow graph definitions and runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from croniter import croniter

from hermes_constants import get_hermes_home
from . import kanban_db as kb
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


_TERMINAL_EXECUTION_STATUSES = {"cancelled", "failed", "succeeded"}
_INIT_DB_LOCK = threading.Lock()
_INITIALIZED_DB_PATHS: set[Path] = set()


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
    kanban_board   TEXT,
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


def _init_cache_key(db_path: Path | None = None) -> Path:
    return _resolve_db_path(db_path).expanduser().resolve(strict=False)


def init_db(db_path: Path | None = None) -> None:
    cache_key = _init_cache_key(db_path)
    with _INIT_DB_LOCK:
        if cache_key in _INITIALIZED_DB_PATHS:
            return
        with contextlib.closing(connect(cache_key)) as conn:
            conn.executescript(SCHEMA_SQL)
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(workflow_node_runs)")
            }
            if "wait_until" not in columns:
                conn.execute("ALTER TABLE workflow_node_runs ADD COLUMN wait_until INTEGER")
            if "kanban_task_id" not in columns:
                conn.execute("ALTER TABLE workflow_node_runs ADD COLUMN kanban_task_id TEXT")
            if "kanban_board" not in columns:
                conn.execute("ALTER TABLE workflow_node_runs ADD COLUMN kanban_board TEXT")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_node_runs_kanban_task
                    ON workflow_node_runs(kanban_task_id)
            """)
        _INITIALIZED_DB_PATHS.add(cache_key)


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


def _json_loads_or_empty(value: str | None) -> Any:
    if value is None:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return {} if decoded is None else decoded


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
    checksum = _checksum(raw)
    now = int(time.time())
    with write_txn(conn):
        existing = conn.execute(
            "SELECT checksum FROM workflow_definitions WHERE workflow_id = ? AND version = ?",
            (spec.id, spec.version),
        ).fetchone()
        if existing is not None and existing["checksum"] != checksum:
            raise ValueError(
                f"workflow definition {spec.id} v{spec.version} already exists with different checksum; bump version"
            )

        inserted = conn.execute(
            """
            INSERT OR IGNORE INTO workflow_definitions (
                workflow_id, version, name, enabled, spec_json, checksum,
                created_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.id,
                spec.version,
                spec.name,
                1 if spec.enabled else 0,
                raw,
                checksum,
                created_by,
                now,
            ),
        ).rowcount > 0
        if not inserted:
            return

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


def get_definition_record(
    conn: sqlite3.Connection,
    workflow_id: str,
    version: int | None = None,
) -> WorkflowDefinitionRecord:
    return _definition_record(conn, workflow_id, version)


def get_definition(
    conn: sqlite3.Connection,
    workflow_id: str,
    version: int | None = None,
) -> WorkflowSpec:
    return get_definition_record(conn, workflow_id, version).spec


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


def list_node_runs(conn: sqlite3.Connection, execution_id: str) -> list[dict[str, Any]]:
    execution = get_execution(conn, execution_id)
    definition = _definition_record(conn, execution.workflow_id, execution.version)
    spec = definition.spec
    incoming = {node_id: 0 for node_id in spec.nodes}
    outgoing = {node_id: [] for node_id in spec.nodes}
    for edge in spec.edges:
        source = edge.from_.split(".", 1)[0]
        if source in outgoing and edge.to in incoming:
            outgoing[source].append(edge.to)
            incoming[edge.to] += 1
    for node_id, node in spec.nodes.items():
        for target in (node.default, node.catch):
            if target in incoming:
                outgoing[node_id].append(target)
                incoming[target] += 1

    pending = [node_id for node_id in spec.nodes if incoming[node_id] == 0]
    ordered_nodes = []
    seen_ordered = set()
    while pending:
        node_id = pending.pop(0)
        if node_id in seen_ordered:
            continue
        seen_ordered.add(node_id)
        ordered_nodes.append(node_id)
        for target in outgoing[node_id]:
            incoming[target] -= 1
            if incoming[target] == 0:
                pending.append(target)
    ordered_nodes.extend(node_id for node_id in spec.nodes if node_id not in seen_ordered)
    node_order = {node_id: index for index, node_id in enumerate(ordered_nodes)}

    rows = conn.execute(
        """
        SELECT * FROM workflow_node_runs
         WHERE execution_id = ?
         ORDER BY id
        """,
        (execution_id,),
    ).fetchall()
    events = conn.execute(
        """
        SELECT id, payload_json, created_at FROM workflow_events
         WHERE execution_id = ? AND kind = 'node_succeeded'
         ORDER BY id
        """,
        (execution_id,),
    ).fetchall()

    events_by_node_id: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        payload = _json_loads_or_empty(event["payload_json"])
        node_id = payload.get("node_id") if isinstance(payload, dict) else None
        if not isinstance(node_id, str):
            continue
        output = payload.get("output", {}) if isinstance(payload, dict) else {}
        events_by_node_id.setdefault(node_id, []).append({
            "event_id": event["id"],
            "created_at": event["created_at"],
            "output": {} if output is None else output,
        })

    result = []
    seen_node_ids = set()
    for row in rows:
        node_id = row["node_id"]
        seen_node_ids.add(node_id)
        raw_output = row["output_json"]
        output = _json_loads_or_empty(raw_output)
        event_infos = events_by_node_id.get(node_id, [])
        event_info = event_infos[-1] if event_infos else None
        if row["status"] == "succeeded" and raw_output in (None, "", "null") and event_info is not None:
            output = event_info["output"]
        result.append(
            {
                "id": row["id"],
                "execution_id": row["execution_id"],
                "node_id": node_id,
                "status": row["status"],
                "input": _json_loads_or_empty(row["input_json"]),
                "output": output,
                "error": _json_loads_or_empty(row["error"]),
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "wait_until": row["wait_until"],
                "kanban_task_id": row["kanban_task_id"],
                "kanban_board": row["kanban_board"],
            }
        )

    for node_id, event_infos in events_by_node_id.items():
        if node_id in seen_node_ids:
            continue
        for event_info in event_infos:
            result.append(
                {
                    "id": None,
                    "event_id": event_info["event_id"],
                    "execution_id": execution_id,
                    "node_id": node_id,
                    "status": "succeeded",
                    "input": {},
                    "output": event_info["output"],
                    "error": {},
                    "started_at": event_info["created_at"],
                    "completed_at": event_info["created_at"],
                    "wait_until": None,
                    "kanban_task_id": None,
                    "kanban_board": None,
                }
            )

    fallback_order = len(node_order)
    result.sort(
        key=lambda run: (
            node_order.get(run["node_id"], fallback_order),
            run["id"] is None,
            run["id"] or run.get("event_id") or 0,
        )
    )
    return result


def _kanban_task_refs(task_refs: list[Any]) -> list[tuple[str, str | None]]:
    refs: list[tuple[str, str | None]] = []
    for ref in task_refs:
        task_id: Any
        board: Any = None
        if isinstance(ref, dict):
            task_id = ref.get("task_id") or ref.get("kanban_task_id")
            board = ref.get("board") or ref.get("kanban_board")
        elif isinstance(ref, (tuple, list)):
            task_id = ref[0] if ref else None
            board = ref[1] if len(ref) > 1 else None
        else:
            task_id = ref
        if task_id:
            refs.append((str(task_id), str(board) if board else None))
    return refs


def block_linked_kanban_tasks(
    task_ids: list[Any],
    *,
    execution_id: str,
    source: str,
    reason: str | None = None,
) -> None:
    refs = _kanban_task_refs(task_ids)
    if not refs:
        return

    reason = reason or f"workflow execution {execution_id} cancelled by {source}"
    for task_id, board in refs:
        with kb.connect_closing(board=board) as kconn:
            task = kb.get_task(kconn, task_id)
            if task is None:
                continue
            if task.status == "running" or task.claim_lock is not None:
                kb.reclaim_task(kconn, task_id, reason=reason)
            kb.block_task(kconn, task_id, reason=reason, kind="capability")


def cancel_execution(
    conn: sqlite3.Connection,
    execution_id: str,
    *,
    source: str = "workflow",
) -> tuple[WorkflowExecution, bool]:
    execution = get_execution(conn, execution_id)
    if execution.status in _TERMINAL_EXECUTION_STATUSES:
        return execution, False

    now = int(time.time())
    terminal_statuses = tuple(sorted(_TERMINAL_EXECUTION_STATUSES))
    placeholders = ", ".join("?" for _ in terminal_statuses)
    linked_task_refs: list[tuple[str, str | None]] = []
    with write_txn(conn):
        linked_task_refs = [
            (row["kanban_task_id"], row["kanban_board"])
            for row in conn.execute(
                """
                SELECT DISTINCT kanban_task_id, kanban_board
                  FROM workflow_node_runs
                 WHERE execution_id = ?
                   AND kanban_task_id IS NOT NULL
                   AND status = 'waiting'
                """,
                (execution_id,),
            ).fetchall()
        ]
        cancelled = conn.execute(
            f"""
            UPDATE workflow_executions
               SET status = 'cancelled', claim_lock = NULL,
                   claim_expires = NULL, updated_at = ?
             WHERE execution_id = ?
               AND status NOT IN ({placeholders})
            """,
            (now, execution_id, *terminal_statuses),
        ).rowcount > 0
        if cancelled:
            append_event(conn, execution_id, "execution_cancelled", {"source": source})

    if cancelled:
        block_linked_kanban_tasks(linked_task_refs, execution_id=execution_id, source=source)

    return get_execution(conn, execution_id), cancelled


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
