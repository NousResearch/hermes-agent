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
from hermes_cli.workflows_expr import resolve_path
from hermes_cli.workflows_intake import evaluate_intake
from hermes_cli.workflows_spec import WorkflowSpec


@dataclass(frozen=True)
class WorkflowDefinitionRecord:
    workflow_id: str
    version: int
    name: str
    enabled: bool
    archived: bool
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


@dataclass(frozen=True)
class WorkflowInputFeed:
    feed_id: str
    workflow_id: str
    version: int
    trigger_id: str | None
    status: str
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class WorkflowInputItem:
    item_id: str
    feed_id: str
    workflow_id: str
    version: int
    trigger_id: str | None
    status: str
    input: dict[str, Any]
    criteria: dict[str, Any]
    dedupe_value: str | None
    execution_id: str | None
    created_at: int
    updated_at: int


@dataclass(frozen=True)
class WorkflowDraftRecord:
    workflow_id: str
    spec: WorkflowSpec
    base_version: int | None
    updated_at: int


class WorkflowVersionConflict(Exception):
    """Raised when a draft's expected_latest_version no longer matches the DB."""


class WorkflowHistoryExists(Exception):
    """Raised when a destructive delete would orphan execution/feed history."""


_TERMINAL_EXECUTION_STATUSES = {"blocked", "cancelled", "failed", "succeeded"}
_MUTABLE_INPUT_ITEM_STATUSES = {"needs_input", "queued"}
_FEED_STATUSES = {"open", "paused", "closed"}
_FEED_TRANSITIONS: dict[str, set[str]] = {
    "open": {"paused", "closed"},
    "paused": {"open", "closed"},
    "closed": set(),
}
_INIT_DB_LOCK = threading.Lock()
_INITIALIZED_DB_PATHS: set[Path] = set()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workflow_definitions (
    workflow_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    name        TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    archived    INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS workflow_input_feeds (
    feed_id     TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    trigger_id  TEXT,
    status      TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_input_items (
    item_id       TEXT PRIMARY KEY,
    feed_id       TEXT NOT NULL,
    workflow_id   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    trigger_id    TEXT,
    status        TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    criteria_json TEXT NOT NULL,
    dedupe_value  TEXT,
    execution_id  TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_status
    ON workflow_executions(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_definition
    ON workflow_executions(workflow_id, version);
CREATE INDEX IF NOT EXISTS idx_workflow_events_execution
    ON workflow_events(execution_id, created_at);
CREATE INDEX IF NOT EXISTS idx_workflow_schedules_enabled
    ON workflow_schedules(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS idx_workflow_input_feeds_status
    ON workflow_input_feeds(status, workflow_id, version);
CREATE INDEX IF NOT EXISTS idx_workflow_input_items_status
    ON workflow_input_items(status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_input_items_dedupe
    ON workflow_input_items(feed_id, dedupe_value)
    WHERE dedupe_value IS NOT NULL;

CREATE TABLE IF NOT EXISTS workflow_drafts (
    workflow_id  TEXT PRIMARY KEY,
    spec_json    TEXT NOT NULL,
    base_version INTEGER,
    updated_at   INTEGER NOT NULL
);
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
            def_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(workflow_definitions)")
            }
            if "archived" not in def_columns:
                conn.execute(
                    "ALTER TABLE workflow_definitions "
                    "ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
                )
            dedupe_index = next(
                (
                    row
                    for row in conn.execute("PRAGMA index_list(workflow_input_items)")
                    if row["name"] == "idx_workflow_input_items_dedupe"
                ),
                None,
            )
            if dedupe_index is not None and not bool(dedupe_index["unique"]):
                conn.execute("DROP INDEX idx_workflow_input_items_dedupe")
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_input_items_dedupe
                    ON workflow_input_items(feed_id, dedupe_value)
                    WHERE dedupe_value IS NOT NULL
            """)
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
    archived_value = row["archived"] if "archived" in row.keys() else 0
    return WorkflowDefinitionRecord(
        workflow_id=row["workflow_id"],
        version=row["version"],
        name=row["name"],
        enabled=bool(row["enabled"]),
        archived=bool(archived_value),
        spec=WorkflowSpec.model_validate(json.loads(row["spec_json"])),
        checksum=row["checksum"],
        created_by=row["created_by"],
        created_at=row["created_at"],
    )


def _feed_from_row(row: sqlite3.Row) -> WorkflowInputFeed:
    return WorkflowInputFeed(
        feed_id=row["feed_id"],
        workflow_id=row["workflow_id"],
        version=row["version"],
        trigger_id=row["trigger_id"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _item_from_row(row: sqlite3.Row) -> WorkflowInputItem:
    return WorkflowInputItem(
        item_id=row["item_id"],
        feed_id=row["feed_id"],
        workflow_id=row["workflow_id"],
        version=row["version"],
        trigger_id=row["trigger_id"],
        status=row["status"],
        input=_json_loads_or_empty(row["input_json"]),
        criteria=_json_loads_or_empty(row["criteria_json"]),
        dedupe_value=row["dedupe_value"],
        execution_id=row["execution_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
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


def _register_schedules(
    conn: sqlite3.Connection,
    spec: WorkflowSpec,
    *,
    now: int,
) -> None:
    """Replace this workflow's schedule rows with the given spec's triggers."""
    conn.execute(
        "DELETE FROM workflow_schedules WHERE workflow_id = ?",
        (spec.id,),
    )
    if not spec.enabled:
        return
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


def deploy_definition(
    conn: sqlite3.Connection,
    spec: WorkflowSpec,
    *,
    created_by: str | None = None,
    auto_bump: bool = False,
) -> int:
    """Deploy a definition; return the version actually deployed.

    Same version + same checksum is an idempotent no-op. Same version +
    different checksum raises unless ``auto_bump`` is set, in which case the
    spec is redeployed as ``max(existing versions) + 1``. Every surface (CLI,
    model tools, dashboard) shares this one explicit contract.
    """
    raw = _spec_json(spec)
    checksum = _checksum(raw)
    now = int(time.time())
    with write_txn(conn):
        existing = conn.execute(
            "SELECT checksum FROM workflow_definitions WHERE workflow_id = ? AND version = ?",
            (spec.id, spec.version),
        ).fetchone()
        if existing is not None and existing["checksum"] != checksum:
            if not auto_bump:
                raise ValueError(
                    f"workflow definition {spec.id} v{spec.version} already exists with different checksum; bump version"
                )
            max_version = conn.execute(
                "SELECT MAX(version) FROM workflow_definitions WHERE workflow_id = ?",
                (spec.id,),
            ).fetchone()[0]
            spec = spec.model_copy(update={"version": int(max_version or spec.version) + 1})
            raw = _spec_json(spec)
            checksum = _checksum(raw)

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
            return spec.version

        _register_schedules(conn, spec, now=now)
    return spec.version


def set_definition_enabled(
    conn: sqlite3.Connection,
    workflow_id: str,
    enabled: bool,
    *,
    version: int | None = None,
) -> WorkflowDefinitionRecord:
    """Enable or disable a deployed definition (latest version by default).

    Disabling removes the workflow's schedule rows and blocks new manual
    runs; enabling re-registers schedules from the stored spec.
    """
    record = _definition_record(conn, workflow_id, version)
    now = int(time.time())
    with write_txn(conn):
        conn.execute(
            "UPDATE workflow_definitions SET enabled = ? WHERE workflow_id = ? AND version = ?",
            (1 if enabled else 0, workflow_id, record.version),
        )
        spec = record.spec.model_copy(update={"enabled": enabled})
        _register_schedules(conn, spec, now=now)
    return _definition_record(conn, workflow_id, record.version)


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


def save_draft(
    conn: sqlite3.Connection,
    spec: WorkflowSpec,
    *,
    base_version: int | None,
    updated_at: int | None = None,
) -> WorkflowDraftRecord:
    ts = int(time.time()) if updated_at is None else updated_at
    with write_txn(conn):
        conn.execute(
            """
            INSERT INTO workflow_drafts (workflow_id, spec_json, base_version, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(workflow_id) DO UPDATE SET
                spec_json = excluded.spec_json,
                base_version = excluded.base_version,
                updated_at = excluded.updated_at
            """,
            (spec.id, _spec_json(spec), base_version, ts),
        )
    return WorkflowDraftRecord(
        workflow_id=spec.id,
        spec=spec,
        base_version=base_version,
        updated_at=ts,
    )


def get_draft(conn: sqlite3.Connection, workflow_id: str) -> WorkflowDraftRecord | None:
    row = conn.execute(
        "SELECT * FROM workflow_drafts WHERE workflow_id = ?",
        (workflow_id,),
    ).fetchone()
    if row is None:
        return None
    return WorkflowDraftRecord(
        workflow_id=row["workflow_id"],
        spec=WorkflowSpec.model_validate(json.loads(row["spec_json"])),
        base_version=row["base_version"],
        updated_at=row["updated_at"],
    )


def delete_draft(conn: sqlite3.Connection, workflow_id: str) -> bool:
    with write_txn(conn):
        deleted = conn.execute(
            "DELETE FROM workflow_drafts WHERE workflow_id = ?",
            (workflow_id,),
        ).rowcount
    return deleted > 0


def _latest_definition_version(conn: sqlite3.Connection, workflow_id: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(version) FROM workflow_definitions WHERE workflow_id = ?",
        (workflow_id,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def publish_draft(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    expected_latest_version: int | None,
    created_by: str | None,
) -> WorkflowDefinitionRecord:
    draft = get_draft(conn, workflow_id)
    if draft is None:
        raise KeyError(f"workflow draft not found: {workflow_id}")
    latest = _latest_definition_version(conn, workflow_id)
    if latest != expected_latest_version:
        raise WorkflowVersionConflict(
            f"workflow {workflow_id} expected latest version "
            f"{expected_latest_version!r}, found {latest!r}"
        )
    next_version = (latest or 0) + 1
    publish_spec = draft.spec.model_copy(update={"version": next_version})
    try:
        with write_txn(conn):
            deployed_version = deploy_definition(
                conn, publish_spec, created_by=created_by, auto_bump=False
            )
            conn.execute(
                "DELETE FROM workflow_drafts WHERE workflow_id = ?",
                (workflow_id,),
            )
    except WorkflowVersionConflict:
        raise
    except Exception:
        raise
    return _definition_record(conn, workflow_id, deployed_version)


def set_workflow_archived(
    conn: sqlite3.Connection,
    workflow_id: str,
    archived: bool,
) -> None:
    with write_txn(conn):
        updated = conn.execute(
            "UPDATE workflow_definitions SET archived = ? WHERE workflow_id = ?",
            (1 if archived else 0, workflow_id),
        ).rowcount
    if not updated:
        raise KeyError(f"workflow definition not found: {workflow_id}")


def list_workflow_summaries(
    conn: sqlite3.Connection,
    *,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    # Single grouped read pass: latest version + enabled + archived + last
    # execution status + open feed count, joined to current draft rows.
    sql = """
        WITH latest AS (
            SELECT workflow_id, MAX(version) AS latest_version
              FROM workflow_definitions
             GROUP BY workflow_id
        ),
        latest_row AS (
            SELECT d.workflow_id, l.latest_version, d.enabled, d.archived, d.name
              FROM workflow_definitions d
              JOIN latest l ON l.workflow_id = d.workflow_id AND l.latest_version = d.version
        ),
        exec_status AS (
            SELECT workflow_id, status FROM (
                SELECT workflow_id, status,
                       ROW_NUMBER() OVER (PARTITION BY workflow_id ORDER BY updated_at DESC, execution_id DESC) AS rn
                  FROM workflow_executions
            ) WHERE rn = 1
        ),
        feed_counts AS (
            SELECT workflow_id, SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_count
              FROM workflow_input_feeds
             GROUP BY workflow_id
        )
        SELECT lr.workflow_id,
               lr.latest_version,
               lr.enabled,
               lr.archived,
               lr.name,
               es.status AS latest_execution_status,
               COALESCE(fc.open_count, 0) AS open_feed_count,
               CASE WHEN d.workflow_id IS NULL THEN 0 ELSE 1 END AS has_draft
          FROM latest_row lr
          LEFT JOIN exec_status es ON es.workflow_id = lr.workflow_id
          LEFT JOIN feed_counts fc ON fc.workflow_id = lr.workflow_id
          LEFT JOIN workflow_drafts d ON d.workflow_id = lr.workflow_id
        ORDER BY lr.workflow_id
    """
    rows = conn.execute(sql).fetchall()
    summaries: list[dict[str, Any]] = []
    for row in rows:
        if not include_archived and bool(row["archived"]):
            continue
        summaries.append({
            "workflow_id": row["workflow_id"],
            "name": row["name"],
            "latest_version": int(row["latest_version"]),
            "enabled": bool(row["enabled"]),
            "archived": bool(row["archived"]),
            "latest_execution_status": row["latest_execution_status"],
            "open_feed_count": int(row["open_feed_count"] or 0),
            "has_draft": bool(row["has_draft"]),
        })
    return summaries


def _has_workflow_history(conn: sqlite3.Connection, workflow_id: str) -> bool:
    for table in ("workflow_executions", "workflow_input_feeds"):
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE workflow_id = ? LIMIT 1",
            (workflow_id,),
        ).fetchone()
        if row is not None:
            return True
    return False


def delete_definition(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    purge: bool = False,
) -> bool:
    if purge:
        return _purge_definition(conn, workflow_id)
    with write_txn(conn):
        has_def = conn.execute(
            "SELECT 1 FROM workflow_definitions WHERE workflow_id = ? LIMIT 1",
            (workflow_id,),
        ).fetchone()
        has_draft = conn.execute(
            "SELECT 1 FROM workflow_drafts WHERE workflow_id = ? LIMIT 1",
            (workflow_id,),
        ).fetchone()
        if has_def is None and has_draft is None:
            return False
        if has_def is not None and _has_workflow_history(conn, workflow_id):
            raise WorkflowHistoryExists(
                f"workflow {workflow_id} has execution or feed history; "
                "purge=true required to delete"
            )
        conn.execute("DELETE FROM workflow_drafts WHERE workflow_id = ?", (workflow_id,))
        if has_def is None:
            return True
        return _purge_definition_rows(conn, workflow_id)


def _purge_definition(conn: sqlite3.Connection, workflow_id: str) -> bool:
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM workflow_definitions WHERE workflow_id = ? LIMIT 1",
            (workflow_id,),
        ).fetchone():
            return False
        conn.execute("DELETE FROM workflow_drafts WHERE workflow_id = ?", (workflow_id,))
        _purge_definition_rows(conn, workflow_id)
    return True


def _purge_definition_rows(conn: sqlite3.Connection, workflow_id: str) -> bool:
    conn.execute(
        """
        DELETE FROM workflow_events
         WHERE execution_id IN (
            SELECT execution_id FROM workflow_executions WHERE workflow_id = ?
         )
        """,
        (workflow_id,),
    )
    conn.execute(
        """
        DELETE FROM workflow_node_runs
         WHERE execution_id IN (
            SELECT execution_id FROM workflow_executions WHERE workflow_id = ?
         )
        """,
        (workflow_id,),
    )
    conn.execute("DELETE FROM workflow_executions WHERE workflow_id = ?", (workflow_id,))
    conn.execute("DELETE FROM workflow_schedules WHERE workflow_id = ?", (workflow_id,))
    conn.execute("DELETE FROM workflow_input_items WHERE workflow_id = ?", (workflow_id,))
    conn.execute("DELETE FROM workflow_input_feeds WHERE workflow_id = ?", (workflow_id,))
    conn.execute("DELETE FROM workflow_definitions WHERE workflow_id = ?", (workflow_id,))
    return True


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
    if not definition.enabled:
        raise ValueError(
            f"workflow {workflow_id} v{definition.version} is disabled; "
            "enable it with `hermes workflow enable` or redeploy with enabled: true"
        )
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


def _continuous_feed_trigger(spec: WorkflowSpec, trigger_id: str | None):
    candidates = [
        trigger
        for trigger in spec.triggers
        if trigger.type == "manual" and trigger.intake.mode == "continuous"
    ]
    if trigger_id is None:
        if candidates:
            return candidates[0]
        raise ValueError("workflow has no continuous manual input trigger")
    for trigger in spec.triggers:
        if trigger.id == trigger_id:
            if trigger in candidates:
                return trigger
            raise ValueError(f"workflow trigger is not a continuous manual input trigger: {trigger_id}")
    raise KeyError(f"workflow trigger not found: {trigger_id}")


def _trigger_for_feed(conn: sqlite3.Connection, feed: WorkflowInputFeed):
    spec = get_definition(conn, feed.workflow_id, feed.version)
    return _continuous_feed_trigger(spec, feed.trigger_id)


def _dedupe_value(trigger: Any, input_data: dict[str, Any]) -> str | None:
    if not trigger.intake.dedupe_key:
        return None
    try:
        value = resolve_path({"input": input_data}, trigger.intake.dedupe_key, default=None)
    except ValueError:
        return None
    if value is None:
        return None
    raw = value if isinstance(value, str) else _json_dumps(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _materialize_input(trigger: Any, input_data: dict[str, Any]) -> dict[str, Any]:
    materialized = dict(trigger.input)
    for name, field in trigger.input_schema.items():
        if name not in materialized and field.default is not None:
            materialized[name] = field.default
    materialized.update(input_data)
    return materialized


def _dedupe_item_row(
    conn: sqlite3.Connection,
    feed_id: str,
    dedupe: str | None,
    *,
    exclude_item_id: str | None = None,
) -> sqlite3.Row | None:
    if dedupe is None:
        return None
    if exclude_item_id is None:
        return conn.execute(
            "SELECT * FROM workflow_input_items WHERE feed_id = ? AND dedupe_value = ? ORDER BY created_at LIMIT 1",
            (feed_id, dedupe),
        ).fetchone()
    return conn.execute(
        """
        SELECT * FROM workflow_input_items
         WHERE feed_id = ? AND dedupe_value = ? AND item_id != ?
         ORDER BY created_at LIMIT 1
        """,
        (feed_id, dedupe, exclude_item_id),
    ).fetchone()


def _criteria_for(trigger: Any, input_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    evaluation = evaluate_intake(trigger, input_data)
    criteria = dict(evaluation.criteria)
    criteria["messages"] = evaluation.messages
    return evaluation.status, criteria


def _manual_trigger(spec: WorkflowSpec, trigger_id: str | None):
    if trigger_id is not None:
        for trigger in spec.triggers:
            if trigger.id == trigger_id:
                if trigger.type != "manual":
                    raise ValueError(f"workflow trigger is not manual: {trigger_id}")
                return trigger
        raise KeyError(f"workflow trigger not found: {trigger_id}")
    for trigger in spec.triggers:
        if trigger.type == "manual":
            return trigger
    return None


def start_manual_execution(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    input_data: dict[str, Any],
    trigger_id: str | None = None,
    version: int | None = None,
    now: int | None = None,
) -> str:
    definition = _definition_record(conn, workflow_id, version)
    trigger = _manual_trigger(definition.spec, trigger_id)
    materialized = dict(input_data)
    resolved_trigger_id = trigger_id
    if trigger is not None:
        materialized = _materialize_input(trigger, input_data)
        evaluation = evaluate_intake(trigger, materialized)
        if not evaluation.ready:
            raise ValueError("; ".join(evaluation.messages) or "workflow input is not ready")
        resolved_trigger_id = trigger.id
    return start_execution(
        conn,
        workflow_id,
        input_data=materialized,
        trigger_type="manual",
        trigger_id=resolved_trigger_id,
        version=version,
        now=now,
    )


def open_input_feed(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    trigger_id: str | None = None,
    version: int | None = None,
    now: int | None = None,
) -> WorkflowInputFeed:
    definition = _definition_record(conn, workflow_id, version)
    trigger = _continuous_feed_trigger(definition.spec, trigger_id)
    feed_id = f"wffeed_{secrets.token_hex(8)}"
    ts = int(time.time()) if now is None else now
    with write_txn(conn):
        conn.execute(
            """
            INSERT INTO workflow_input_feeds (
                feed_id, workflow_id, version, trigger_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'open', ?, ?)
            """,
            (feed_id, workflow_id, definition.version, trigger.id, ts, ts),
        )
    return get_input_feed(conn, feed_id)


def get_input_feed(conn: sqlite3.Connection, feed_id: str) -> WorkflowInputFeed:
    row = conn.execute("SELECT * FROM workflow_input_feeds WHERE feed_id = ?", (feed_id,)).fetchone()
    if row is None:
        raise KeyError(f"workflow input feed not found: {feed_id}")
    return _feed_from_row(row)


def list_input_feeds(conn: sqlite3.Connection, *, status: str | None = None) -> list[WorkflowInputFeed]:
    if status is None:
        rows = conn.execute("SELECT * FROM workflow_input_feeds ORDER BY created_at, feed_id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM workflow_input_feeds WHERE status = ? ORDER BY created_at, feed_id",
            (status,),
        ).fetchall()
    return [_feed_from_row(row) for row in rows]


def set_input_feed_status(conn: sqlite3.Connection, feed_id: str, status: str) -> WorkflowInputFeed:
    if status not in _FEED_STATUSES:
        raise ValueError(f"invalid feed status: {status}")
    now = int(time.time())
    feed = get_input_feed(conn, feed_id)
    if status == feed.status:
        return feed
    if status not in _FEED_TRANSITIONS[feed.status]:
        raise ValueError(
            f"workflow input feed {feed_id} cannot transition from "
            f"{feed.status} to {status}"
            + ("; closed feed cannot transition" if feed.status == "closed" else "")
        )
    with write_txn(conn):
        conn.execute(
            "UPDATE workflow_input_feeds SET status = ?, updated_at = ? WHERE feed_id = ?",
            (status, now, feed_id),
        )
    return get_input_feed(conn, feed_id)


def _require_open_feed(feed: WorkflowInputFeed) -> None:
    if feed.status != "open":
        raise ValueError(
            f"workflow input feed is {feed.status}: {feed.feed_id}"
        )


def get_input_item(conn: sqlite3.Connection, item_id: str) -> WorkflowInputItem:
    row = conn.execute("SELECT * FROM workflow_input_items WHERE item_id = ?", (item_id,)).fetchone()
    if row is None:
        raise KeyError(f"workflow input item not found: {item_id}")
    return _item_from_row(row)


def list_input_items(
    conn: sqlite3.Connection,
    *,
    feed_id: str | None = None,
    status: str | None = None,
) -> list[WorkflowInputItem]:
    clauses = []
    params: list[Any] = []
    if feed_id is not None:
        clauses.append("feed_id = ?")
        params.append(feed_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM workflow_input_items{where} ORDER BY created_at, item_id",
        params,
    ).fetchall()
    return [_item_from_row(row) for row in rows]


def enqueue_input_item(
    conn: sqlite3.Connection,
    feed_id: str,
    input_data: dict[str, Any],
    *,
    now: int | None = None,
) -> WorkflowInputItem:
    feed = get_input_feed(conn, feed_id)
    _require_open_feed(feed)
    trigger = _trigger_for_feed(conn, feed)
    materialized = _materialize_input(trigger, input_data)
    status, criteria = _criteria_for(trigger, materialized)
    dedupe = _dedupe_value(trigger, materialized)
    row = _dedupe_item_row(conn, feed_id, dedupe)
    if row is not None:
        return _item_from_row(row)
    item_id = f"wfitem_{secrets.token_hex(8)}"
    ts = int(time.time()) if now is None else now
    try:
        with write_txn(conn):
            conn.execute(
                """
                INSERT INTO workflow_input_items (
                    item_id, feed_id, workflow_id, version, trigger_id, status,
                    input_json, criteria_json, dedupe_value, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    feed_id,
                    feed.workflow_id,
                    feed.version,
                    feed.trigger_id,
                    status,
                    _json_dumps(materialized),
                    _json_dumps(criteria),
                    dedupe,
                    ts,
                    ts,
                ),
            )
    except sqlite3.IntegrityError:
        row = _dedupe_item_row(conn, feed_id, dedupe)
        if row is not None:
            return _item_from_row(row)
        raise
    return get_input_item(conn, item_id)


def update_input_item(
    conn: sqlite3.Connection,
    item_id: str,
    input_data: dict[str, Any],
    *,
    now: int | None = None,
) -> WorkflowInputItem:
    item = get_input_item(conn, item_id)
    feed = get_input_feed(conn, item.feed_id)
    _require_open_feed(feed)
    if item.status not in _MUTABLE_INPUT_ITEM_STATUSES:
        raise ValueError(f"workflow input item is not mutable: {item_id}")
    trigger = _trigger_for_feed(conn, feed)
    materialized = _materialize_input(trigger, input_data)
    status, criteria = _criteria_for(trigger, materialized)
    dedupe = _dedupe_value(trigger, materialized)
    row = _dedupe_item_row(conn, item.feed_id, dedupe, exclude_item_id=item_id)
    if row is not None:
        raise ValueError(f"workflow input item dedupe conflict: {item_id} conflicts with {row['item_id']}")
    ts = int(time.time()) if now is None else now
    try:
        with write_txn(conn):
            conn.execute(
                """
                UPDATE workflow_input_items
                   SET input_json = ?, criteria_json = ?, dedupe_value = ?, status = ?, updated_at = ?
                 WHERE item_id = ?
                """,
                (_json_dumps(materialized), _json_dumps(criteria), dedupe, status, ts, item_id),
            )
    except sqlite3.IntegrityError:
        row = _dedupe_item_row(conn, item.feed_id, dedupe, exclude_item_id=item_id)
        if row is not None:
            raise ValueError(f"workflow input item dedupe conflict: {item_id} conflicts with {row['item_id']}") from None
        raise
    return get_input_item(conn, item_id)


def claim_next_ready_input_item(conn: sqlite3.Connection) -> WorkflowInputItem | None:
    if not conn.in_transaction:
        raise RuntimeError("claim_next_ready_input_item must be called inside write_txn")
    row = conn.execute(
        """
        SELECT item.* FROM workflow_input_items item
          JOIN workflow_input_feeds feed ON feed.feed_id = item.feed_id
         WHERE feed.status = 'open' AND item.status = 'queued'
         ORDER BY item.created_at, item.item_id
         LIMIT 1
        """
    ).fetchone()
    return _item_from_row(row) if row is not None else None


def mark_input_item_running(
    conn: sqlite3.Connection,
    item_id: str,
    execution_id: str,
    *,
    now: int | None = None,
) -> WorkflowInputItem:
    ts = int(time.time()) if now is None else now
    with write_txn(conn):
        updated = conn.execute(
            """
            UPDATE workflow_input_items
               SET status = 'running', execution_id = ?, updated_at = ?
             WHERE item_id = ? AND status = 'queued'
            """,
            (execution_id, ts, item_id),
        ).rowcount
    if not updated:
        raise ValueError(f"workflow input item is not queued: {item_id}")
    return get_input_item(conn, item_id)


def mark_input_item_terminal(
    conn: sqlite3.Connection,
    item_id: str,
    status: str,
    *,
    now: int | None = None,
) -> WorkflowInputItem:
    if status not in _TERMINAL_EXECUTION_STATUSES:
        raise ValueError(f"invalid terminal status: {status}")
    ts = int(time.time()) if now is None else now
    with write_txn(conn):
        updated = conn.execute(
            """
            UPDATE workflow_input_items
               SET status = ?, updated_at = ?
             WHERE item_id = ?
            """,
            (status, ts, item_id),
        ).rowcount
    if not updated:
        raise KeyError(f"workflow input item not found: {item_id}")
    return get_input_item(conn, item_id)


def sync_terminal_input_items(conn: sqlite3.Connection, *, now: int | None = None) -> int:
    ts = int(time.time()) if now is None else now
    terminal_statuses = tuple(sorted(_TERMINAL_EXECUTION_STATUSES))
    placeholders = ", ".join("?" for _ in terminal_statuses)
    with write_txn(conn):
        updated = conn.execute(
            f"""
            UPDATE workflow_input_items
               SET status = (
                    SELECT status FROM workflow_executions
                     WHERE workflow_executions.execution_id = workflow_input_items.execution_id
               ), updated_at = ?
             WHERE status = 'running'
               AND execution_id IN (
                    SELECT execution_id FROM workflow_executions
                     WHERE status IN ({placeholders})
               )
            """,
            (ts, *terminal_statuses),
        ).rowcount
    return int(updated)


def list_executions(
    conn: sqlite3.Connection,
    workflow_id: str | None = None,
    *,
    status: str | None = None,
    version: int | None = None,
    trigger_id: str | None = None,
    before: tuple[int, str] | None = None,
    limit: int | None = None,
) -> list[WorkflowExecution]:
    """List executions newest-first with optional filters.

    ``before`` is a ``(created_at, execution_id)`` cursor for keyset
    pagination — no offset pagination.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if workflow_id:
        clauses.append("workflow_id = ?")
        params.append(workflow_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if version is not None:
        clauses.append("version = ?")
        params.append(version)
    if trigger_id is not None:
        clauses.append("trigger_id = ?")
        params.append(trigger_id)
    if before is not None:
        # In newest-first order, "before the cursor" means newer items.
        clauses.append("(created_at > ? OR (created_at = ? AND execution_id > ?))")
        params.extend([before[0], before[0], before[1]])
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    query = f"SELECT execution_id FROM workflow_executions{where} ORDER BY created_at DESC, execution_id DESC"
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [get_execution(conn, row["execution_id"]) for row in rows]


def get_execution_detail(
    conn: sqlite3.Connection,
    execution_id: str,
) -> dict[str, Any]:
    """Return execution, definition summary, node runs, and events in one response."""
    execution = get_execution(conn, execution_id)
    definition = _definition_record(conn, execution.workflow_id, execution.version)
    node_runs = list_node_runs(conn, execution_id)
    events = list_events(conn, execution_id)
    return {
        "execution": execution,
        "definition": definition,
        "node_runs": node_runs,
        "events": events,
    }


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


def list_events(conn: sqlite3.Connection, execution_id: str) -> list[dict[str, Any]]:
    """Return an execution's event timeline (oldest first), raising on unknown id."""
    get_execution(conn, execution_id)
    rows = conn.execute(
        """
        SELECT id, execution_id, node_run_id, kind, payload_json, created_at
          FROM workflow_events
         WHERE execution_id = ?
         ORDER BY id
        """,
        (execution_id,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "execution_id": row["execution_id"],
            "node_run_id": row["node_run_id"],
            "kind": row["kind"],
            "payload": _json_loads_or_empty(row["payload_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


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
            # Terminalize in-flight node runs so a cancelled execution's
            # drill-down doesn't show nodes eternally "waiting".
            conn.execute(
                """
                UPDATE workflow_node_runs
                   SET status = 'cancelled', completed_at = ?, wait_until = NULL
                 WHERE execution_id = ? AND status IN ('waiting', 'queued')
                """,
                (now, execution_id),
            )
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
