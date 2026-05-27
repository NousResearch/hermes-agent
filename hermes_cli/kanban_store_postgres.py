"""PostgreSQL Kanban storage adapter.

Local runtime adapter for Hermes Kanban.  It intentionally reuses the existing
``hermes_cli.kanban_db`` domain logic through a small DB-API compatibility
wrapper, while replacing the physical SQLite connection with PostgreSQL.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import psycopg

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_store import StoreCapabilities

_DSN_ENV = "HERMES_KANBAN_POSTGRES_DSN"
_SCHEMA_ENV = "HERMES_KANBAN_POSTGRES_SCHEMA"
_DEFAULT_DSN = "postgresql://kanban@/hermes_kanban?host=/home/pancho/.local/hermes-postgres/run&port=55432"
_SERIAL_TABLES = {"task_comments", "task_events", "task_runs"}


class PostgresRow:
    """sqlite3.Row-compatible wrapper over a PostgreSQL tuple row."""

    def __init__(self, columns: list[str], values: tuple[Any, ...]):
        self._columns = columns
        self._values = values
        self._index = {name: i for i, name in enumerate(columns)}

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._values[self._index[key]]

    def keys(self) -> list[str]:
        return list(self._columns)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class PostgresCursor:
    def __init__(self, cursor: Any = None, *, rows: Optional[list[PostgresRow]] = None, rowcount: int = -1, lastrowid: int = 0):
        self._cursor = cursor
        self._rows = rows
        self.rowcount = rowcount if cursor is None else cursor.rowcount
        self.lastrowid = lastrowid
        if cursor is not None and cursor.description:
            self._columns = [col.name for col in cursor.description]
        else:
            self._columns = []

    def _materialize(self) -> list[PostgresRow]:
        if self._rows is not None:
            return self._rows
        if self._cursor is None or not self._cursor.description:
            self._rows = []
            return self._rows
        self._rows = [PostgresRow(self._columns, tuple(row)) for row in self._cursor.fetchall()]
        return self._rows

    def fetchone(self) -> Optional[PostgresRow]:
        rows = self._materialize()
        if not rows:
            return None
        row = rows[0]
        self._rows = rows[1:]
        return row

    def fetchall(self) -> list[PostgresRow]:
        rows = self._materialize()
        self._rows = []
        return rows

    def __iter__(self):
        return iter(self.fetchall())


class PostgresConnection:
    def __init__(self, raw: psycopg.Connection):
        self._raw = raw

    def execute(self, sql: str, parameters: Iterable[Any] = (), /) -> PostgresCursor:
        sql_clean = sql.strip()
        upper = sql_clean.upper()
        if upper.startswith("PRAGMA"):
            return self._handle_pragma(sql_clean)
        if upper == "BEGIN IMMEDIATE":
            self._raw.execute("BEGIN")
            return PostgresCursor(rowcount=-1)
        translated, wants_lastrowid = _translate_sql(sql_clean)
        cur = self._raw.execute(translated, tuple(parameters or ()))
        lastrowid = 0
        if wants_lastrowid and cur.description:
            row = cur.fetchone()
            if row:
                lastrowid = int(row[0])
        return PostgresCursor(cur, lastrowid=lastrowid)

    def executescript(self, sql_script: str, /) -> PostgresCursor:
        # The PostgreSQL adapter owns schema initialization; this is kept for
        # compatibility with DB-API callers that may execute simple scripts.
        for statement in [s.strip() for s in sql_script.split(";") if s.strip()]:
            self.execute(statement)
        return PostgresCursor(rowcount=-1)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()

    def _handle_pragma(self, sql: str) -> PostgresCursor:
        upper = sql.upper()
        if upper.startswith("PRAGMA INTEGRITY_CHECK"):
            return PostgresCursor(rows=[PostgresRow(["integrity_check"], ("ok",))])
        m = re.match(r"PRAGMA\s+table_info\((\w+)\)", sql, flags=re.I)
        if m:
            table = m.group(1)
            with self._raw.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema() AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table,),
                )
                rows = [PostgresRow(["name"], (r[0],)) for r in cur.fetchall()]
            return PostgresCursor(rows=rows)
        return PostgresCursor(rowcount=-1)


class _PostgresContext:
    def __init__(self, dsn: str, schema: Optional[str] = None):
        self.dsn = dsn
        self.schema = schema
        self.conn: Optional[PostgresConnection] = None

    def __enter__(self) -> PostgresConnection:
        raw = psycopg.connect(self.dsn, autocommit=True)
        if self.schema:
            _validate_identifier(self.schema)
            raw.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            raw.execute(f'SET search_path TO "{self.schema}"')
        self.conn = PostgresConnection(raw)
        _init_postgres_schema(self.conn)
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.conn is not None:
            self.conn.close()


def _validate_identifier(value: str) -> None:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        raise ValueError(f"Unsafe PostgreSQL identifier: {value!r}")


def _storage_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        kanban_cfg = cfg.get("kanban") or {}
        storage_cfg = kanban_cfg.get("storage") or {}
        return storage_cfg if isinstance(storage_cfg, dict) else {}
    except Exception:
        return {}


def _configured_dsn() -> str:
    storage = _storage_config()
    return os.environ.get(_DSN_ENV) or storage.get("postgres_dsn") or _DEFAULT_DSN


def _configured_schema() -> Optional[str]:
    storage = _storage_config()
    return os.environ.get(_SCHEMA_ENV) or storage.get("postgres_schema") or _schema_from_dsn_options(_configured_dsn())


def _schema_from_dsn_options(dsn: str) -> Optional[str]:
    try:
        qs = parse_qs(urlparse(dsn).query)
        options = qs.get("options", [""])[0]
        options = unquote(options)
        m = re.search(r"search_path=([A-Za-z_][A-Za-z0-9_]*)", options)
        return m.group(1) if m else None
    except Exception:
        return None


def _replace_qmarks(sql: str) -> str:
    out: list[str] = []
    in_single = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            out.append(ch)
            if in_single and i + 1 < len(sql) and sql[i + 1] == "'":
                out.append(sql[i + 1])
                i += 2
                continue
            in_single = not in_single
        elif ch == "?" and not in_single:
            out.append("%s")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _translate_sql(sql: str) -> tuple[str, bool]:
    sql = re.sub(r"\bINSERT\s+OR\s+IGNORE\s+INTO\s+", "INSERT INTO ", sql, flags=re.I)
    add_do_nothing = "INSERT INTO task_links" in sql and "ON CONFLICT" not in sql.upper()
    wants_lastrowid = False
    m = re.match(r"INSERT\s+INTO\s+(\w+)\s", sql, flags=re.I)
    if m and m.group(1) in _SERIAL_TABLES and "RETURNING" not in sql.upper():
        wants_lastrowid = True
        sql = sql.rstrip().rstrip(";") + " RETURNING id"
    if add_do_nothing:
        sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    sql = _replace_qmarks(sql)
    return sql, wants_lastrowid


POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           BIGINT NOT NULL,
    started_at           BIGINT,
    completed_at         BIGINT,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    branch_name          TEXT,
    claim_lock           TEXT,
    claim_expires        BIGINT,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    BIGINT,
    current_run_id       INTEGER,
    workflow_template_id TEXT,
    current_step_key     TEXT,
    skills               TEXT,
    model_override       TEXT,
    max_retries          INTEGER,
    session_id           TEXT
);
CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);
CREATE TABLE IF NOT EXISTS task_comments (
    id         SERIAL PRIMARY KEY,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_events (
    id         SERIAL PRIMARY KEY,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_runs (
    id                  SERIAL PRIMARY KEY,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    claim_lock          TEXT,
    claim_expires       BIGINT,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   BIGINT,
    started_at          BIGINT NOT NULL,
    ended_at            BIGINT,
    outcome             TEXT,
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    notifier_profile TEXT,
    created_at    BIGINT NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_task             ON task_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status           ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_notify_task           ON kanban_notify_subs(task_id);
"""


def _init_postgres_schema(conn: PostgresConnection) -> None:
    for statement in [s.strip() for s in POSTGRES_SCHEMA_SQL.split(";") if s.strip()]:
        conn.execute(statement)


@dataclass(frozen=True)
class PostgresKanbanStore:
    """PostgreSQL-backed Kanban store for local high-concurrency boards."""

    capabilities: StoreCapabilities = StoreCapabilities(
        backend="postgres",
        supports_row_level_locking=True,
        supports_skip_locked=True,
        supports_concurrent_writers=True,
        production_ready=True,
    )

    def __getattr__(self, name: str) -> Any:
        return getattr(kb, name)

    def connect(self, *args: Any, **kwargs: Any) -> _PostgresContext:
        return _PostgresContext(_configured_dsn(), _configured_schema())

    def init_db(self, conn: Optional[PostgresConnection] = None, *args: Any, **kwargs: Any) -> None:
        if conn is not None:
            _init_postgres_schema(conn)
            return
        with self.connect() as pg:
            _init_postgres_schema(pg)

    def write_txn(self, conn: PostgresConnection):
        return kb.write_txn(conn)

    def run_daemon(self, *args: Any, **kwargs: Any):
        return kb.run_daemon(*args, **kwargs)

    def create_task(self, conn, **kwargs: Any) -> str:
        return kb.create_task(conn, **kwargs)

    def get_task(self, conn, task_id: str):
        return kb.get_task(conn, task_id)

    def list_tasks(self, conn, *args: Any, **kwargs: Any):
        return kb.list_tasks(conn, *args, **kwargs)

    def recompute_ready(self, conn) -> int:
        return kb.recompute_ready(conn)

    def claim_task(self, conn, task_id: str, *, ttl_seconds: Optional[int] = None, claimer: Optional[str] = None):
        return kb.claim_task(conn, task_id, ttl_seconds=ttl_seconds, claimer=claimer)

    def heartbeat_claim(self, conn, task_id: str, *, ttl_seconds: Optional[int] = None, claimer: Optional[str] = None) -> bool:
        return kb.heartbeat_claim(conn, task_id, ttl_seconds=ttl_seconds, claimer=claimer)

    def release_stale_claims(self, conn, *args: Any, **kwargs: Any) -> int:
        return kb.release_stale_claims(conn, *args, **kwargs)

    def complete_task(self, conn, task_id: str, **kwargs: Any) -> bool:
        return kb.complete_task(conn, task_id, **kwargs)

    def block_task(self, conn, task_id: str, *args: Any, **kwargs: Any) -> bool:
        return kb.block_task(conn, task_id, *args, **kwargs)

    def unblock_task(self, conn, task_id: str) -> bool:
        return kb.unblock_task(conn, task_id)

    def add_comment(self, conn, task_id: str, author: str, body: str) -> int:
        return kb.add_comment(conn, task_id, author, body)

    def list_comments(self, conn, task_id: str):
        return kb.list_comments(conn, task_id)

    def list_events(self, conn, task_id: str):
        return kb.list_events(conn, task_id)

    def dispatch_once(self, conn, *args: Any, **kwargs: Any):
        return kb.dispatch_once(conn, *args, **kwargs)

    def board_stats(self, conn, *args: Any, **kwargs: Any) -> dict[str, int]:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
        stats = {status: 0 for status in ["todo", "ready", "running", "blocked", "done", "archived", "scheduled"]}
        for row in rows:
            stats[row["status"]] = int(row["n"])
        return stats


def _delegate_to_legacy(name: str):
    def method(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(kb, name)(*args, **kwargs)
    method.__name__ = name
    method.__qualname__ = f"PostgresKanbanStore.{name}"
    return method


for _method_name in [
    "add_notify_sub", "archive_task", "assign_task", "board_exists",
    "build_worker_context", "child_ids", "create_board",
    "delete_archived_task", "edit_completed_task_result", "gc_events",
    "gc_worker_logs", "get_current_board", "has_spawnable_ready",
    "heartbeat_worker", "kanban_db_path", "known_assignees",
    "latest_summary", "link_tasks", "list_boards", "list_notify_subs",
    "list_profiles_on_disk", "list_runs", "parent_ids", "promote_task",
    "read_board_metadata", "read_worker_log", "reassign_task",
    "reclaim_task", "remove_board", "remove_notify_sub",
    "resolve_workspace", "schedule_task", "set_current_board",
    "set_workspace_path", "unlink_tasks", "workspaces_root",
    "write_board_metadata",
]:
    if not hasattr(PostgresKanbanStore, _method_name):
        setattr(PostgresKanbanStore, _method_name, _delegate_to_legacy(_method_name))


__all__ = ["PostgresKanbanStore", "PostgresConnection", "PostgresRow"]
