"""Postgres-backed kanban shim preserving the sqlite caller surface.

This module keeps the externally used ``hermes_cli.kanban_db`` function
signatures intact while storing board state in Postgres. Compatibility is
provided by replaying each board into a transient SQLite database, delegating
business logic to ``hermes_cli.kanban_db``, then syncing the resulting board
state back into Postgres inside a transaction-scoped advisory lock.

The approach is intentionally conservative:
- callers keep their existing Python API surface
- filesystem-backed board metadata / workspace helpers remain unchanged
- Postgres is the persistence layer and single source of truth for board rows
- the existing SQLite implementation remains untouched and reusable

This is a bridge module, not the backend selector / cutover itself.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from hermes_cli import kanban_db_sqlite as _sqlite

try:  # pragma: no cover - exercised in environments with psycopg installed
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - tests monkeypatch the transport
    psycopg = None


# Re-export the shared types / constants so callers see the same shapes.
Task = _sqlite.Task
Run = _sqlite.Run
Comment = _sqlite.Comment
Event = _sqlite.Event
DispatchResult = _sqlite.DispatchResult
HallucinatedCardsError = _sqlite.HallucinatedCardsError

VALID_STATUSES = _sqlite.VALID_STATUSES
VALID_INITIAL_STATUSES = _sqlite.VALID_INITIAL_STATUSES
VALID_WORKSPACE_KINDS = _sqlite.VALID_WORKSPACE_KINDS
KNOWN_TOOLSET_NAMES = _sqlite.KNOWN_TOOLSET_NAMES
DEFAULT_CLAIM_TTL_SECONDS = _sqlite.DEFAULT_CLAIM_TTL_SECONDS
DEFAULT_SPAWN_FAILURE_LIMIT = _sqlite.DEFAULT_SPAWN_FAILURE_LIMIT
DEFAULT_BOARD = _sqlite.DEFAULT_BOARD

_normalize_board_slug = _sqlite._normalize_board_slug
_resolve_claim_ttl_seconds = _sqlite._resolve_claim_ttl_seconds
kanban_home = _sqlite.kanban_home
boards_root = _sqlite.boards_root
current_board_path = _sqlite.current_board_path
get_current_board = _sqlite.get_current_board
set_current_board = _sqlite.set_current_board
kanban_db_path = _sqlite.kanban_db_path
workspaces_root = _sqlite.workspaces_root
board_exists = _sqlite.board_exists
list_boards = _sqlite.list_boards
read_board_metadata = _sqlite.read_board_metadata
write_board_metadata = _sqlite.write_board_metadata
create_board = _sqlite.create_board
remove_board = _sqlite.remove_board
list_profiles_on_disk = _sqlite.list_profiles_on_disk
resolve_workspace = _sqlite.resolve_workspace
read_worker_log = _sqlite.read_worker_log
worker_logs_dir = _sqlite.worker_logs_dir
worker_log_path = _sqlite.worker_log_path
_gc_worker_logs_impl = _sqlite.gc_worker_logs


_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "tasks": (
        "id", "title", "body", "assignee", "status", "priority", "created_by",
        "created_at", "started_at", "completed_at", "workspace_kind",
        "workspace_path", "branch_name", "claim_lock", "claim_expires", "tenant",
        "result", "idempotency_key", "consecutive_failures", "worker_pid",
        "last_failure_error", "max_runtime_seconds", "last_heartbeat_at",
        "current_run_id", "workflow_template_id", "current_step_key", "skills",
        "model_override", "max_retries", "session_id",
    ),
    "task_links": ("parent_id", "child_id"),
    "task_comments": ("id", "task_id", "author", "body", "created_at"),
    "task_events": ("id", "task_id", "run_id", "kind", "payload", "created_at"),
    "task_runs": (
        "id", "task_id", "profile", "step_key", "status", "claim_lock",
        "claim_expires", "worker_pid", "max_runtime_seconds", "last_heartbeat_at",
        "started_at", "ended_at", "outcome", "summary", "metadata", "error",
    ),
    "kanban_notify_subs": (
        "task_id", "platform", "chat_id", "thread_id", "user_id",
        "notifier_profile", "created_at", "last_event_id",
    ),
}

_TABLE_LOAD_ORDER = (
    "tasks",
    "task_links",
    "task_comments",
    "task_events",
    "task_runs",
    "kanban_notify_subs",
)

_TABLE_FLUSH_ORDER = (
    "task_links",
    "task_comments",
    "task_events",
    "task_runs",
    "kanban_notify_subs",
    "tasks",
)

_PG_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.kanban_boards (
    board_slug TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    icon TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '',
    default_workdir TEXT,
    legacy_db_path TEXT,
    legacy_workspace_root TEXT,
    legacy_logs_root TEXT,
    archived BOOLEAN NOT NULL DEFAULT FALSE,
    created_at BIGINT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS {schema}.kanban_tasks (
    board_slug TEXT NOT NULL,
    id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    assignee TEXT,
    status TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    created_by TEXT,
    created_at BIGINT NOT NULL,
    started_at BIGINT,
    completed_at BIGINT,
    workspace_kind TEXT NOT NULL DEFAULT 'scratch',
    workspace_path TEXT,
    branch_name TEXT,
    claim_lock TEXT,
    claim_expires BIGINT,
    tenant TEXT,
    result TEXT,
    idempotency_key TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid INTEGER,
    last_failure_error TEXT,
    max_runtime_seconds INTEGER,
    last_heartbeat_at BIGINT,
    current_run_id BIGINT,
    workflow_template_id TEXT,
    current_step_key TEXT,
    skills TEXT,
    model_override TEXT,
    max_retries INTEGER,
    session_id TEXT,
    PRIMARY KEY (board_slug, id)
);

CREATE TABLE IF NOT EXISTS {schema}.kanban_task_links (
    board_slug TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    PRIMARY KEY (board_slug, parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS {schema}.kanban_task_comments (
    board_slug TEXT NOT NULL,
    id BIGINT NOT NULL,
    task_id TEXT NOT NULL,
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (board_slug, id)
);

CREATE TABLE IF NOT EXISTS {schema}.kanban_task_events (
    board_slug TEXT NOT NULL,
    id BIGINT NOT NULL,
    task_id TEXT NOT NULL,
    run_id BIGINT,
    kind TEXT NOT NULL,
    payload TEXT,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (board_slug, id)
);

CREATE TABLE IF NOT EXISTS {schema}.kanban_task_runs (
    board_slug TEXT NOT NULL,
    id BIGINT NOT NULL,
    task_id TEXT NOT NULL,
    profile TEXT,
    step_key TEXT,
    status TEXT NOT NULL,
    claim_lock TEXT,
    claim_expires BIGINT,
    worker_pid INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at BIGINT,
    started_at BIGINT NOT NULL,
    ended_at BIGINT,
    outcome TEXT,
    summary TEXT,
    metadata TEXT,
    error TEXT,
    PRIMARY KEY (board_slug, id)
);

CREATE TABLE IF NOT EXISTS {schema}.kanban_notify_subs (
    board_slug TEXT NOT NULL,
    task_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    thread_id TEXT NOT NULL DEFAULT '',
    user_id TEXT,
    notifier_profile TEXT,
    created_at BIGINT NOT NULL,
    last_event_id BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (board_slug, task_id, platform, chat_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_pg_tasks_assignee_status ON {schema}.kanban_tasks(board_slug, assignee, status);
CREATE INDEX IF NOT EXISTS idx_pg_tasks_status ON {schema}.kanban_tasks(board_slug, status);
CREATE INDEX IF NOT EXISTS idx_pg_links_child ON {schema}.kanban_task_links(board_slug, child_id);
CREATE INDEX IF NOT EXISTS idx_pg_links_parent ON {schema}.kanban_task_links(board_slug, parent_id);
CREATE INDEX IF NOT EXISTS idx_pg_comments_task ON {schema}.kanban_task_comments(board_slug, task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pg_events_task ON {schema}.kanban_task_events(board_slug, task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pg_runs_task ON {schema}.kanban_task_runs(board_slug, task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_pg_runs_status ON {schema}.kanban_task_runs(board_slug, status);
CREATE INDEX IF NOT EXISTS idx_pg_notify_task ON {schema}.kanban_notify_subs(board_slug, task_id);
"""


class PgKanbanConnection:
    """Board-scoped shim connection that exposes a SQLite-like read surface."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        board: Optional[str] = None,
        dsn: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.board = _normalize_board_slug(board) or DEFAULT_BOARD
        self.dsn = dsn or os.environ.get("HERMES_KANBAN_POSTGRES_DSN", "").strip()
        self.schema = (schema or os.environ.get("HERMES_KANBAN_POSTGRES_SCHEMA", "core") or "core").strip()
        self._pg = _open_pg_connection(self.dsn)
        self._tmpdir = Path(tempfile.mkdtemp(prefix=f"kanban-pg-{self.board}-"))
        self._sqlite_path = self._tmpdir / "shim.sqlite3"
        self._sqlite = _new_sqlite_snapshot(self._sqlite_path)
        self._closed = False
        self._ensure_schema()
        self._refresh_from_postgres()

    def __enter__(self) -> "PgKanbanConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._sqlite.close()
        with contextlib.suppress(Exception):
            self._pg.close()
        with contextlib.suppress(Exception):
            self._sqlite_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            self._tmpdir.rmdir()

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def cursor(self):
        self._refresh_from_postgres()
        return self._sqlite.cursor()

    def execute(self, sql: str, params: Sequence[Any] = ()):  # SQLite-compatible read surface
        self._refresh_from_postgres()
        return self._sqlite.execute(sql, tuple(params))

    def with_snapshot(self, writable: bool, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        if writable:
            try:
                self._lock_board()
                self._refresh_from_postgres()
                result = fn(self._sqlite)
                self._flush_to_postgres()
                self._pg.commit()
                return result
            except Exception:
                with contextlib.suppress(Exception):
                    self._pg.rollback()
                raise
        self._refresh_from_postgres()
        try:
            return fn(self._sqlite)
        finally:
            with contextlib.suppress(Exception):
                self._pg.rollback()

    def _ensure_schema(self) -> None:
        with self._pg.cursor() as cur:
            cur.execute(_PG_SCHEMA_SQL.format(schema=self.schema))
        self._pg.commit()
        self._ensure_board_row()

    def _ensure_board_row(self) -> None:
        meta = read_board_metadata(self.board)
        display_name = str(meta.get("name") or self.board)
        description = str(meta.get("description") or "")
        icon = str(meta.get("icon") or "")
        color = str(meta.get("color") or "")
        default_workdir = meta.get("default_workdir")
        archived = bool(meta.get("archived", False))
        legacy_db = str(kanban_db_path(self.board))
        legacy_ws = str(workspaces_root(self.board))
        legacy_logs = str(worker_logs_dir(self.board))
        with self._pg.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.schema}.kanban_boards (
                    board_slug, display_name, description, icon, color,
                    default_workdir, legacy_db_path, legacy_workspace_root,
                    legacy_logs_root, archived, created_at, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, EXTRACT(EPOCH FROM NOW())::bigint, %s::jsonb)
                ON CONFLICT (board_slug) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    description = EXCLUDED.description,
                    icon = EXCLUDED.icon,
                    color = EXCLUDED.color,
                    default_workdir = EXCLUDED.default_workdir,
                    legacy_db_path = EXCLUDED.legacy_db_path,
                    legacy_workspace_root = EXCLUDED.legacy_workspace_root,
                    legacy_logs_root = EXCLUDED.legacy_logs_root,
                    archived = EXCLUDED.archived,
                    metadata = EXCLUDED.metadata
                """,
                (
                    self.board,
                    display_name,
                    description,
                    icon,
                    color,
                    default_workdir,
                    legacy_db,
                    legacy_ws,
                    legacy_logs,
                    archived,
                    "{}",
                ),
            )
        self._pg.commit()

    def _lock_board(self) -> None:
        with self._pg.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"kanban:{self.schema}:{self.board}",))

    def _refresh_from_postgres(self) -> None:
        self._reset_sqlite_snapshot()
        self._ensure_board_row()
        for table in _TABLE_LOAD_ORDER:
            rows = self._fetch_rows(table)
            if not rows:
                continue
            cols = _TABLE_COLUMNS[table]
            placeholders = ", ".join("?" for _ in cols)
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
            self._sqlite.executemany(sql, [tuple(row.get(col) for col in cols) for row in rows])
        self._sqlite.commit()

    def _reset_sqlite_snapshot(self) -> None:
        with contextlib.suppress(Exception):
            self._sqlite.close()
        self._sqlite = _new_sqlite_snapshot(self._sqlite_path)

    def _flush_to_postgres(self) -> None:
        self._ensure_board_row()
        with self._pg.cursor() as cur:
            for table in _TABLE_FLUSH_ORDER:
                cur.execute(f"DELETE FROM {self._qualified_table(table)} WHERE board_slug = %s", (self.board,))
            for table in _TABLE_LOAD_ORDER:
                cols = _TABLE_COLUMNS[table]
                rows = self._sqlite.execute(
                    f"SELECT {', '.join(cols)} FROM {table}"
                ).fetchall()
                if not rows:
                    continue
                sql = (
                    f"INSERT INTO {self._qualified_table(table)} (board_slug, {', '.join(cols)}) "
                    f"VALUES ({', '.join(['%s'] * (len(cols) + 1))})"
                )
                payload = [tuple([self.board] + [row[col] for col in cols]) for row in rows]
                cur.executemany(sql, payload)

    def _fetch_rows(self, table: str) -> list[dict[str, Any]]:
        cols = _TABLE_COLUMNS[table]
        with self._pg.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(cols)} FROM {self._qualified_table(table)} WHERE board_slug = %s ORDER BY {self._order_by(table)}",
                (self.board,),
            )
            rows = cur.fetchall() or []
            names = [getattr(d, "name", d[0]) for d in cur.description or []]
        return [dict(zip(names, row)) for row in rows]

    def _qualified_table(self, table: str) -> str:
        mapping = {
            "tasks": "kanban_tasks",
            "task_links": "kanban_task_links",
            "task_comments": "kanban_task_comments",
            "task_events": "kanban_task_events",
            "task_runs": "kanban_task_runs",
            "kanban_notify_subs": "kanban_notify_subs",
        }
        return f"{self.schema}.{mapping[table]}"

    def _order_by(self, table: str) -> str:
        return {
            "tasks": "id",
            "task_links": "parent_id, child_id",
            "task_comments": "id",
            "task_events": "id",
            "task_runs": "id",
            "kanban_notify_subs": "task_id, platform, chat_id, thread_id",
        }[table]


def _new_sqlite_snapshot(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(_sqlite.SCHEMA_SQL)
    _sqlite._migrate_add_optional_columns(conn)
    return conn


def _open_pg_connection(dsn: str):
    if not dsn:
        raise RuntimeError(
            "kanban_db_pg.connect requires HERMES_KANBAN_POSTGRES_DSN or an explicit dsn"
        )
    if psycopg is None:
        raise RuntimeError(
            "kanban_db_pg requires psycopg (psycopg[binary]) to be installed"
        )
    return psycopg.connect(dsn)


def _with_snapshot(conn, writable: bool, fn: Callable[[sqlite3.Connection], Any]) -> Any:
    if hasattr(conn, "with_snapshot"):
        return conn.with_snapshot(writable, fn)
    return fn(conn)


def _read(conn, fn: Callable[..., Any], *args, **kwargs):
    return _with_snapshot(conn, False, lambda sqlite_conn: fn(sqlite_conn, *args, **kwargs))


def _write(conn, fn: Callable[..., Any], *args, **kwargs):
    return _with_snapshot(conn, True, lambda sqlite_conn: fn(sqlite_conn, *args, **kwargs))


def connect(db_path=None, *, board=None):
    return PgKanbanConnection(db_path=db_path, board=board)


def init_db(db_path=None, *, board=None):
    with contextlib.closing(connect(db_path=db_path, board=board)):
        return None


def add_comment(conn, task_id, author, body):
    return _write(conn, _sqlite.add_comment, task_id, author, body)


def add_notify_sub(conn, *, task_id, platform, chat_id, thread_id=None, user_id=None, notifier_profile=None):
    return _write(
        conn,
        _sqlite.add_notify_sub,
        task_id=task_id,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=user_id,
        notifier_profile=notifier_profile,
    )


def advance_notify_cursor(conn, *, task_id, platform, chat_id, thread_id=None, new_cursor):
    return _write(
        conn,
        _sqlite.advance_notify_cursor,
        task_id=task_id,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        new_cursor=new_cursor,
    )


def archive_task(conn, task_id):
    return _write(conn, _sqlite.archive_task, task_id)


def assign_task(conn, task_id, profile):
    return _write(conn, _sqlite.assign_task, task_id, profile)


def block_task(conn, task_id, *, reason=None, expected_run_id=None):
    return _write(conn, _sqlite.block_task, task_id, reason=reason, expected_run_id=expected_run_id)


def board_stats(conn):
    return _read(conn, _sqlite.board_stats)


def build_worker_context(conn, task_id):
    return _read(conn, _sqlite.build_worker_context, task_id)


def child_ids(conn, task_id):
    return _read(conn, _sqlite.child_ids, task_id)


def claim_task(conn, task_id, *, ttl_seconds=None, claimer=None):
    return _write(conn, _sqlite.claim_task, task_id, ttl_seconds=ttl_seconds, claimer=claimer)


def claim_unseen_events_for_sub(conn, *, task_id, platform, chat_id, thread_id=None, kinds=None):
    return _write(
        conn,
        _sqlite.claim_unseen_events_for_sub,
        task_id=task_id,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        kinds=kinds,
    )


def complete_task(conn, task_id, *, result=None, summary=None, metadata=None, created_cards=None, expected_run_id=None):
    return _write(
        conn,
        _sqlite.complete_task,
        task_id,
        result=result,
        summary=summary,
        metadata=metadata,
        created_cards=created_cards,
        expected_run_id=expected_run_id,
    )


def create_task(conn, *, title, body=None, assignee=None, created_by=None, workspace_kind='scratch', workspace_path=None, branch_name=None, tenant=None, priority=0, parents=(), triage=False, idempotency_key=None, max_runtime_seconds=None, skills=None, max_retries=None, initial_status='running', session_id=None, board=None):
    return _write(
        conn,
        _sqlite.create_task,
        title=title,
        body=body,
        assignee=assignee,
        created_by=created_by,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        branch_name=branch_name,
        tenant=tenant,
        priority=priority,
        parents=parents,
        triage=triage,
        idempotency_key=idempotency_key,
        max_runtime_seconds=max_runtime_seconds,
        skills=skills,
        max_retries=max_retries,
        initial_status=initial_status,
        session_id=session_id,
        board=board,
    )


def delete_archived_task(conn, task_id):
    return _write(conn, _sqlite.delete_archived_task, task_id)


def dispatch_once(conn, *, spawn_fn=None, ttl_seconds=None, dry_run=False, max_spawn=None, max_in_progress=None, failure_limit=DEFAULT_SPAWN_FAILURE_LIMIT, stale_timeout_seconds=0, board=None):
    return _write(
        conn,
        _sqlite.dispatch_once,
        spawn_fn=spawn_fn,
        ttl_seconds=ttl_seconds,
        dry_run=dry_run,
        max_spawn=max_spawn,
        max_in_progress=max_in_progress,
        failure_limit=failure_limit,
        stale_timeout_seconds=stale_timeout_seconds,
        board=board,
    )


def edit_completed_task_result(conn, task_id, *, result, summary=None, metadata=None):
    return _write(conn, _sqlite.edit_completed_task_result, task_id, result=result, summary=summary, metadata=metadata)


def gc_events(conn, *, older_than_seconds=30 * 24 * 3600):
    return _write(conn, _sqlite.gc_events, older_than_seconds=older_than_seconds)


def gc_worker_logs(*, older_than_seconds=30 * 24 * 3600, board=None):
    return _gc_worker_logs_impl(older_than_seconds=older_than_seconds, board=board)


def get_task(conn, task_id):
    return _read(conn, _sqlite.get_task, task_id)


def has_spawnable_ready(conn):
    return _read(conn, _sqlite.has_spawnable_ready)


def has_spawnable_review(conn):
    return _read(conn, _sqlite.has_spawnable_review)


def heartbeat_claim(conn, task_id, *, ttl_seconds=None, claimer=None):
    return _write(conn, _sqlite.heartbeat_claim, task_id, ttl_seconds=ttl_seconds, claimer=claimer)


def heartbeat_worker(conn, task_id, *, note=None, expected_run_id=None):
    return _write(conn, _sqlite.heartbeat_worker, task_id, note=note, expected_run_id=expected_run_id)


def known_assignees(conn):
    return _read(conn, _sqlite.known_assignees)


def latest_run(conn, task_id):
    return _read(conn, _sqlite.latest_run, task_id)


def latest_summary(conn, task_id):
    return _read(conn, _sqlite.latest_summary, task_id)


def link_tasks(conn, parent_id, child_id):
    return _write(conn, _sqlite.link_tasks, parent_id, child_id)


def list_comments(conn, task_id):
    return _read(conn, _sqlite.list_comments, task_id)


def list_events(conn, task_id):
    return _read(conn, _sqlite.list_events, task_id)


def list_notify_subs(conn, task_id=None):
    return _read(conn, _sqlite.list_notify_subs, task_id=task_id)


def list_runs(conn, task_id, *, include_active=True, state_type=None, state_name=None):
    return _read(conn, _sqlite.list_runs, task_id, include_active=include_active, state_type=state_type, state_name=state_name)


def list_tasks(conn, *, assignee=None, status=None, tenant=None, session_id=None, include_archived=False, limit=None, order_by=None, workflow_template_id=None, current_step_key=None):
    return _read(
        conn,
        _sqlite.list_tasks,
        assignee=assignee,
        status=status,
        tenant=tenant,
        session_id=session_id,
        include_archived=include_archived,
        limit=limit,
        order_by=order_by,
        workflow_template_id=workflow_template_id,
        current_step_key=current_step_key,
    )


def parent_ids(conn, task_id):
    return _read(conn, _sqlite.parent_ids, task_id)


def reassign_task(conn, task_id, profile, *, reclaim_first=False, reason=None):
    return _write(conn, _sqlite.reassign_task, task_id, profile, reclaim_first=reclaim_first, reason=reason)


def reclaim_task(conn, task_id, *, reason=None, signal_fn=None):
    return _write(conn, _sqlite.reclaim_task, task_id, reason=reason, signal_fn=signal_fn)


def recompute_ready(conn):
    return _write(conn, _sqlite.recompute_ready)


def remove_notify_sub(conn, *, task_id, platform, chat_id, thread_id=None):
    return _write(
        conn,
        _sqlite.remove_notify_sub,
        task_id=task_id,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
    )


def rewind_notify_cursor(conn, *, task_id, platform, chat_id, thread_id=None, claimed_cursor, old_cursor):
    return _write(
        conn,
        _sqlite.rewind_notify_cursor,
        task_id=task_id,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        claimed_cursor=claimed_cursor,
        old_cursor=old_cursor,
    )


def run_daemon(*, interval=60.0, max_spawn=None, failure_limit=DEFAULT_SPAWN_FAILURE_LIMIT, stop_event=None, on_tick=None):
    import signal
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    def _handle(_signum, _frame):
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handle)
                except (ValueError, OSError):
                    pass

    while not stop_event.is_set():
        try:
            with contextlib.closing(connect()) as conn:
                res = dispatch_once(conn, max_spawn=max_spawn, failure_limit=failure_limit)
            if on_tick is not None:
                try:
                    on_tick(res)
                except Exception:
                    pass
        except Exception:
            import traceback
            traceback.print_exc()
        stop_event.wait(timeout=interval)


def schedule_task(conn, task_id, *, reason=None, expected_run_id=None):
    return _write(conn, _sqlite.schedule_task, task_id, reason=reason, expected_run_id=expected_run_id)


def set_workspace_path(conn, task_id, path):
    return _write(conn, _sqlite.set_workspace_path, task_id, path)


def unblock_task(conn, task_id):
    return _write(conn, _sqlite.unblock_task, task_id)


def unlink_tasks(conn, parent_id, child_id):
    return _write(conn, _sqlite.unlink_tasks, parent_id, child_id)
