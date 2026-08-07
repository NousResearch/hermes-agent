"""Durable, profile-local usage receipts for direct and Kanban runs.

Runtime hooks only enqueue in-memory operations after the ledger has been
initialised. A dedicated writer owns SQLite writes. Finalization drains the
writer, replays failed/overflow operations, persists diagnostics, and writes the
finish marker synchronously so a completed run cannot silently lose its totals.
"""

from __future__ import annotations

import atexit
from collections import deque
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_default_hermes_root, get_hermes_home

logger = logging.getLogger(__name__)

_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_runs (
    run_id TEXT PRIMARY KEY,
    process_id TEXT NOT NULL,
    task_run_id INTEGER,
    session_id TEXT,
    task_id TEXT,
    board TEXT,
    model TEXT,
    provider TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    cost_usd REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    turn_count INTEGER NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    tool_call_count INTEGER NOT NULL DEFAULT 0 CHECK (tool_call_count >= 0),
    elapsed REAL CHECK (elapsed IS NULL OR elapsed >= 0),
    outcome TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    failure_reason TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    run_id TEXT NOT NULL REFERENCES usage_runs(run_id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    session_id TEXT,
    turn_id TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    cost_usd REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    model TEXT,
    provider TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (run_id, event_id)
);
CREATE TABLE IF NOT EXISTS usage_turns (
    run_id TEXT NOT NULL REFERENCES usage_runs(run_id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL,
    PRIMARY KEY (run_id, turn_id)
);
CREATE TABLE IF NOT EXISTS usage_run_models (
    run_id TEXT NOT NULL REFERENCES usage_runs(run_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown',
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    cost_usd REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    event_count INTEGER NOT NULL DEFAULT 0 CHECK (event_count >= 0),
    PRIMARY KEY (run_id, model, provider)
);
CREATE TABLE IF NOT EXISTS usage_diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    diagnostic_type TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1 CHECK (count > 0),
    detail TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_diagnostics_run ON usage_diagnostics(run_id, created_at);
"""

_PROCESS_ID = f"proc-{os.getpid()}-{uuid.uuid4().hex}"
_PROCESS_INVOCATION_ID = _PROCESS_ID
_CURRENT_SESSION: ContextVar[str | None] = ContextVar("hermes_usage_session", default=None)
_LEDGER_CACHE: dict[str, "UsageLedger"] = {}
_LEDGER_CACHE_LOCK = threading.Lock()


def current_source_profile() -> str:
    """Return the active profile without requiring HERMES_PROFILE to be set."""
    home = get_hermes_home().resolve()
    root = get_default_hermes_root().resolve()
    try:
        relative = home.relative_to(root)
        if len(relative.parts) >= 2 and relative.parts[0] == "profiles":
            return relative.parts[1]
    except ValueError:
        pass
    return "default"


def process_run_id() -> str:
    """Return the authoritative task/direct launcher identity."""
    kanban_run = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    if kanban_run:
        return f"task-run:{kanban_run}"
    explicit = os.environ.get("HERMES_RUN_ID", "").strip()
    return explicit or _PROCESS_INVOCATION_ID


def process_invocation_id() -> str:
    """Return a unique identity for this interpreter invocation."""
    return _PROCESS_INVOCATION_ID


def run_id_for_session(session_id: str | None = None) -> str:
    """Resolve one stable receipt identity while preserving session metadata."""
    kanban_run = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    if kanban_run:
        return f"task-run:{kanban_run}"
    explicit = os.environ.get("HERMES_RUN_ID", "").strip()
    if explicit:
        return explicit
    session = str(session_id or _CURRENT_SESSION.get() or "").strip()
    if not session:
        # A session-less direct invocation is still one invocation, not the
        # process-global identity used by unrelated gateway sessions.
        return _PROCESS_INVOCATION_ID
    digest = hashlib.sha256(session.encode("utf-8")).hexdigest()[:16]
    return f"session-run:{_PROCESS_INVOCATION_ID}:{digest}"


def bind_session(session_id: str | None) -> None:
    _CURRENT_SESSION.set(str(session_id) if session_id else None)


def current_task_run_id() -> int | None:
    raw = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        logger.warning("Ignoring non-integer HERMES_KANBAN_RUN_ID=%r", raw)
        return None


def default_ledger_path() -> Path:
    return get_hermes_home() / "state.db"


def _nonempty(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _provider_key(provider: Any) -> str:
    value = _nonempty(provider)
    return value.lower() if value else "unknown"


def _nonnegative_int(value: Any, field: str) -> int:
    result = int(value or 0)
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _nonnegative_float(value: Any, field: str) -> float:
    result = float(value or 0.0)
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


Operation = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]


class UsageLedger:
    """SQLite ledger with bounded asynchronous writes and durable finalization."""

    def __init__(self, database_path: str | Path | None = None, *, queue_size: int = 2048) -> None:
        self.database_path = Path(database_path or default_ledger_path())
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue_limit = max(1, int(queue_size))
        self._queue: deque[Operation] = deque(maxlen=self._queue_limit)
        self._failed_operations: deque[Operation] = deque(maxlen=self._queue_limit)
        self._dropped: dict[tuple[str | None, str], int] = {}
        self._incomplete_runs: dict[str, int] = {}
        self._diagnostic_limit = 256
        self._global_incomplete = False
        self._global_diagnostics: dict[str, int] = {}
        self._queue_cond = threading.Condition()
        self._writer: threading.Thread | None = None
        self._writer_stop = False
        self._writer_busy = False
        self._closed = False
        self._ensure_schema()
        atexit.register(self.shutdown)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _ensure_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript(_USAGE_SCHEMA)
            self._migrate_usage_events(connection)
            cols = {row["name"] for row in connection.execute("PRAGMA table_info(usage_runs)")}
            optional_columns = {
                "task_run_id": "task_run_id INTEGER",
                "session_id": "session_id TEXT",
                "task_id": "task_id TEXT",
                "board": "board TEXT",
                "model": "model TEXT",
                "provider": "provider TEXT",
                "input_tokens": "input_tokens INTEGER NOT NULL DEFAULT 0",
                "output_tokens": "output_tokens INTEGER NOT NULL DEFAULT 0",
                "cost_usd": "cost_usd REAL NOT NULL DEFAULT 0",
                "turn_count": "turn_count INTEGER NOT NULL DEFAULT 0",
                "tool_call_count": "tool_call_count INTEGER NOT NULL DEFAULT 0",
                "elapsed": "elapsed REAL",
                "outcome": "outcome TEXT",
                "retry_count": "retry_count INTEGER NOT NULL DEFAULT 0",
                "failure_reason": "failure_reason TEXT",
                "ended_at": "ended_at REAL",
            }
            for name, definition in optional_columns.items():
                if name not in cols:
                    connection.execute(f"ALTER TABLE usage_runs ADD COLUMN {definition}")
            self._migrate_usage_run_models(connection)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_runs_board ON usage_runs(board, started_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_runs_task ON usage_runs(task_id, started_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_runs_task_run ON usage_runs(task_run_id, started_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_runs_session ON usage_runs(session_id, started_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_run ON usage_events(run_id, created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_event ON usage_events(event_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_run_models_run ON usage_run_models(run_id, model, provider)")

    @staticmethod
    def _migrate_usage_events(connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(usage_events)").fetchall()
        names = {row[1] for row in columns}
        primary_key = [row[1] for row in columns if row[5]]
        if primary_key != ["event_id"]:
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("ALTER TABLE usage_events RENAME TO usage_events_legacy")
            connection.execute(
                """CREATE TABLE usage_events (
                    run_id TEXT NOT NULL REFERENCES usage_runs(run_id) ON DELETE CASCADE,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    session_id TEXT,
                    turn_id TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
                    cost_usd REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
                    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
                    model TEXT,
                    provider TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, event_id)
                )"""
            )
            source = {
                "run_id": "run_id",
                "event_id": "event_id",
                "event_type": "event_type",
                "session_id": "session_id" if "session_id" in names else "NULL",
                "turn_id": "turn_id" if "turn_id" in names else "NULL",
                "input_tokens": "input_tokens" if "input_tokens" in names else "0",
                "output_tokens": "output_tokens" if "output_tokens" in names else "0",
                "cost_usd": "cost_usd" if "cost_usd" in names else "0",
                "retry_count": "retry_count" if "retry_count" in names else "0",
                "model": "model" if "model" in names else "NULL",
                "provider": "provider" if "provider" in names else "NULL",
                "created_at": "created_at" if "created_at" in names else "0",
            }
            target_columns = ", ".join(source)
            source_columns = ", ".join(source.values())
            connection.execute(
                f"INSERT INTO usage_events({target_columns}) SELECT {source_columns} FROM usage_events_legacy"
            )
            connection.execute("DROP TABLE usage_events_legacy")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _migrate_usage_run_models(connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(usage_run_models)").fetchall()
        if not columns:
            return
        names = {row[1] for row in columns}
        primary_key = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
        provider_not_null = any(row[1] == "provider" and row[3] for row in columns)
        if names >= {"run_id", "model", "provider"} and provider_not_null and primary_key == ["run_id", "model", "provider"]:
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("ALTER TABLE usage_run_models RENAME TO usage_run_models_legacy")
            connection.execute(
                """CREATE TABLE usage_run_models (
                    run_id TEXT NOT NULL REFERENCES usage_runs(run_id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'unknown',
                    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
                    cost_usd REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
                    event_count INTEGER NOT NULL DEFAULT 0 CHECK (event_count >= 0),
                    PRIMARY KEY (run_id, model, provider)
                )"""
            )
            provider_expr = "COALESCE(NULLIF(provider, ''), 'unknown')" if "provider" in names else "'unknown'"
            connection.execute(
                f"""INSERT INTO usage_run_models(
                    run_id, model, provider, input_tokens, output_tokens, cost_usd, event_count
                ) SELECT run_id, model, {provider_expr},
                    SUM(input_tokens), SUM(output_tokens), SUM(cost_usd), SUM(event_count)
                FROM usage_run_models_legacy
                GROUP BY run_id, model, {provider_expr}"""
            )
            connection.execute("DROP TABLE usage_run_models_legacy")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _mark_global_incomplete(self, reason: str) -> None:
        self._global_incomplete = True
        if reason in self._global_diagnostics or len(self._global_diagnostics) < self._diagnostic_limit:
            self._global_diagnostics[reason] = self._global_diagnostics.get(reason, 0) + 1

    def _record_drop(self, run_id: str | None, kind: str) -> None:
        key = (run_id, kind)
        if key in self._dropped or len(self._dropped) < self._diagnostic_limit:
            self._dropped[key] = self._dropped.get(key, 0) + 1
        else:
            self._mark_global_incomplete("diagnostic_capacity_exhausted")
        if run_id and (run_id in self._incomplete_runs or len(self._incomplete_runs) < self._diagnostic_limit):
            self._incomplete_runs[run_id] = self._incomplete_runs.get(run_id, 0) + 1
        else:
            self._mark_global_incomplete("run_capacity_exhausted")

    def _enqueue(self, fn: Callable[..., Any], *args: Any, critical: bool = False, **kwargs: Any) -> bool:
        operation: Operation = (fn, args, kwargs)
        with self._queue_cond:
            if self._closed or self._writer_stop:
                self._record_drop(kwargs.get("run_id"), "closed")
                return False
            if len(self._queue) >= self._queue_limit:
                if critical:
                    self._record_drop(kwargs.get("run_id"), "queue_saturated")
                    return False
                self._record_drop(kwargs.get("run_id"), fn.__name__)
                logger.warning("usage ledger queue full; dropping noncritical event (%s)", fn.__name__)
                return False
            self._queue.append(operation)
            if self._writer is None or not self._writer.is_alive():
                self._writer = threading.Thread(target=self._writer_loop, name="usage-ledger-writer", daemon=True)
                self._writer.start()
            self._queue_cond.notify_all()
            return True

    def _writer_loop(self) -> None:
        while True:
            with self._queue_cond:
                while not self._queue and not self._writer_stop:
                    self._queue_cond.wait()
                if not self._queue:
                    return
                self._writer_busy = True
                operation = self._queue.popleft()
            try:
                operation[0](*operation[1], **operation[2])
            except Exception:
                with self._queue_cond:
                    if len(self._failed_operations) < self._queue_limit:
                        self._failed_operations.append(operation)
                        key = (operation[2].get("run_id"), "writer_failure")
                        if key in self._dropped or len(self._dropped) < self._diagnostic_limit:
                            self._dropped[key] = self._dropped.get(key, 0) + 1
                        else:
                            self._mark_global_incomplete("diagnostic_capacity_exhausted")
                    else:
                        self._mark_global_incomplete("failed_buffer_full")
                        self._record_drop(operation[2].get("run_id"), "failed_buffer_full")
                logger.warning("usage ledger write failed; continuing runtime", exc_info=True)
            finally:
                with self._queue_cond:
                    self._writer_busy = False
                    self._queue_cond.notify_all()

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._queue_cond:
                if not self._queue and not self._writer_busy:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue_cond.wait(min(remaining, 0.05))

    def _replay_failed_sync(self) -> None:
        with self._queue_cond:
            failed = list(self._failed_operations)
            self._failed_operations.clear()
        still_failed: list[Operation] = []
        for operation in failed:
            try:
                operation[0](*operation[1], **operation[2])
            except Exception:
                still_failed.append(operation)
        if still_failed:
            with self._queue_cond:
                available = self._queue_limit - len(self._failed_operations)
                for operation in still_failed[:available]:
                    self._failed_operations.append(operation)
                    self._record_drop(operation[2].get("run_id"), "persistent_writer_failure")
                for operation in still_failed[available:]:
                    self._mark_global_incomplete("failed_buffer_full")
                    self._record_drop(operation[2].get("run_id"), "failed_buffer_full")

    def _persist_diagnostics(self) -> None:
        with self._queue_cond:
            dropped = self._dropped
            self._dropped = {}
            global_diagnostics = self._global_diagnostics
            self._global_diagnostics = {}
        if not dropped and not global_diagnostics:
            return
        with self._connection() as connection:
            now = time.time()
            for (run_id, kind), count in dropped.items():
                connection.execute(
                    "INSERT INTO usage_diagnostics(run_id, diagnostic_type, count, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                    (run_id, "dropped_event", count, kind, now),
                )
            for reason, count in global_diagnostics.items():
                connection.execute(
                    "INSERT INTO usage_diagnostics(run_id, diagnostic_type, count, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                    (None, "usage_incomplete_global", count, reason, now),
                )

    def finalize_run(self, *, run_id: str, outcome: str | None = None,
                     failure_reason: str | None = None, ended_at: float | None = None,
                     elapsed: float | None = None) -> bool:
        """Drain all pending accounting, then synchronously finish the run."""
        drained = self.flush(timeout=5.0)
        self._replay_failed_sync()
        self._replay_failed_sync()
        with self._queue_cond:
            complete = (
                drained and not self._queue and not self._writer_busy
                and not self._failed_operations
                and not self._global_incomplete
                and run_id not in self._incomplete_runs
            )
        if not complete:
            self._record_drop(run_id, "incomplete_finalization")
            try:
                self._persist_diagnostics()
            except Exception:
                pass
            return False
        try:
            self._persist_diagnostics()
            self._finish_run(run_id=run_id, outcome=outcome, failure_reason=failure_reason,
                             ended_at=ended_at, elapsed=elapsed)
            return True
        except Exception:
            # Lifecycle observers are non-fatal. Leave a diagnostic attempt for
            # shutdown and let the caller continue without a deadlock.
            self._record_drop(run_id, "finalize_error")
            logger.warning("usage ledger finalization failed", exc_info=True)
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._queue_cond:
            if self._closed:
                return
            self._writer_stop = True
            self._queue_cond.notify_all()
            writer = self._writer
        if writer is not None and writer.is_alive():
            writer.join(timeout=max(0.0, timeout))
        self.flush(timeout=timeout)
        self._replay_failed_sync()
        self._replay_failed_sync()
        try:
            self._persist_diagnostics()
        except Exception:
            logger.debug("usage ledger diagnostic persistence failed", exc_info=True)
        with self._queue_cond:
            self._closed = True

    @staticmethod
    def _ensure_run(connection: sqlite3.Connection, *, run_id: str, process_id: str,
                    task_run_id: int | None = None, session_id: str | None = None,
                    task_id: str | None = None, board: str | None = None,
                    model: str | None = None, provider: str | None = None,
                    started_at: float | None = None) -> None:
        now = time.time() if started_at is None else float(started_at)
        connection.execute(
            """INSERT INTO usage_runs(
                run_id, process_id, task_run_id, session_id, task_id, board, model, provider,
                started_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                task_run_id = COALESCE(usage_runs.task_run_id, excluded.task_run_id),
                session_id = COALESCE(usage_runs.session_id, excluded.session_id),
                task_id = COALESCE(usage_runs.task_id, excluded.task_id),
                board = COALESCE(usage_runs.board, excluded.board),
                model = CASE WHEN usage_runs.model IS NULL THEN excluded.model ELSE usage_runs.model END,
                provider = CASE WHEN usage_runs.provider IS NULL OR usage_runs.provider = 'unknown' THEN excluded.provider ELSE usage_runs.provider END,
                updated_at = excluded.updated_at""",
            (run_id, str(process_id), task_run_id, _nonempty(session_id), _nonempty(task_id),
             _nonempty(board), _nonempty(model), _provider_key(provider), now, time.time()),
        )

    def _start_run(self, *, run_id: str, process_id: str, task_run_id: int | None = None,
                   session_id: str | None = None, task_id: str | None = None,
                   board: str | None = None, model: str | None = None,
                   provider: str | None = None, started_at: float | None = None) -> None:
        with self._connection() as connection:
            self._ensure_run(connection, run_id=run_id, process_id=process_id, task_run_id=task_run_id,
                              session_id=session_id, task_id=task_id, board=board, model=model,
                              provider=provider, started_at=started_at)

    def start_run(self, **kwargs: Any) -> None:
        self._start_run(**kwargs)

    def queue_start_run(self, **kwargs: Any) -> bool:
        return self._enqueue(self._start_run, **kwargs)

    def _record_event(self, *, run_id: str, event_id: str, event_type: str,
                      session_id: str | None, turn_id: str | None = None,
                      input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
                      retry_count: int = 0, model: str | None = None, provider: str | None = None,
                      tool_delta: int = 0, turn_delta: int = 0, process_id: str | None = None,
                      task_run_id: int | None = None, task_id: str | None = None,
                      board: str | None = None) -> bool:
        if not _nonempty(run_id) or not _nonempty(event_id):
            raise ValueError("run_id and event_id are required")
        input_tokens = _nonnegative_int(input_tokens, "input_tokens")
        output_tokens = _nonnegative_int(output_tokens, "output_tokens")
        cost_usd = _nonnegative_float(cost_usd, "cost_usd")
        retry_count = _nonnegative_int(retry_count, "retry_count")
        now = time.time()
        with self._connection() as connection:
            self._ensure_run(connection, run_id=run_id, process_id=process_id or _PROCESS_ID,
                              task_run_id=task_run_id, session_id=session_id, task_id=task_id,
                              board=board, model=model, provider=provider)
            inserted = connection.execute(
                """INSERT OR IGNORE INTO usage_events(
                    event_id, run_id, event_type, session_id, turn_id, input_tokens,
                    output_tokens, cost_usd, retry_count, model, provider, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, run_id, event_type, _nonempty(session_id), _nonempty(turn_id),
                 input_tokens, output_tokens, cost_usd, retry_count, _nonempty(model),
                 _nonempty(provider), now),
            ).rowcount
            if not inserted:
                return False
            actual_turn_delta = 0
            if turn_id:
                actual_turn_delta = connection.execute(
                    "INSERT OR IGNORE INTO usage_turns(run_id, turn_id) VALUES (?, ?)",
                    (run_id, turn_id),
                ).rowcount
            connection.execute(
                """UPDATE usage_runs SET input_tokens=input_tokens+?, output_tokens=output_tokens+?,
                    cost_usd=ROUND(cost_usd+?, 12), turn_count=turn_count+?, tool_call_count=tool_call_count+?,
                    retry_count=retry_count+?, updated_at=? WHERE run_id=?""",
                (input_tokens, output_tokens, round(cost_usd, 12), max(int(turn_delta or 0), actual_turn_delta),
                 _nonnegative_int(tool_delta, "tool_delta"), retry_count, now, run_id),
            )
            if _nonempty(model):
                connection.execute(
                    """INSERT INTO usage_run_models(run_id, model, provider, input_tokens, output_tokens, cost_usd, event_count)
                       VALUES (?, ?, ?, ?, ?, ?, 1)
                       ON CONFLICT(run_id, model, provider) DO UPDATE SET
                         input_tokens=input_tokens+excluded.input_tokens,
                         output_tokens=output_tokens+excluded.output_tokens,
                         cost_usd=ROUND(cost_usd+excluded.cost_usd, 12),
                         event_count=event_count+1""",
                    (run_id, _nonempty(model), _provider_key(provider), input_tokens, output_tokens, cost_usd),
                )
                models = connection.execute(
                    "SELECT DISTINCT model FROM usage_run_models WHERE run_id=? ORDER BY model", (run_id,)
                ).fetchall()
                providers = connection.execute(
                    "SELECT DISTINCT provider FROM usage_run_models WHERE run_id=? AND provider IS NOT NULL ORDER BY provider", (run_id,)
                ).fetchall()
                connection.execute(
                    "UPDATE usage_runs SET model=?, provider=? WHERE run_id=?",
                    (models[0][0] if len(models) == 1 else "mixed",
                     providers[0][0] if len(providers) == 1 else ("mixed" if providers else None), run_id),
                )
            return True

    def record_model_usage(self, **kwargs: Any) -> bool:
        kwargs["event_type"] = "model"
        return self._record_event(**kwargs)

    def queue_model_usage(self, **kwargs: Any) -> bool:
        kwargs["event_type"] = "model"
        return self._enqueue(self._record_event, critical=True, **kwargs)

    def record_tool_call(self, **kwargs: Any) -> bool:
        kwargs.update(event_type="tool", tool_delta=1)
        return self._record_event(**kwargs)

    def queue_tool_call(self, **kwargs: Any) -> bool:
        kwargs.update(event_type="tool", tool_delta=1)
        return self._enqueue(self._record_event, critical=True, **kwargs)

    def record_retry(self, **kwargs: Any) -> bool:
        kwargs.update(event_type="retry", retry_count=1)
        return self._record_event(**kwargs)

    def queue_retry(self, **kwargs: Any) -> bool:
        kwargs.update(event_type="retry", retry_count=1)
        return self._enqueue(self._record_event, critical=True, **kwargs)

    def _finish_run(self, *, run_id: str, outcome: str | None = None,
                    failure_reason: str | None = None, ended_at: float | None = None,
                    elapsed: float | None = None) -> None:
        ended = time.time() if ended_at is None else float(ended_at)
        with self._connection() as connection:
            row = connection.execute("SELECT started_at FROM usage_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                self._ensure_run(connection, run_id=run_id, process_id=_PROCESS_ID, started_at=ended)
                started = ended
            else:
                started = float(row["started_at"])
            effective_elapsed = elapsed if elapsed is not None else max(0.0, ended - started)
            if effective_elapsed is not None and float(effective_elapsed) < 0:
                raise ValueError("elapsed must be nonnegative")
            connection.execute(
                """UPDATE usage_runs SET ended_at=COALESCE(ended_at,?), elapsed=COALESCE(elapsed,?),
                    outcome=COALESCE(outcome,?), failure_reason=COALESCE(failure_reason,?), updated_at=?
                    WHERE run_id=?""",
                (ended, effective_elapsed, _nonempty(outcome), _nonempty(failure_reason), time.time(), run_id),
            )

    def finish_run(self, **kwargs: Any) -> None:
        self._finish_run(**kwargs)

    def queue_finish_run(self, **kwargs: Any) -> bool:
        return self._enqueue(self._finish_run, critical=True, **kwargs)

    def _model_breakdown(self, connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT model, provider, input_tokens, output_tokens, cost_usd, event_count "
            "FROM usage_run_models WHERE run_id=? ORDER BY model, provider", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        self.flush()
        with self._connection() as connection:
            row = connection.execute("SELECT *, elapsed AS elapsed_seconds FROM usage_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            result = dict(row)
            result["model_breakdown"] = self._model_breakdown(connection, run_id)
        return result

    def report(self, *, board: str | None = None, task_id: str | None = None,
               task_run_id: int | None = None, run_id: str | None = None,
               session_id: str | None = None, process_id: str | None = None,
               source_profile: str = "default", include_unassigned: bool = False) -> list[dict[str, Any]]:
        self.flush()
        clauses: list[str] = []
        params: list[Any] = []
        if board is not None:
            clauses.append("(board = ? OR (? = 1 AND board IS NULL))")
            params.extend([board, 1 if include_unassigned else 0])
        elif not include_unassigned:
            clauses.append("board IS NULL")
        for field, value in (("task_id", task_id), ("task_run_id", task_run_id),
                             ("run_id", run_id), ("session_id", session_id), ("process_id", process_id)):
            if value is not None:
                clauses.append(f"{field} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT *, elapsed AS elapsed_seconds FROM usage_runs {where} ORDER BY started_at ASC, run_id ASC",
                params,
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["source_profile"] = source_profile
                item["model_breakdown"] = self._model_breakdown(connection, item["run_id"])
                result.append(item)
        return result

    def link_kanban_run(self, *, task_run_id: int, usage_run_id: str | None = None,
                        kanban_db: str | Path | None = None, source_profile: str | None = None,
                        board: str | None = None) -> bool:
        """Add an exact finalized receipt projection keyed by ``task_runs.id``."""
        self.flush()
        usage_run_id = usage_run_id or process_run_id()
        try:
            authoritative_id = current_task_run_id()
            if authoritative_id is not None and authoritative_id != int(task_run_id):
                return False
            row = self.get_run(usage_run_id)
            if row.get("task_run_id") not in (None, int(task_run_id)):
                return False
            raw_db_path = str(kanban_db or os.environ.get("HERMES_KANBAN_DB", "")).strip()
            if not raw_db_path:
                return False
            from hermes_cli import kanban_db as kb
            resolved_source = source_profile or current_source_profile()
            with kb.connect_closing(Path(raw_db_path)) as connection:
                task_row = connection.execute(
                    "SELECT task_id, profile FROM task_runs WHERE id=?", (int(task_run_id),)
                ).fetchone()
                if task_row is None:
                    return False
                if row.get("task_id") and row["task_id"] != task_row["task_id"]:
                    return False
                if task_row["profile"] and task_row["profile"] != resolved_source:
                    return False
                if board is not None and row.get("board") not in (None, board):
                    return False
                connection.execute(
                    """INSERT INTO task_run_usage(
                        task_run_id, usage_run_id, source_profile, input_tokens, output_tokens,
                        cost_usd, model, provider, outcome, retry_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_run_id) DO UPDATE SET
                        usage_run_id=excluded.usage_run_id, source_profile=excluded.source_profile,
                        input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,
                        cost_usd=excluded.cost_usd, model=excluded.model, provider=excluded.provider,
                        outcome=excluded.outcome, retry_count=excluded.retry_count,
                        updated_at=excluded.updated_at""",
                    (int(task_run_id), usage_run_id, resolved_source, row["input_tokens"], row["output_tokens"],
                     row["cost_usd"], row["model"], row["provider"], row["outcome"], row["retry_count"], time.time()),
                )
            return True
        except Exception:
            logger.warning("Kanban usage link failed; runtime remains successful", exc_info=True)
            return False


def process_ledger() -> UsageLedger:
    path = str(default_ledger_path().resolve())
    with _LEDGER_CACHE_LOCK:
        ledger = _LEDGER_CACHE.get(path)
        if ledger is None:
            ledger = UsageLedger(path)
            _LEDGER_CACHE[path] = ledger
        return ledger


def reset_process_ledger_cache() -> None:
    with _LEDGER_CACHE_LOCK:
        ledgers = list(_LEDGER_CACHE.values())
        _LEDGER_CACHE.clear()
    for ledger in ledgers:
        ledger.shutdown()


atexit.register(reset_process_ledger_cache)
