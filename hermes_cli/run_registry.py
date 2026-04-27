"""
Persistent run registry for HermesWeb chat executions.

Stores each POST /api/chat run with full lifecycle status (queued → running →
completed | failed | timeout) in a SQLite database so GET /api/chat/runs/:runId
can always return the current state, even after WebSocket drops, page reloads,
or HTTP timeouts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

# Statuses
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"

TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_TIMEOUT}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_runs (
    run_id        TEXT PRIMARY KEY,
    session_id    TEXT,
    status        TEXT NOT NULL DEFAULT 'queued',
    user_message  TEXT NOT NULL,
    messages      TEXT,
    error         TEXT,
    started_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    completed_at  REAL,
    provider      TEXT,
    model         TEXT
);
CREATE INDEX IF NOT EXISTS chat_runs_status ON chat_runs (status);
CREATE INDEX IF NOT EXISTS chat_runs_updated ON chat_runs (updated_at);

CREATE TABLE IF NOT EXISTS run_steps (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    type       TEXT NOT NULL DEFAULT 'log',
    title      TEXT NOT NULL DEFAULT '',
    content    TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS run_steps_run_id ON run_steps (run_id, created_at);
"""


def _utc_now() -> float:
    return time.time()


def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class RunRegistry:
    """Thread-safe SQLite-backed run registry.

    Safe to call from both sync (thread) and async (event loop) contexts
    because all DB ops use a threading.Lock.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home

                db_path = get_hermes_home() / "chat_runs.db"
            except Exception:
                db_path = Path.home() / ".hermes" / "chat_runs.db"

        self._path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(
                    str(self._path),
                    check_same_thread=False,
                    isolation_level=None,  # autocommit
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.row_factory = sqlite3.Row
            except Exception as exc:
                _log.warning(
                    "[run-registry] could not open %s, using :memory: (%s)", self._path, exc
                )
                self._conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
                self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA)
            # Safe migration: add columns if they don't exist yet
            for col, typedef in [("provider", "TEXT"), ("model", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE chat_runs ADD COLUMN {col} {typedef}")
                except Exception:
                    pass  # column already exists

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._get_conn().execute(sql, params)

    # ------------------------------------------------------------------
    # Public CRUD
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        session_id: Optional[str],
        user_message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        self._execute(
            """
            INSERT OR IGNORE INTO chat_runs
                (run_id, session_id, status, user_message, started_at, updated_at, provider, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, session_id, STATUS_QUEUED, user_message, now, now, provider, model),
        )
        _log.info("[run-registry] created run_id=%s session_id=%s", run_id, session_id)

    def start_run(self, run_id: str) -> None:
        now = _utc_now()
        self._execute(
            "UPDATE chat_runs SET status=?, updated_at=? WHERE run_id=? AND status=?",
            (STATUS_RUNNING, now, run_id, STATUS_QUEUED),
        )
        _log.info("[run-registry] started run_id=%s", run_id)

    def complete_run(
        self,
        run_id: str,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        messages_json = json.dumps(messages or [])
        with self._lock:
            conn = self._get_conn()
            # Only complete if not already in a terminal state
            row = conn.execute(
                "SELECT status FROM chat_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row and row["status"] in TERMINAL_STATUSES:
                _log.debug("[run-registry] complete_run skipped (already terminal): %s", run_id)
                return
            conn.execute(
                """
                UPDATE chat_runs
                SET status=?, messages=?, updated_at=?, completed_at=?,
                    session_id=COALESCE(?, session_id),
                    provider=COALESCE(?, provider),
                    model=COALESCE(?, model)
                WHERE run_id=?
                """,
                (STATUS_COMPLETED, messages_json, now, now, session_id, provider, model, run_id),
            )
        _log.info("[run-registry] completed run_id=%s msgs=%d", run_id, len(messages or []))

    def fail_run(self, run_id: str, error: str) -> None:
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT status FROM chat_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row and row["status"] in TERMINAL_STATUSES:
                return
            conn.execute(
                "UPDATE chat_runs SET status=?, error=?, updated_at=?, completed_at=? WHERE run_id=?",
                (STATUS_FAILED, error, now, now, run_id),
            )
        _log.warning("[run-registry] failed run_id=%s error=%r", run_id, error)

    def timeout_run(self, run_id: str) -> None:
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT status FROM chat_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row and row["status"] in TERMINAL_STATUSES:
                return
            conn.execute(
                "UPDATE chat_runs SET status=?, error=?, updated_at=?, completed_at=? WHERE run_id=?",
                (STATUS_TIMEOUT, "Execution exceeded maximum allowed time", now, now, run_id),
            )
        _log.warning("[run-registry] timed out run_id=%s", run_id)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM chat_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_stale_running_runs(self, max_age_seconds: float = 600.0) -> List[str]:
        """Return run_ids that are still 'running' but started > max_age_seconds ago."""
        cutoff = _utc_now() - max_age_seconds
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT run_id FROM chat_runs WHERE status=? AND started_at < ?",
                (STATUS_RUNNING, cutoff),
            ).fetchall()
        return [r["run_id"] for r in rows]

    def delete_old_runs(self, max_age_seconds: float = 7 * 86400) -> int:
        """Delete terminal runs older than max_age_seconds. Returns deleted count."""
        cutoff = _utc_now() - max_age_seconds
        with self._lock:
            cur = self._get_conn().execute(
                "DELETE FROM chat_runs WHERE status IN (?,?,?) AND completed_at < ?",
                (STATUS_COMPLETED, STATUS_FAILED, STATUS_TIMEOUT, cutoff),
            )
        count = cur.rowcount
        if count:
            _log.info("[run-registry] deleted %d old terminal runs", count)
        return count

    # ------------------------------------------------------------------
    # Run steps
    # ------------------------------------------------------------------

    def create_step(
        self,
        *,
        run_id: str,
        step_id: str,
        type_: str = "log",
        title: str = "",
        content: Optional[str] = None,
        status: str = "running",
    ) -> None:
        now = _utc_now()
        self._execute(
            """
            INSERT OR IGNORE INTO run_steps
                (id, run_id, type, title, content, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (step_id, run_id, type_, title, content, status, now, now),
        )

    def update_step(
        self,
        *,
        step_id: str,
        status: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        if content is not None and title is not None:
            self._execute(
                "UPDATE run_steps SET status=?, content=?, title=?, updated_at=? WHERE id=?",
                (status, content, title, now, step_id),
            )
        elif content is not None:
            self._execute(
                "UPDATE run_steps SET status=?, content=?, updated_at=? WHERE id=?",
                (status, content, now, step_id),
            )
        elif title is not None:
            self._execute(
                "UPDATE run_steps SET status=?, title=?, updated_at=? WHERE id=?",
                (status, title, now, step_id),
            )
        else:
            self._execute(
                "UPDATE run_steps SET status=?, updated_at=? WHERE id=?",
                (status, now, step_id),
            )

    def complete_step(self, *, step_id: str, content: Optional[str] = None) -> None:
        self.update_step(step_id=step_id, status="completed", content=content)

    def fail_step(self, *, step_id: str, error: str) -> None:
        self.update_step(step_id=step_id, status="failed", content=error)

    def get_steps(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT * FROM run_steps WHERE run_id=? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        return [self._step_row_to_dict(r) for r in rows]

    @staticmethod
    def _step_row_to_dict(row: "sqlite3.Row") -> Dict[str, Any]:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "type": row["type"],
            "title": row["title"],
            "content": row["content"],
            "status": row["status"],
            "created_at": _ts_to_iso(row["created_at"]),
            "updated_at": _ts_to_iso(row["updated_at"]),
        }

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        raw_messages = row["messages"]
        if raw_messages:
            try:
                messages = json.loads(raw_messages)
            except Exception:
                messages = []

        return {
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "status": row["status"],
            "messages": messages,
            "error": row["error"],
            "started_at": _ts_to_iso(row["started_at"]),
            "updated_at": _ts_to_iso(row["updated_at"]),
            "completed_at": _ts_to_iso(row["completed_at"]),
            "provider": row["provider"],
            "model": row["model"],
        }
