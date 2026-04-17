"""Durable background job storage, launcher helpers, and worker utilities."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_hermes_home
from gateway.session import SessionSource


logger = logging.getLogger(__name__)

BACKGROUND_JOBS_DB_FILENAME = "background_jobs.db"
JOB_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
JOB_ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})
DELIVERABLE_JOB_STATUSES = frozenset({"completed", "failed"})


def _db_path() -> Path:
    return get_hermes_home() / BACKGROUND_JOBS_DB_FILENAME


def _json_dumps(value: Any, *, default: Any) -> str:
    payload = default if value is None else value
    return json.dumps(payload, ensure_ascii=False)


def _json_loads(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _job_query_status(status: str, pending_approval_count: int) -> tuple[str, str]:
    normalized = str(status or "queued").strip().lower() or "queued"
    approvals = max(int(pending_approval_count or 0), 0)
    if approvals > 0:
        return "approval_pending", "待授权"
    if normalized == "running":
        return "running", "进行中"
    if normalized == "queued":
        return "queued", "排队中"
    if normalized == "cancelling":
        return "cancelling", "取消中"
    if normalized == "completed":
        return "completed", "已完成"
    if normalized == "failed":
        return "failed", "失败"
    if normalized == "cancelled":
        return "cancelled", "已取消"
    return normalized, normalized


def background_job_chat_key(source: SessionSource) -> str:
    """Return a stable chat-scoped key for durable background jobs."""
    platform = source.platform.value if getattr(source, "platform", None) else "unknown"
    thread_id = getattr(source, "thread_id", None) or "-"
    return f"{platform}:{source.chat_type}:{source.chat_id}:{thread_id}"


def background_job_scope_key(source: SessionSource, *, session_key: str = "") -> str:
    """Return the session-scoped key used to isolate background jobs."""
    resolved = str(session_key or "").strip()
    return resolved or background_job_chat_key(source)


def _default_job_preview(prompt: str) -> str:
    prompt = str(prompt or "")
    return prompt[:60] + ("..." if len(prompt) > 60 else "")


def _systemd_unit_failure_summary(unit: str, scope: str = "") -> str:
    """Best-effort extract of recent unit failure details from journalctl."""
    normalized_unit = str(unit or "").strip()
    if not normalized_unit or not shutil.which("journalctl"):
        return ""

    args = ["journalctl"]
    if str(scope or "").strip().lower() == "user":
        args.append("--user")
    args.extend(["-u", normalized_unit, "-n", "40", "--no-pager", "-o", "cat"])

    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        logger.debug("Failed reading journal for background unit %s", normalized_unit, exc_info=True)
        return ""

    output = str(proc.stdout or "").strip()
    if not output:
        return ""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""

    keywords = (
        "traceback",
        "error",
        "exception",
        "failed",
        "importerror",
        "modulenotfounderror",
        "nameerror",
        "syntaxerror",
        "runtimeerror",
    )
    interesting: list[str] = []
    for line in reversed(lines):
        if any(keyword in line.lower() for keyword in keywords):
            interesting.append(line)
        if len(interesting) >= 4:
            break

    if interesting:
        interesting.reverse()
        return " | ".join(interesting)
    return lines[-1]


class BackgroundJobStore:
    """SQLite-backed durable store for gateway background jobs."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or _db_path())
        self._init_lock = threading.Lock()
        self._initialized = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS background_jobs (
                        task_id TEXT PRIMARY KEY,
                        chat_key TEXT NOT NULL,
                        scope_key TEXT NOT NULL,
                        session_key TEXT,
                        source_json TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        preview TEXT NOT NULL,
                        kind TEXT NOT NULL DEFAULT 'manual',
                        worker_name TEXT NOT NULL DEFAULT '',
                        preloaded_skills_json TEXT NOT NULL DEFAULT '[]',
                        loaded_skills_json TEXT NOT NULL DEFAULT '[]',
                        missing_skills_json TEXT NOT NULL DEFAULT '[]',
                        conversation_history_json TEXT NOT NULL DEFAULT '[]',
                        context_prompt TEXT NOT NULL DEFAULT '',
                        admin_user_ids_json TEXT NOT NULL DEFAULT '[]',
                        is_admin_user INTEGER,
                        status TEXT NOT NULL DEFAULT 'queued',
                        raw_response TEXT,
                        error TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        started_at REAL,
                        finished_at REAL,
                        last_heartbeat_at REAL,
                        heartbeat_count INTEGER NOT NULL DEFAULT 0,
                        recovered_at REAL,
                        launcher_type TEXT,
                        launcher_pid INTEGER,
                        launcher_unit TEXT,
                        launcher_scope TEXT,
                        launcher_meta_json TEXT NOT NULL DEFAULT '{}',
                        delivery_status TEXT NOT NULL DEFAULT 'pending',
                        delivery_attempts INTEGER NOT NULL DEFAULT 0,
                        delivery_claimed_by TEXT,
                        delivery_claimed_at REAL,
                        delivered_at REAL,
                        delivery_error TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_background_jobs_chat_scope
                    ON background_jobs(chat_key, scope_key, created_at);

                    CREATE INDEX IF NOT EXISTS idx_background_jobs_delivery
                    ON background_jobs(status, delivery_status, delivered_at, delivery_claimed_at);

                    CREATE TABLE IF NOT EXISTS background_approval_requests (
                        request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        session_key TEXT NOT NULL,
                        chat_key TEXT NOT NULL,
                        source_json TEXT NOT NULL,
                        command TEXT NOT NULL,
                        description TEXT NOT NULL,
                        prompt_title TEXT NOT NULL,
                        approver_name TEXT NOT NULL,
                        allow_persistence INTEGER NOT NULL DEFAULT 1,
                        pattern_key TEXT,
                        pattern_keys_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'pending',
                        choice TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        notification_claimed_by TEXT,
                        notification_claimed_at REAL,
                        notified_at REAL,
                        resolved_at REAL
                    );

                    CREATE INDEX IF NOT EXISTS idx_background_approval_session
                    ON background_approval_requests(session_key, status, created_at);

                    CREATE INDEX IF NOT EXISTS idx_background_approval_notify
                    ON background_approval_requests(status, notification_claimed_at, created_at);
                    """
                )
                self._ensure_columns(
                    conn,
                    "background_jobs",
                    {
                        "last_heartbeat_at": "REAL",
                        "heartbeat_count": "INTEGER NOT NULL DEFAULT 0",
                        "recovered_at": "REAL",
                    },
                )
                conn.commit()
            finally:
                conn.close()
            self._initialized = True

    @staticmethod
    def _ensure_columns(
        conn: sqlite3.Connection,
        table_name: str,
        columns: dict[str, str],
    ) -> None:
        existing = {
            str(row[1]).strip()
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            if row and len(row) > 1
        }
        for column_name, column_ddl in columns.items():
            if column_name in existing:
                continue
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_ddl}"
            )

    def create_job(
        self,
        *,
        task_id: str,
        prompt: str,
        source: SessionSource,
        session_key: str = "",
        job_kind: str = "manual",
        worker_name: str = "",
        preloaded_skills: list[str] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        context_prompt: str = "",
        admin_user_ids: list[str] | None = None,
        is_admin_user: Optional[bool] = None,
    ) -> dict[str, Any]:
        now = time.time()
        chat_key = background_job_chat_key(source)
        scope_key = background_job_scope_key(source, session_key=session_key)
        preview = _default_job_preview(prompt)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO background_jobs (
                    task_id, chat_key, scope_key, session_key, source_json,
                    prompt, preview, kind, worker_name,
                    preloaded_skills_json, loaded_skills_json, missing_skills_json,
                    conversation_history_json, context_prompt,
                    admin_user_ids_json, is_admin_user,
                    status, created_at, updated_at, delivery_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    chat_key,
                    scope_key,
                    session_key or None,
                    _json_dumps(source.to_dict(), default={}),
                    str(prompt or ""),
                    preview,
                    str(job_kind or "manual"),
                    str(worker_name or ""),
                    _json_dumps(list(preloaded_skills or []), default=[]),
                    _json_dumps([], default=[]),
                    _json_dumps([], default=[]),
                    _json_dumps(list(conversation_history or []), default=[]),
                    str(context_prompt or ""),
                    _json_dumps(list(admin_user_ids or []), default=[]),
                    None if is_admin_user is None else (1 if is_admin_user else 0),
                    "queued",
                    now,
                    now,
                    "pending",
                ),
            )
            conn.commit()
        return self.get_job(task_id) or {}

    def _row_to_job(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "task_id": row["task_id"],
            "chat_key": row["chat_key"],
            "scope_key": row["scope_key"],
            "session_key": row["session_key"] or "",
            "source": _json_loads(row["source_json"], default={}),
            "prompt": row["prompt"] or "",
            "preview": row["preview"] or "",
            "kind": row["kind"] or "manual",
            "worker_name": row["worker_name"] or "",
            "preloaded_skills": _json_loads(row["preloaded_skills_json"], default=[]),
            "loaded_skills": _json_loads(row["loaded_skills_json"], default=[]),
            "missing_skills": _json_loads(row["missing_skills_json"], default=[]),
            "conversation_history": _json_loads(row["conversation_history_json"], default=[]),
            "context_prompt": row["context_prompt"] or "",
            "admin_user_ids": _json_loads(row["admin_user_ids_json"], default=[]),
            "is_admin_user": (
                None if row["is_admin_user"] is None else bool(row["is_admin_user"])
            ),
            "status": row["status"] or "queued",
            "raw_response": row["raw_response"] or "",
            "error": row["error"] or None,
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
            "started_at": None if row["started_at"] is None else float(row["started_at"]),
            "finished_at": None if row["finished_at"] is None else float(row["finished_at"]),
            "last_heartbeat_at": (
                None
                if row["last_heartbeat_at"] is None
                else float(row["last_heartbeat_at"])
            ),
            "heartbeat_count": int(row["heartbeat_count"] or 0),
            "recovered_at": None if row["recovered_at"] is None else float(row["recovered_at"]),
            "launcher_type": row["launcher_type"] or "",
            "launcher_pid": row["launcher_pid"],
            "launcher_unit": row["launcher_unit"] or "",
            "launcher_scope": row["launcher_scope"] or "",
            "launcher_meta": _json_loads(row["launcher_meta_json"], default={}),
            "delivery_status": row["delivery_status"] or "pending",
            "delivery_attempts": int(row["delivery_attempts"] or 0),
            "delivery_claimed_by": row["delivery_claimed_by"] or "",
            "delivery_claimed_at": (
                None if row["delivery_claimed_at"] is None else float(row["delivery_claimed_at"])
            ),
            "delivered_at": None if row["delivered_at"] is None else float(row["delivered_at"]),
            "delivery_error": row["delivery_error"] or None,
        }

    @staticmethod
    def _pending_approval_counts(
        conn: sqlite3.Connection,
        task_ids: list[str],
    ) -> dict[str, int]:
        normalized = [str(task_id or "").strip() for task_id in task_ids if str(task_id or "").strip()]
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        rows = conn.execute(
            f"""
            SELECT task_id, COUNT(*) AS total
            FROM background_approval_requests
            WHERE task_id IN ({placeholders})
              AND status IN ('pending', 'notified')
            GROUP BY task_id
            """,
            tuple(normalized),
        ).fetchall()
        return {
            str(row["task_id"]): int((row["total"] if row else 0) or 0)
            for row in rows
        }

    @staticmethod
    def _enrich_job_query_fields(
        job: dict[str, Any] | None,
        *,
        pending_approval_count: int = 0,
    ) -> dict[str, Any] | None:
        if job is None:
            return None
        enriched = dict(job)
        approvals = max(int(pending_approval_count or 0), 0)
        query_status, query_status_text = _job_query_status(
            enriched.get("status") or "queued",
            approvals,
        )
        enriched["pending_approval_count"] = approvals
        enriched["query_status"] = query_status
        enriched["query_status_text"] = query_status_text
        enriched["is_queryable_active"] = query_status in {
            "approval_pending",
            "queued",
            "running",
            "cancelling",
        }
        return enriched

    def get_job(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM background_jobs WHERE task_id = ?",
                (str(task_id or "").strip(),),
            ).fetchone()
            job = self._row_to_job(row)
            if not job:
                return None
            pending_counts = self._pending_approval_counts(conn, [job["task_id"]])
            return self._enrich_job_query_fields(
                job,
                pending_approval_count=pending_counts.get(job["task_id"], 0),
            )

    def list_jobs(
        self,
        *,
        chat_key: str | None = None,
        scope_key: str | None = None,
        active_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized_chat_key = str(chat_key or "").strip()
        normalized_scope_key = str(scope_key or "").strip()
        if bool(normalized_chat_key) != bool(normalized_scope_key):
            raise ValueError("chat_key and scope_key must be provided together")

        params: list[Any] = []
        where_clauses: list[str] = []
        if normalized_chat_key and normalized_scope_key:
            where_clauses.extend(["chat_key = ?", "scope_key = ?"])
            params.extend([normalized_chat_key, normalized_scope_key])
        if active_only:
            where_clauses.append("status IN ('queued', 'running', 'cancelling')")

        sql = ["SELECT * FROM background_jobs"]
        if where_clauses:
            sql.append("WHERE " + " AND ".join(where_clauses))
        sql.append("ORDER BY created_at ASC")
        if isinstance(limit, int) and limit > 0:
            sql.append("LIMIT ?")
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), tuple(params)).fetchall()
            jobs = [self._row_to_job(row) for row in rows if row is not None]
            pending_counts = self._pending_approval_counts(
                conn,
                [str(job.get("task_id") or "") for job in jobs if job],
            )
            return [
                self._enrich_job_query_fields(
                    job,
                    pending_approval_count=pending_counts.get(str(job.get("task_id") or ""), 0),
                )
                for job in jobs
                if job is not None
            ]

    def mark_job_running(self, task_id: str, *, launcher_pid: int | None = None) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET status = CASE
                        WHEN status = 'queued' THEN 'running'
                        ELSE status
                    END,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?,
                    recovered_at = NULL,
                    launcher_pid = COALESCE(launcher_pid, ?)
                WHERE task_id = ?
                """,
                (now, now, launcher_pid, task_id),
            )
            conn.commit()
        return self.get_job(task_id)

    def touch_job_heartbeat(self, task_id: str, *, now_ts: float | None = None) -> dict[str, Any] | None:
        now = float(now_ts) if now_ts is not None else time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET last_heartbeat_at = ?,
                    heartbeat_count = heartbeat_count + 1,
                    updated_at = ?
                WHERE task_id = ?
                  AND status IN ('queued', 'running', 'cancelling')
                """,
                (now, now, task_id),
            )
            conn.commit()
        return self.get_job(task_id)

    def update_job_launcher(self, task_id: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
        meta = dict(metadata or {})
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET launcher_type = ?,
                    launcher_pid = ?,
                    launcher_unit = ?,
                    launcher_scope = ?,
                    launcher_meta_json = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    _normalize_optional_text(meta.get("launcher_type")),
                    meta.get("launcher_pid"),
                    _normalize_optional_text(meta.get("launcher_unit")),
                    _normalize_optional_text(meta.get("launcher_scope")),
                    _json_dumps(meta, default={}),
                    time.time(),
                    task_id,
                ),
            )
            conn.commit()
        return self.get_job(task_id)

    def update_job_skills(
        self,
        task_id: str,
        *,
        loaded_skills: list[str] | None = None,
        missing_skills: list[str] | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET loaded_skills_json = ?,
                    missing_skills_json = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    _json_dumps(list(loaded_skills or []), default=[]),
                    _json_dumps(list(missing_skills or []), default=[]),
                    time.time(),
                    task_id,
                ),
            )
            conn.commit()
        return self.get_job(task_id)

    def mark_job_completed(self, task_id: str, *, raw_response: str) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM background_jobs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"] or "").strip().lower()
            if status == "cancelled":
                return self.get_job(task_id)
            conn.execute(
                """
                UPDATE background_jobs
                SET status = 'completed',
                    raw_response = ?,
                    error = NULL,
                    finished_at = ?,
                    updated_at = ?,
                    delivery_status = CASE
                        WHEN delivery_status = 'delivered' THEN 'delivered'
                        ELSE 'pending'
                    END,
                    delivery_claimed_by = NULL,
                    delivery_claimed_at = NULL
                WHERE task_id = ?
                """,
                (str(raw_response or ""), now, now, task_id),
            )
            conn.commit()
        return self.get_job(task_id)

    def mark_job_failed(
        self,
        task_id: str,
        *,
        error: str,
        recovered_at: float | None = None,
    ) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM background_jobs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            status = str(row["status"] or "").strip().lower()
            if status == "cancelled":
                return self.get_job(task_id)
            conn.execute(
                """
                UPDATE background_jobs
                SET status = 'failed',
                    error = ?,
                    finished_at = ?,
                    updated_at = ?,
                    recovered_at = COALESCE(?, recovered_at),
                    delivery_status = CASE
                        WHEN delivery_status = 'delivered' THEN 'delivered'
                        ELSE 'pending'
                    END,
                    delivery_claimed_by = NULL,
                    delivery_claimed_at = NULL
                WHERE task_id = ?
                """,
                (
                    str(error or "background task failed"),
                    now,
                    now,
                    recovered_at,
                    task_id,
                ),
            )
            conn.commit()
        return self.get_job(task_id)

    def mark_job_cancelling(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET status = CASE
                        WHEN status IN ('queued', 'running') THEN 'cancelling'
                        ELSE status
                    END,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (time.time(), task_id),
            )
            conn.commit()
        return self.get_job(task_id)

    def mark_job_cancelled(self, task_id: str, *, reason: str = "cancelled") -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET status = 'cancelled',
                    error = ?,
                    finished_at = COALESCE(finished_at, ?),
                    updated_at = ?,
                    raw_response = CASE
                        WHEN status = 'cancelled' THEN raw_response
                        ELSE ''
                    END,
                    delivery_status = CASE
                        WHEN delivery_status = 'delivered' THEN 'delivered'
                        ELSE 'skipped'
                    END,
                    delivery_claimed_by = NULL,
                    delivery_claimed_at = NULL
                WHERE task_id = ?
                """,
                (str(reason or "cancelled"), now, now, task_id),
            )
            conn.commit()
        return self.get_job(task_id)

    def claim_delivery_jobs(
        self,
        *,
        claimer: str,
        limit: int = 20,
        lease_seconds: float = 60.0,
    ) -> list[dict[str, Any]]:
        now = time.time()
        expired_before = now - max(float(lease_seconds or 0.0), 0.0)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT task_id
                FROM background_jobs
                WHERE status IN ('completed', 'failed')
                  AND delivered_at IS NULL
                  AND delivery_status IN ('pending', 'delivering')
                  AND (
                    delivery_claimed_by IS NULL
                    OR delivery_claimed_at IS NULL
                    OR delivery_claimed_at < ?
                  )
                ORDER BY finished_at ASC, updated_at ASC
                LIMIT ?
                """,
                (expired_before, int(limit)),
            ).fetchall()
            task_ids = [row["task_id"] for row in rows]
            if task_ids:
                conn.executemany(
                    """
                    UPDATE background_jobs
                    SET delivery_status = 'delivering',
                        delivery_claimed_by = ?,
                        delivery_claimed_at = ?,
                        updated_at = ?
                    WHERE task_id = ?
                    """,
                    [(claimer, now, now, task_id) for task_id in task_ids],
                )
            conn.commit()
        return [self.get_job(task_id) for task_id in task_ids if self.get_job(task_id)]

    def release_delivery_claim(self, task_id: str, *, error: str = "") -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET delivery_status = 'pending',
                    delivery_attempts = delivery_attempts + 1,
                    delivery_claimed_by = NULL,
                    delivery_claimed_at = NULL,
                    delivery_error = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (_normalize_optional_text(error), time.time(), task_id),
            )
            conn.commit()
        return self.get_job(task_id)

    def mark_job_delivered(self, task_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET delivery_status = 'delivered',
                    delivered_at = ?,
                    delivery_claimed_by = NULL,
                    delivery_claimed_at = NULL,
                    delivery_error = NULL,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (now, now, task_id),
            )
            conn.commit()
        return self.get_job(task_id)

    def _row_to_approval_request(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "request_id": int(row["request_id"]),
            "task_id": row["task_id"],
            "session_key": row["session_key"],
            "chat_key": row["chat_key"],
            "source": _json_loads(row["source_json"], default={}),
            "command": row["command"] or "",
            "description": row["description"] or "",
            "prompt_title": row["prompt_title"] or "",
            "approver_name": row["approver_name"] or "",
            "allow_persistence": bool(row["allow_persistence"]),
            "pattern_key": row["pattern_key"] or "",
            "pattern_keys": _json_loads(row["pattern_keys_json"], default=[]),
            "status": row["status"] or "pending",
            "choice": row["choice"] or None,
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
            "notification_claimed_by": row["notification_claimed_by"] or "",
            "notification_claimed_at": (
                None if row["notification_claimed_at"] is None else float(row["notification_claimed_at"])
            ),
            "notified_at": None if row["notified_at"] is None else float(row["notified_at"]),
            "resolved_at": None if row["resolved_at"] is None else float(row["resolved_at"]),
        }

    def create_approval_request(
        self,
        *,
        task_id: str,
        session_key: str,
        source: SessionSource,
        approval_data: dict[str, Any],
    ) -> int:
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO background_approval_requests (
                    task_id, session_key, chat_key, source_json,
                    command, description, prompt_title, approver_name,
                    allow_persistence, pattern_key, pattern_keys_json,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    task_id,
                    session_key,
                    background_job_chat_key(source),
                    _json_dumps(source.to_dict(), default={}),
                    str(approval_data.get("command") or ""),
                    str(approval_data.get("description") or ""),
                    str(approval_data.get("prompt_title") or "Dangerous command requires approval"),
                    str(approval_data.get("approver_name") or "管理员"),
                    1 if bool(approval_data.get("allow_persistence", True)) else 0,
                    _normalize_optional_text(approval_data.get("pattern_key")),
                    _json_dumps(list(approval_data.get("pattern_keys") or []), default=[]),
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def claim_approval_notifications(
        self,
        *,
        claimer: str,
        limit: int = 20,
        lease_seconds: float = 60.0,
    ) -> list[dict[str, Any]]:
        now = time.time()
        expired_before = now - max(float(lease_seconds or 0.0), 0.0)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT request_id
                FROM background_approval_requests
                WHERE status = 'pending'
                  AND (
                    notification_claimed_by IS NULL
                    OR notification_claimed_at IS NULL
                    OR notification_claimed_at < ?
                  )
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (expired_before, int(limit)),
            ).fetchall()
            request_ids = [int(row["request_id"]) for row in rows]
            if request_ids:
                conn.executemany(
                    """
                    UPDATE background_approval_requests
                    SET notification_claimed_by = ?,
                        notification_claimed_at = ?,
                        updated_at = ?
                    WHERE request_id = ?
                    """,
                    [(claimer, now, now, request_id) for request_id in request_ids],
                )
            conn.commit()
        return [self.get_approval_request(request_id) for request_id in request_ids if self.get_approval_request(request_id)]

    def get_approval_request(self, request_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM background_approval_requests WHERE request_id = ?",
                (int(request_id),),
            ).fetchone()
        return self._row_to_approval_request(row)

    def mark_approval_notified(self, request_id: int) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_approval_requests
                SET status = 'notified',
                    notified_at = ?,
                    notification_claimed_by = NULL,
                    notification_claimed_at = NULL,
                    updated_at = ?
                WHERE request_id = ?
                """,
                (now, now, int(request_id)),
            )
            conn.commit()
        return self.get_approval_request(request_id)

    def release_approval_notification_claim(self, request_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE background_approval_requests
                SET notification_claimed_by = NULL,
                    notification_claimed_at = NULL,
                    updated_at = ?
                WHERE request_id = ? AND status = 'pending'
                """,
                (time.time(), int(request_id)),
            )
            conn.commit()
        return self.get_approval_request(request_id)

    def has_pending_approval_requests(self, session_key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM background_approval_requests
                WHERE session_key = ?
                  AND status IN ('pending', 'notified')
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (str(session_key or "").strip(),),
            ).fetchone()
        return row is not None

    def count_pending_approval_requests(self, session_key: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM background_approval_requests
                WHERE session_key = ?
                  AND status IN ('pending', 'notified')
                """,
                (str(session_key or "").strip(),),
            ).fetchone()
        return int((row["total"] if row else 0) or 0)

    def count_all_pending_approval_requests(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM background_approval_requests
                WHERE status IN ('pending', 'notified')
                """
            ).fetchone()
        return int((row["total"] if row else 0) or 0)

    def peek_pending_approval_request(self, session_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM background_approval_requests
                WHERE session_key = ?
                  AND status IN ('pending', 'notified')
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (str(session_key or "").strip(),),
            ).fetchone()
        return self._row_to_approval_request(row)

    def resolve_approval_requests(
        self,
        *,
        session_key: str,
        choice: str,
        resolve_all: bool = False,
    ) -> int:
        session_key = str(session_key or "").strip()
        if not session_key:
            return 0
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT request_id
                FROM background_approval_requests
                WHERE session_key = ?
                  AND status IN ('pending', 'notified')
                ORDER BY created_at ASC
                """,
                (session_key,),
            ).fetchall()
            if not rows:
                return 0
            request_ids = [int(row["request_id"]) for row in rows]
            if not resolve_all:
                request_ids = request_ids[:1]
            now = time.time()
            conn.executemany(
                """
                UPDATE background_approval_requests
                SET status = 'resolved',
                    choice = ?,
                    resolved_at = ?,
                    updated_at = ?,
                    notification_claimed_by = NULL,
                    notification_claimed_at = NULL
                WHERE request_id = ?
                """,
                [(choice, now, now, request_id) for request_id in request_ids],
            )
            conn.commit()
        return len(request_ids)

    def wait_for_approval_resolution(
        self,
        request_id: int,
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 0.5,
    ) -> str | None:
        deadline = time.monotonic() + max(float(timeout_seconds or 0.0), 0.0)
        interval = max(float(poll_interval_seconds or 0.0), 0.01)
        while time.monotonic() <= deadline:
            request = self.get_approval_request(int(request_id))
            if not request:
                return None
            if request["status"] == "resolved":
                return str(request.get("choice") or "").strip() or None
            time.sleep(interval)
        return None

    @staticmethod
    def _subprocess_pid_alive(pid: Any) -> bool:
        if pid in (None, ""):
            return False
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
        return True

    def recover_stale_jobs(
        self,
        *,
        now_ts: float | None = None,
        queued_grace_seconds: float = 120.0,
        heartbeat_stale_seconds: float = 120.0,
    ) -> list[dict[str, Any]]:
        now = float(now_ts) if now_ts is not None else time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM background_jobs
                WHERE status IN ('queued', 'running', 'cancelling')
                ORDER BY updated_at ASC, created_at ASC
                """
            ).fetchall()

        recovered: list[dict[str, Any]] = []
        for row in rows:
            job = self._row_to_job(row)
            if not job:
                continue
            task_id = str(job.get("task_id") or "").strip()
            status = str(job.get("status") or "").strip().lower()
            launcher_type = str(job.get("launcher_type") or "").strip().lower()
            launcher_pid = job.get("launcher_pid")

            stale_reason = ""
            if launcher_type == "subprocess" and launcher_pid not in (None, ""):
                if not self._subprocess_pid_alive(launcher_pid):
                    stale_reason = (
                        "background worker heartbeat stale or worker process exited unexpectedly"
                    )

            if not stale_reason and status == "queued":
                if launcher_type == "systemd-run":
                    summary = _systemd_unit_failure_summary(
                        str(job.get("launcher_unit") or ""),
                        str(job.get("launcher_scope") or ""),
                    )
                    if summary:
                        stale_reason = f"background worker failed before heartbeat: {summary}"
                queued_anchor = max(
                    float(job.get("updated_at") or 0.0),
                    float(job.get("created_at") or 0.0),
                )
                if (
                    not stale_reason
                    and queued_grace_seconds > 0
                    and now - queued_anchor > float(queued_grace_seconds)
                ):
                    stale_reason = "background worker stayed queued too long without starting"

            if not stale_reason and status in {"running", "cancelling"}:
                heartbeat_anchor = (
                    float(job.get("last_heartbeat_at") or 0.0)
                    or float(job.get("started_at") or 0.0)
                    or float(job.get("updated_at") or 0.0)
                    or float(job.get("created_at") or 0.0)
                )
                if (
                    heartbeat_stale_seconds > 0
                    and heartbeat_anchor > 0
                    and now - heartbeat_anchor > float(heartbeat_stale_seconds)
                ):
                    stale_reason = "background worker heartbeat stale"

            if not stale_reason or not task_id:
                continue

            recovered_job = self.mark_job_failed(
                task_id,
                error=stale_reason,
                recovered_at=now,
            )
            if recovered_job:
                recovered.append(recovered_job)
        return recovered


class ExternalApprovalBridge:
    """Cross-process approval bridge backed by ``BackgroundJobStore``."""

    def __init__(
        self,
        *,
        store: BackgroundJobStore,
        task_id: str,
        session_key: str,
        source: SessionSource,
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.session_key = session_key
        self.source = source

    def request_and_wait(self, approval_data: dict[str, Any], *, timeout_seconds: int) -> str | None:
        request_id = self.store.create_approval_request(
            task_id=self.task_id,
            session_key=self.session_key,
            source=self.source,
            approval_data=approval_data,
        )
        return self.store.wait_for_approval_resolution(
            request_id,
            timeout_seconds=float(timeout_seconds),
            poll_interval_seconds=0.5,
        )


def build_worker_launch_command(task_id: str) -> list[str]:
    return [sys.executable, "-m", "gateway.background_worker", "--task-id", str(task_id)]


def launch_background_worker(
    *,
    task_id: str,
    hermes_home: Path | None = None,
) -> dict[str, Any]:
    """Launch a durable background worker, preferring ``systemd-run`` when available."""
    hermes_home = Path(hermes_home or get_hermes_home())
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    command = build_worker_launch_command(task_id)

    if (
        os.name == "posix"
        and shutil.which("systemd-run")
        and os.getenv("HERMES_BACKGROUND_DISABLE_SYSTEMD_RUN", "").strip().lower()
        not in {"1", "true", "yes", "on"}
    ):
        scope = "system" if os.geteuid() == 0 else "user"
        unit = f"hermes-bg-{datetime.now().strftime('%H%M%S')}-{task_id.replace('_', '-')[:24]}"
        args = ["systemd-run"]
        if scope == "user":
            args.append("--user")
        args.extend(
            [
                "--unit",
                unit,
                "--same-dir",
                "--collect",
                "--quiet",
                "--setenv",
                f"HERMES_HOME={hermes_home}",
                "--setenv",
                f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
                *command,
            ]
        )
        try:
            subprocess.run(args, check=True, env=env, cwd=os.getcwd())
            return {
                "launcher_type": "systemd-run",
                "launcher_unit": unit,
                "launcher_scope": scope,
                "launcher_pid": None,
                "launcher_command": shlex.join(command),
            }
        except subprocess.CalledProcessError:
            logger.warning("systemd-run launch failed for %s; falling back to subprocess", task_id)

    proc = subprocess.Popen(
        command,
        cwd=os.getcwd(),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return {
        "launcher_type": "subprocess",
        "launcher_pid": int(proc.pid),
        "launcher_unit": "",
        "launcher_scope": "",
        "launcher_command": shlex.join(command),
    }


def stop_background_worker(job: dict[str, Any]) -> bool:
    """Best-effort stop for a launched background worker."""
    launcher_type = str(job.get("launcher_type") or "").strip()
    if launcher_type == "systemd-run":
        unit = str(job.get("launcher_unit") or "").strip()
        if not unit:
            return False
        scope = str(job.get("launcher_scope") or "").strip().lower()
        args = ["systemctl"]
        if scope == "user":
            args.append("--user")
        args.extend(["stop", unit])
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    pid = job.get("launcher_pid")
    if pid in (None, ""):
        return False
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return True
    return True
