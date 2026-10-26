"""Persistent task state for Hermes' cross-platform personal-computer mode.

This module deliberately stores task metadata only. It does not execute shell,
browser, or computer actions; those must be connected through an authorized
worker and the existing terminal/browser backends.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class InvalidTaskTransition(ValueError):
    """Raised when a task status transition is not permitted."""


@dataclass(frozen=True)
class ComputerTask:
    id: str
    prompt: str
    source: str
    status: TaskStatus
    created_at: str
    updated_at: str
    result: str | None = None
    approval_state: str | None = None
    approval_reason: str | None = None


@dataclass(frozen=True)
class TaskEvent:
    id: int
    task_id: str
    event: str
    detail: str
    created_at: str


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.RUNNING: frozenset({
        TaskStatus.AWAITING_APPROVAL, TaskStatus.PAUSED, TaskStatus.COMPLETED,
        TaskStatus.FAILED, TaskStatus.CANCELLED,
    }),
    TaskStatus.AWAITING_APPROVAL: frozenset({TaskStatus.PAUSED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.PAUSED: frozenset({TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.FAILED: frozenset({TaskStatus.QUEUED, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class ComputerTaskStore:
    """SQLite-backed task ledger shared by desktop, mobile, and messaging adapters.

    The caller chooses the database path. Use a private application-data
    directory and restrictive file permissions for real deployments.
    """

    def __init__(self, database_path: str | Path):
        raw_path = str(database_path)
        if raw_path == ":memory:":
            raise ValueError("ComputerTaskStore requires a persistent database file path")
        self.database_path = str(Path(raw_path).expanduser())
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit/rollback each unit of work and always close its connection."""
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def _initialize(self) -> None:
        with self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS computer_tasks (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result TEXT,
                    approval_state TEXT,
                    approval_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS computer_tasks_updated
                    ON computer_tasks(updated_at DESC);
                CREATE TABLE IF NOT EXISTS computer_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES computer_tasks(id) ON DELETE CASCADE,
                    event TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS computer_task_events_task
                    ON computer_task_events(task_id, id);
                """
            )

    @staticmethod
    def _task(row: sqlite3.Row | None) -> ComputerTask | None:
        if row is None:
            return None
        return ComputerTask(
            id=row["id"],
            prompt=row["prompt"],
            source=row["source"],
            status=TaskStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=row["result"],
            approval_state=row["approval_state"],
            approval_reason=row["approval_reason"],
        )

    @staticmethod
    def _event(db: sqlite3.Connection, task_id: str, event: str, detail: str = "") -> None:
        db.execute(
            "INSERT INTO computer_task_events(task_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
            (task_id, event, detail, _now()),
        )

    def create_task(self, prompt: str, source: str = "desktop", task_id: str | None = None) -> ComputerTask:
        prompt = prompt.strip()
        source = source.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        now = _now()
        task_id = task_id or str(uuid.uuid4())
        with self._connection() as db:
            db.execute(
                """INSERT INTO computer_tasks
                   (id, prompt, source, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (task_id, prompt, source, TaskStatus.QUEUED.value, now, now),
            )
            self._event(db, task_id, "task_created", f"source={source}")
            row = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
        task = self._task(row)
        assert task is not None
        return task

    def get_task(self, task_id: str) -> ComputerTask | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
        return self._task(row)

    def list_tasks(self, limit: int = 50) -> list[ComputerTask]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM computer_tasks ORDER BY updated_at DESC, id LIMIT ?", (limit,)
            ).fetchall()
        return [task for row in rows if (task := self._task(row)) is not None]

    def transition(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: str | None = None,
        detail: str = "",
    ) -> ComputerTask:
        if not isinstance(status, TaskStatus):
            status = TaskStatus(status)
        with self._connection() as db:
            row = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
            task = self._task(row)
            if task is None:
                raise KeyError(f"unknown task: {task_id}")
            if task.approval_state == "pending" and status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                raise InvalidTaskTransition("a pending approval must be resolved before the task can resume")
            if task.approval_state == "denied" and status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                raise InvalidTaskTransition("a denied action must be approved again before the task can resume")
            if status not in _ALLOWED_TRANSITIONS[task.status]:
                raise InvalidTaskTransition(f"{task.status.value} -> {status.value} is not allowed")
            now = _now()
            db.execute(
                "UPDATE computer_tasks SET status = ?, updated_at = ?, result = COALESCE(?, result) WHERE id = ?",
                (status.value, now, result, task_id),
            )
            self._event(db, task_id, "status_changed", f"{task.status.value}->{status.value}" + (f"; {detail}" if detail else ""))
            updated = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
        task = self._task(updated)
        assert task is not None
        return task

    def request_approval(self, task_id: str, reason: str) -> ComputerTask:
        reason = reason.strip()
        if not reason:
            raise ValueError("approval reason must not be empty")
        with self._connection() as db:
            row = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
            task = self._task(row)
            if task is None:
                raise KeyError(f"unknown task: {task_id}")
            retrying_denied_request = task.status is TaskStatus.PAUSED and task.approval_state == "denied"
            if task.status is not TaskStatus.RUNNING and not retrying_denied_request:
                raise InvalidTaskTransition("approval can only be requested for a running task or a paused task with a denied request")
            now = _now()
            db.execute(
                """UPDATE computer_tasks SET status = ?, approval_state = 'pending',
                   approval_reason = ?, updated_at = ? WHERE id = ?""",
                (TaskStatus.AWAITING_APPROVAL.value, reason, now, task_id),
            )
            self._event(db, task_id, "approval_requested", reason)
            updated = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
        task = self._task(updated)
        assert task is not None
        return task

    def resolve_approval(self, task_id: str, *, approved: bool, detail: str = "") -> ComputerTask:
        with self._connection() as db:
            row = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
            task = self._task(row)
            if task is None:
                raise KeyError(f"unknown task: {task_id}")
            if task.status not in {TaskStatus.AWAITING_APPROVAL, TaskStatus.PAUSED} or task.approval_state != "pending":
                raise InvalidTaskTransition("task has no pending approval")
            new_status = TaskStatus.RUNNING if approved else TaskStatus.PAUSED
            decision = "approved" if approved else "denied"
            now = _now()
            db.execute(
                """UPDATE computer_tasks SET status = ?, approval_state = ?,
                   updated_at = ? WHERE id = ?""",
                (new_status.value, decision, now, task_id),
            )
            self._event(db, task_id, f"approval_{decision}", detail)
            updated = db.execute("SELECT * FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
        task = self._task(updated)
        assert task is not None
        return task

    def list_events(self, task_id: str) -> list[TaskEvent]:
        with self._connection() as db:
            exists = db.execute("SELECT 1 FROM computer_tasks WHERE id = ?", (task_id,)).fetchone()
            if exists is None:
                raise KeyError(f"unknown task: {task_id}")
            rows = db.execute(
                "SELECT id, task_id, event, detail, created_at FROM computer_task_events WHERE task_id = ? ORDER BY id",
                (task_id,),
            ).fetchall()
        return [TaskEvent(**dict(row)) for row in rows]
