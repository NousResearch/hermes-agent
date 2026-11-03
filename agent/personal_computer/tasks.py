"""Persistent, cross-client task state for Personal Computer mode.

The store is deliberately independent of a UI or worker backend so desktop,
mobile, and messaging adapters can share task identities. Execution remains
owned by the existing Hermes runtime; this module never runs shell commands.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


_TERMINAL = {TaskStatus.FAILED, TaskStatus.COMPLETED, TaskStatus.CANCELLED}
_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.AWAITING_APPROVAL, TaskStatus.PAUSED, TaskStatus.FAILED,
        TaskStatus.COMPLETED, TaskStatus.CANCELLED,
    },
    TaskStatus.AWAITING_APPROVAL: {
        TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.CANCELLED, TaskStatus.FAILED,
    },
    TaskStatus.PAUSED: {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.FAILED: set(),
    TaskStatus.COMPLETED: set(),
    TaskStatus.CANCELLED: set(),
}


@dataclass(frozen=True, slots=True)
class PersonalComputerTask:
    """A durable task record; profile and workspace IDs scope its permissions."""

    id: str
    title: str
    requester_id: str
    source: str
    profile_id: str
    status: TaskStatus
    created_at: str
    updated_at: str
    summary: str = ""
    workspace_id: str = "default"
    idempotency_key: str | None = None
    approval_required: bool = True
    approval_state: str = "not_requested"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "PersonalComputerTask":
        data = dict(row)
        data["status"] = TaskStatus(data["status"])
        data["approval_required"] = bool(data["approval_required"])
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return cls(**data)


class PersonalComputerTaskStore:
    """SQLite-backed task ledger safe to reopen after a process restart.

    The database should live under the user's Hermes home directory. Do not
    put credentials, message bodies, or browser cookies in metadata.
    """

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate_legacy_idempotency_constraint(self, connection: sqlite3.Connection) -> None:
        """Replace the pre-profile unique key without losing existing task rows."""
        indexes = connection.execute(
            "PRAGMA index_list(personal_computer_tasks)"
        ).fetchall()
        legacy_index_found = False
        for index in indexes:
            if not index["unique"]:
                continue
            columns = connection.execute(
                f"PRAGMA index_info('{index['name']}')"
            ).fetchall()
            if [row["name"] for row in columns] == [
                "requester_id", "source", "idempotency_key"
            ]:
                legacy_index_found = True
                break

        if legacy_index_found:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE personal_computer_tasks_migrating (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    requester_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    workspace_id TEXT NOT NULL DEFAULT 'default',
                    idempotency_key TEXT,
                    approval_required INTEGER NOT NULL DEFAULT 1,
                    approval_state TEXT NOT NULL DEFAULT 'not_requested',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(requester_id, source, profile_id, idempotency_key)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO personal_computer_tasks_migrating
                SELECT id, title, requester_id, source, profile_id, status,
                       created_at, updated_at, summary, workspace_id,
                       idempotency_key, approval_required,
                       CASE
                           WHEN status = 'awaiting_approval' AND approval_required = 1 THEN 'pending'
                           ELSE approval_state
                       END,
                       metadata_json
                FROM personal_computer_tasks
                """
            )
            connection.execute("DROP TABLE personal_computer_tasks")
            connection.execute(
                "ALTER TABLE personal_computer_tasks_migrating RENAME TO personal_computer_tasks"
            )
            break

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_computer_tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    requester_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    workspace_id TEXT NOT NULL DEFAULT 'default',
                    idempotency_key TEXT,
                    approval_required INTEGER NOT NULL DEFAULT 1,
                    approval_state TEXT NOT NULL DEFAULT 'not_requested',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(requester_id, source, profile_id, idempotency_key)
                )
                """
            )
            self._migrate_legacy_idempotency_constraint(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pc_tasks_profile_updated "
                "ON personal_computer_tasks(profile_id, updated_at DESC)"
            )

    def create_task(
        self,
        *,
        title: str,
        requester_id: str,
        source: str,
        profile_id: str,
        workspace_id: str = "default",
        idempotency_key: str | None = None,
        approval_required: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> PersonalComputerTask:
        """Create a task or return the prior task for a repeated delivery key."""
        required = {
            "title": title.strip(),
            "requester_id": requester_id.strip(),
            "source": source.strip(),
            "profile_id": profile_id.strip(),
            "workspace_id": workspace_id.strip(),
        }
        if any(not value for value in required.values()):
            raise ValueError("title, requester_id, source, profile_id and workspace_id are required")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

        now = _now()
        task_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    """
                    SELECT * FROM personal_computer_tasks
                    WHERE requester_id = ? AND source = ? AND profile_id = ?
                      AND idempotency_key = ?
                    """,
                    (
                        required["requester_id"], required["source"],
                        required["profile_id"], idempotency_key,
                    ),
                ).fetchone()
                if existing:
                    return PersonalComputerTask.from_row(existing)
            connection.execute(
                """
                INSERT INTO personal_computer_tasks (
                    id, title, requester_id, source, profile_id, status,
                    created_at, updated_at, workspace_id, idempotency_key,
                    approval_required, approval_state, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, required["title"], required["requester_id"],
                    required["source"], required["profile_id"], TaskStatus.QUEUED.value,
                    now, now, required["workspace_id"], idempotency_key,
                    int(approval_required), "not_requested" if approval_required else "not_required",
                    json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True),
                ),
            )
            row = connection.execute(
                "SELECT * FROM personal_computer_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return PersonalComputerTask.from_row(row)

    def get_task(self, task_id: str) -> PersonalComputerTask | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM personal_computer_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return PersonalComputerTask.from_row(row) if row else None

    def list_tasks(
        self, *, profile_id: str, limit: int = 50, statuses: set[TaskStatus] | None = None
    ) -> list[PersonalComputerTask]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        query = "SELECT * FROM personal_computer_tasks WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if statuses is not None and not statuses:
            return []
        if statuses is not None:
            values = sorted(status.value for status in statuses)
            query += f" AND status IN ({','.join('?' for _ in values)})"
            params.extend(values)
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [PersonalComputerTask.from_row(row) for row in rows]

    def transition(
        self,
        task_id: str,
        new_status: TaskStatus,
        *,
        expected_status: TaskStatus | None = None,
        summary: str | None = None,
        approval_state: str | None = None,
    ) -> PersonalComputerTask:
        """Atomically transition a task; approval decisions must use approve()."""
        if approval_state is not None:
            raise ValueError("approval_state can only be changed by approve()")
        return self._transition_task(
            task_id,
            new_status,
            expected_status=expected_status,
            summary=summary,
            approval_state=approval_state,
            allow_approval_resume=False,
        )

    def _transition_task(
        self,
        task_id: str,
        new_status: TaskStatus,
        *,
        expected_status: TaskStatus | None = None,
        summary: str | None = None,
        approval_state: str | None = None,
        allow_approval_resume: bool,
    ) -> PersonalComputerTask:
        """Apply a transition under a write lock, optionally from approve()."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM personal_computer_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown task: {task_id}")
            current = TaskStatus(row["status"])
            if expected_status is not None and current != expected_status:
                raise ValueError(f"Task state changed: expected {expected_status.value}, found {current.value}")
            if new_status != current and new_status not in _TRANSITIONS[current]:
                raise ValueError(f"Invalid task transition: {current.value} -> {new_status.value}")
            if approval_state not in (None, "pending", "approved", "rejected", "not_required", "not_requested"):
                raise ValueError("Invalid approval_state")
            next_approval = approval_state if approval_state is not None else row["approval_state"]
            if new_status == TaskStatus.AWAITING_APPROVAL and not row["approval_required"]:
                raise ValueError("This task does not require approval")
            if new_status == TaskStatus.AWAITING_APPROVAL:
                if approval_state not in (None, "pending"):
                    raise ValueError("An approval request must remain pending until approve() is called")
                next_approval = "pending"
            if (
                new_status == TaskStatus.RUNNING
                and row["approval_required"]
                and (
                    current == TaskStatus.AWAITING_APPROVAL
                    or row["approval_state"] == "pending"
                )
                and not allow_approval_resume
            ):
                raise ValueError("Task must be approved through approve() before it can resume")
            next_summary = row["summary"] if summary is None else summary
            connection.execute(
                """
                UPDATE personal_computer_tasks
                SET status = ?, updated_at = ?, summary = ?, approval_state = ?
                WHERE id = ?
                """,
                (new_status.value, _now(), next_summary, next_approval, task_id),
            )
            updated = connection.execute(
                "SELECT * FROM personal_computer_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return PersonalComputerTask.from_row(updated)

    def approve(self, task_id: str, *, approved: bool) -> PersonalComputerTask:
        """Record an approval decision without executing the approved action."""
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        if task.status not in {
            TaskStatus.AWAITING_APPROVAL, TaskStatus.PAUSED, TaskStatus.QUEUED
        }:
            raise ValueError("Task is not waiting for approval")
        if not task.approval_required or task.approval_state != "pending":
            raise ValueError("Task does not have a pending approval")
        if approved:
            return self._transition_task(
                task_id, TaskStatus.RUNNING,
                expected_status=task.status,
                approval_state="approved",
                allow_approval_resume=True,
            )
        return self._transition_task(
            task_id, TaskStatus.CANCELLED,
            expected_status=task.status,
            approval_state="rejected",
            allow_approval_resume=True,
        )
