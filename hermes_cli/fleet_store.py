"""SQLite-backed coordination primitives for the federated runner pool."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from hermes_cli.fleet_protocol import FleetTask, RunnerCapability, TaskRequirement


@dataclass(frozen=True)
class TaskClaim:
    task_id: str
    claim_id: str
    node_id: str
    runner_profile: str
    lease_expires_at: float
    attempt: int


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    title: str
    body: str
    requirement: TaskRequirement
    idempotency_key: str
    status: str
    claim_id: str | None
    node_id: str | None
    runner_profile: str | None
    lease_expires_at: float | None
    attempt: int
    result: str | None
    error: str | None
    created_at: float
    updated_at: float


def _now() -> float:
    return time.time()


def _capability_matches(requirement: TaskRequirement, capability: RunnerCapability) -> bool:
    if requirement.models and not set(requirement.models).issubset(capability.models):
        return False
    if requirement.tools and not set(requirement.tools).issubset(capability.tools):
        return False
    if requirement.project and requirement.project not in capability.projects:
        return False
    return True


class FleetStore:
    """A local coordinator store with transactional compare-and-swap claims."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS fleet_runners (
                    node_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    capability_json TEXT NOT NULL,
                    last_seen REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    active_load INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (node_id, profile)
                );
                CREATE TABLE IF NOT EXISTS fleet_tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    requirement_json TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    claim_id TEXT,
                    node_id TEXT,
                    runner_profile TEXT,
                    lease_expires_at REAL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS fleet_tasks_claimable
                    ON fleet_tasks(status, created_at, lease_expires_at);
                CREATE TABLE IF NOT EXISTS fleet_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    event_time REAL NOT NULL,
                    detail TEXT
                );
                """
            )

    @property
    def journal_mode(self) -> str:
        with self._connection() as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def _event(
        self, connection: sqlite3.Connection, task_id: str | None, event_type: str, now: float
    ) -> None:
        connection.execute(
            "INSERT INTO fleet_events(task_id, event_type, event_time) VALUES (?, ?, ?)",
            (task_id, event_type, now),
        )

    def register_runner(self, capability: RunnerCapability, *, now: float | None = None, ttl: float = 30.0) -> None:
        now = _now() if now is None else now
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO fleet_runners(node_id, profile, capability_json, last_seen, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(node_id, profile) DO UPDATE SET
                    capability_json=excluded.capability_json,
                    last_seen=excluded.last_seen,
                    expires_at=excluded.expires_at
                """,
                (
                    capability.node_id,
                    capability.profile,
                    json.dumps(capability.to_dict(), sort_keys=True),
                    now,
                    now + ttl,
                ),
            )
            self._event(connection, None, "runner_registered", now)

    def heartbeat_runner(
        self, node_id: str, profile: str, *, now: float | None = None, ttl: float = 30.0
    ) -> bool:
        now = _now() if now is None else now
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE fleet_runners SET last_seen=?, expires_at=? WHERE node_id=? AND profile=?",
                (now, now + ttl, node_id, profile),
            )
            if cursor.rowcount:
                self._event(connection, None, "runner_heartbeat", now)
            return bool(cursor.rowcount)

    def submit_task(self, task: FleetTask, *, now: float | None = None) -> TaskRecord:
        now = _now() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM fleet_tasks WHERE idempotency_key=?", (task.idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["task_id"] != task.task_id:
                    raise ValueError("idempotency_key is already bound to another task")
                return self._record(existing)
            connection.execute(
                """
                INSERT INTO fleet_tasks(
                    task_id, title, body, requirement_json, idempotency_key, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    task.task_id,
                    task.title,
                    task.body,
                    json.dumps(task.requirement.to_dict(), sort_keys=True),
                    task.idempotency_key,
                    now,
                    now,
                ),
            )
            self._event(connection, task.task_id, "task_submitted", now)
            row = connection.execute("SELECT * FROM fleet_tasks WHERE task_id=?", (task.task_id,)).fetchone()
            assert row is not None
            return self._record(row)

    def claim_task(
        self,
        capability: RunnerCapability,
        *,
        now: float | None = None,
        lease_seconds: float = 60.0,
    ) -> TaskClaim | None:
        now = _now() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            runner = connection.execute(
                "SELECT * FROM fleet_runners WHERE node_id=? AND profile=? AND expires_at>?",
                (capability.node_id, capability.profile, now),
            ).fetchone()
            if runner is None:
                return None
            rows = connection.execute(
                "SELECT * FROM fleet_tasks WHERE status='pending' ORDER BY created_at, task_id"
            ).fetchall()
            selected = next(
                (
                    row
                    for row in rows
                    if _capability_matches(
                        TaskRequirement.from_dict(json.loads(row["requirement_json"])), capability
                    )
                ),
                None,
            )
            if selected is None:
                return None
            claim_id = uuid.uuid4().hex
            expires = now + lease_seconds
            updated = connection.execute(
                """
                UPDATE fleet_tasks
                SET status='running', claim_id=?, node_id=?, runner_profile=?,
                    lease_expires_at=?, attempt=attempt+1, updated_at=?
                WHERE task_id=? AND status='pending'
                """,
                (
                    claim_id,
                    capability.node_id,
                    capability.profile,
                    expires,
                    now,
                    selected["task_id"],
                ),
            )
            if updated.rowcount != 1:
                return None
            connection.execute(
                "UPDATE fleet_runners SET active_load=active_load+1 WHERE node_id=? AND profile=?",
                (capability.node_id, capability.profile),
            )
            self._event(connection, selected["task_id"], "task_claimed", now)
            return TaskClaim(
                task_id=selected["task_id"],
                claim_id=claim_id,
                node_id=capability.node_id,
                runner_profile=capability.profile,
                lease_expires_at=expires,
                attempt=selected["attempt"] + 1,
            )

    def renew_claim(
        self, task_id: str, claim_id: str, *, now: float | None = None, lease_seconds: float = 60.0
    ) -> TaskClaim | None:
        now = _now() if now is None else now
        expires = now + lease_seconds
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE fleet_tasks SET lease_expires_at=?, updated_at=?
                WHERE task_id=? AND claim_id=? AND status='running' AND lease_expires_at>?
                """,
                (expires, now, task_id, claim_id, now),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute("SELECT * FROM fleet_tasks WHERE task_id=?", (task_id,)).fetchone()
            assert row is not None
            self._event(connection, task_id, "claim_renewed", now)
            return self._claim(row)

    def complete_task(
        self, task_id: str, claim_id: str, *, result: str = "", now: float | None = None
    ) -> bool:
        return self._finish(task_id, claim_id, status="completed", result=result, error=None, now=now)

    def fail_task(
        self, task_id: str, claim_id: str, *, error: str, now: float | None = None
    ) -> bool:
        return self._finish(task_id, claim_id, status="failed", result=None, error=error, now=now)

    def retry_task(self, task_id: str, *, now: float | None = None) -> bool:
        now = _now() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE fleet_tasks SET status='pending', result=NULL, error=NULL, updated_at=?
                WHERE task_id=? AND status IN ('failed', 'cancelled')
                """,
                (now, task_id),
            )
            if updated.rowcount:
                self._event(connection, task_id, "task_retried", now)
            return bool(updated.rowcount)

    def cancel_task(self, task_id: str, *, now: float | None = None) -> bool:
        now = _now() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE fleet_tasks SET status='cancelled', updated_at=?
                WHERE task_id=? AND status IN ('pending', 'failed')
                """,
                (now, task_id),
            )
            if updated.rowcount:
                self._event(connection, task_id, "task_cancelled", now)
            return bool(updated.rowcount)

    def _finish(
        self,
        task_id: str,
        claim_id: str,
        *,
        status: str,
        result: str | None,
        error: str | None,
        now: float | None,
    ) -> bool:
        now = _now() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM fleet_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                return False
            if row["status"] == status and row["claim_id"] == claim_id:
                return True
            if row["status"] != "running" or row["claim_id"] != claim_id:
                return False
            connection.execute(
                """
                UPDATE fleet_tasks SET status=?, result=?, error=?, lease_expires_at=NULL, updated_at=?
                WHERE task_id=? AND claim_id=? AND status='running'
                """,
                (status, result, error, now, task_id, claim_id),
            )
            connection.execute(
                """
                UPDATE fleet_runners SET active_load=MAX(active_load-1, 0)
                WHERE node_id=? AND profile=?
                """,
                (row["node_id"], row["runner_profile"]),
            )
            self._event(connection, task_id, f"task_{status}", now)
            return True

    def reclaim_expired(self, *, now: float | None = None) -> int:
        now = _now() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM fleet_tasks WHERE status='running' AND lease_expires_at<=?", (now,)
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE fleet_tasks
                    SET status='pending', claim_id=NULL, node_id=NULL, runner_profile=NULL,
                        lease_expires_at=NULL, updated_at=?
                    WHERE task_id=? AND status='running' AND claim_id=?
                    """,
                    (now, row["task_id"], row["claim_id"]),
                )
                connection.execute(
                    """
                    UPDATE fleet_runners SET active_load=MAX(active_load-1, 0)
                    WHERE node_id=? AND profile=?
                    """,
                    (row["node_id"], row["runner_profile"]),
                )
                self._event(connection, row["task_id"], "claim_reclaimed", now)
            return len(rows)

    def list_tasks(self) -> list[TaskRecord]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM fleet_tasks ORDER BY created_at, task_id").fetchall()
            return [self._record(row) for row in rows]

    def list_runners(self, *, now: float | None = None) -> list[dict]:
        """Return the latest capability and liveness record for every runner.

        Expired rows stay visible so an operator can distinguish a runner that
        was never registered from one whose Desktop process went offline.
        Callers receive plain dictionaries because this is a read-only status
        surface, not a claim/lease protocol.
        """
        now = _now() if now is None else now
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT node_id, profile, capability_json, last_seen, expires_at, active_load "
                "FROM fleet_runners ORDER BY node_id, profile"
            ).fetchall()
        runners = []
        for row in rows:
            capability = json.loads(row["capability_json"])
            runners.append(
                {
                    "node_id": row["node_id"],
                    "profile": row["profile"],
                    "capability": capability,
                    "last_seen": row["last_seen"],
                    "expires_at": row["expires_at"],
                    "active_load": row["active_load"],
                    "online": row["expires_at"] > now,
                }
            )
        return runners

    @staticmethod
    def _record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            title=row["title"],
            body=row["body"],
            requirement=TaskRequirement.from_dict(json.loads(row["requirement_json"])),
            idempotency_key=row["idempotency_key"],
            status=row["status"],
            claim_id=row["claim_id"],
            node_id=row["node_id"],
            runner_profile=row["runner_profile"],
            lease_expires_at=row["lease_expires_at"],
            attempt=row["attempt"],
            result=row["result"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _claim(row: sqlite3.Row) -> TaskClaim:
        return TaskClaim(
            task_id=row["task_id"],
            claim_id=row["claim_id"],
            node_id=row["node_id"],
            runner_profile=row["runner_profile"],
            lease_expires_at=row["lease_expires_at"],
            attempt=row["attempt"],
        )
