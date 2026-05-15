"""SQLite-backed durable controller state for Hermes frontdesk.

This module is a default-off foundation: it does not start workers, register
gateway handlers, or replace the current in-memory runtime.  It gives later
frontdesk phases a restart-safe place to record tasks, worker/reviewer jobs,
events, and artifacts, plus small atomic helpers for the lifecycle transitions
that must not be inferred from process-local thread state.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from agent.task_registry import (
    FRONTDESK_BLOCKED_USER_INPUT,
    FRONTDESK_CANCEL_REQUESTED,
    FRONTDESK_CANCELLED,
    FRONTDESK_DONE_PRESENTED,
    FRONTDESK_ERROR,
    FRONTDESK_QUEUED,
    FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION,
    FRONTDESK_REVIEW_PASSED,
    FRONTDESK_RUNNING_REVIEW,
    FRONTDESK_RUNNING_WORKER,
    FRONTDESK_WORKER_DONE_PENDING_REVIEW,
    REVIEW_BLOCKED,
    REVIEW_FAILED,
    REVIEW_NEEDS_ITERATION,
    REVIEW_PASSED,
)

__all__ = [
    "ARTIFACT_DISCARDED",
    "ARTIFACT_IMPORT_REQUESTED",
    "ARTIFACT_PENDING_IMPORT",
    "JOB_CANCELLED",
    "JOB_FAILED",
    "JOB_QUEUED",
    "JOB_RECOVERING",
    "JOB_REVIEWER",
    "JOB_RUNNING",
    "JOB_SUCCEEDED",
    "JOB_WORKER",
    "REVIEW_REJECTED",
    "REVIEW_UNSAFE",
    "TERMINAL_JOB_STATES",
    "FrontdeskArtifactRecord",
    "FrontdeskEventRecord",
    "FrontdeskJobRecord",
    "FrontdeskStore",
    "FrontdeskTaskRecord",
]

JOB_WORKER = "worker"
JOB_REVIEWER = "reviewer"
JOB_KINDS = frozenset({JOB_WORKER, JOB_REVIEWER})

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
JOB_RECOVERING = "recovering"
JOB_STATES = frozenset(
    {JOB_QUEUED, JOB_RUNNING, JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED, JOB_RECOVERING}
)
TERMINAL_JOB_STATES = frozenset({JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED})

ARTIFACT_PENDING_IMPORT = "pending"
ARTIFACT_IMPORT_REQUESTED = "import_requested"
ARTIFACT_DISCARDED = "discarded"

REVIEW_REJECTED = "rejected"
REVIEW_UNSAFE = "unsafe"
_REJECTED_REVIEW_STATUSES = frozenset({REVIEW_REJECTED, "reject", REVIEW_UNSAFE})
_DISCARDABLE_TASK_STATES = frozenset(
    {
        FRONTDESK_REVIEW_PASSED,
        FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION,
        FRONTDESK_BLOCKED_USER_INPUT,
        FRONTDESK_DONE_PRESENTED,
        FRONTDESK_ERROR,
    }
)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _now() -> float:
    return time.time()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _json_load(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


def _json_safe(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_json_dump(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text or None


def _chmod_private(path: str) -> None:
    if path == ":memory:":
        return
    try:
        if os.path.exists(path):
            os.chmod(path, 0o600)
    except OSError:
        pass


def _secure_sqlite_files(path: str) -> None:
    if path == ":memory:":
        return
    _chmod_private(path)
    _chmod_private(f"{path}-wal")
    _chmod_private(f"{path}-shm")


def _validate_kind(kind: str) -> str:
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown frontdesk job kind {kind!r}")
    return kind


def _validate_job_state(state: str) -> str:
    if state not in JOB_STATES:
        raise ValueError(f"unknown frontdesk job state {state!r}")
    return state


@dataclass(frozen=True, slots=True)
class FrontdeskTaskRecord:
    id: str
    session_key: str | None
    origin: dict[str, Any]
    user_goal: str
    state: str
    created_at: float
    updated_at: float
    cancel_requested_at: float | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FrontdeskTaskRecord":
        return cls(
            id=row["id"],
            session_key=row["session_key"],
            origin=_json_load(row["origin_json"], {}),
            user_goal=row["user_goal"],
            state=row["state"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            cancel_requested_at=row["cancel_requested_at"],
        )


@dataclass(frozen=True, slots=True)
class FrontdeskJobRecord:
    id: str
    task_id: str
    kind: str
    state: str
    attempt: int
    lease_owner: str | None
    lease_expires_at: float | None
    pid: str | None
    session_id: str | None
    heartbeat_at: float | None
    exit_status: str | None
    result: dict[str, Any] | None
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FrontdeskJobRecord":
        return cls(
            id=row["id"],
            task_id=row["task_id"],
            kind=row["kind"],
            state=row["state"],
            attempt=int(row["attempt"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            pid=row["pid"],
            session_id=row["session_id"],
            heartbeat_at=row["heartbeat_at"],
            exit_status=row["exit_status"],
            result=_json_load(row["result_json"], None),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )


@dataclass(frozen=True, slots=True)
class FrontdeskEventRecord:
    id: str
    task_id: str | None
    job_id: str | None
    event_type: str
    payload: dict[str, Any]
    created_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FrontdeskEventRecord":
        return cls(
            id=row["id"],
            task_id=row["task_id"],
            job_id=row["job_id"],
            event_type=row["event_type"],
            payload=_json_load(row["payload_json"], {}),
            created_at=float(row["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class FrontdeskArtifactRecord:
    id: str
    task_id: str
    job_id: str | None
    path: str
    artifact_type: str
    checksum: str | None
    size: int | None
    import_status: str
    created_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FrontdeskArtifactRecord":
        return cls(
            id=row["id"],
            task_id=row["task_id"],
            job_id=row["job_id"],
            path=row["path"],
            artifact_type=row["artifact_type"],
            checksum=row["checksum"],
            size=row["size"],
            import_status=row["import_status"],
            created_at=float(row["created_at"]),
        )


class FrontdeskStore:
    """Durable SQLite controller store for frontdesk task/job state."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).touch(mode=0o600, exist_ok=True)
            _secure_sqlite_files(self.path)
        self._lock = RLock()
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            _secure_sqlite_files(self.path)
        self._init_schema()
        _secure_sqlite_files(self.path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "FrontdeskStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS frontdesk_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO frontdesk_meta(key, value)
                VALUES ('schema_version', '1');

            CREATE TABLE IF NOT EXISTS frontdesk_tasks (
                id TEXT PRIMARY KEY,
                session_key TEXT,
                origin_json TEXT NOT NULL DEFAULT '{}',
                user_goal TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                cancel_requested_at REAL
            );

            CREATE TABLE IF NOT EXISTS frontdesk_jobs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES frontdesk_tasks(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_expires_at REAL,
                pid TEXT,
                session_id TEXT,
                heartbeat_at REAL,
                exit_status TEXT,
                result_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS frontdesk_one_reviewer_job
                ON frontdesk_jobs(task_id, kind)
                WHERE kind = 'reviewer';
            CREATE INDEX IF NOT EXISTS frontdesk_jobs_claim_idx
                ON frontdesk_jobs(kind, state, created_at);

            CREATE TABLE IF NOT EXISTS frontdesk_events (
                id TEXT PRIMARY KEY,
                task_id TEXT REFERENCES frontdesk_tasks(id) ON DELETE CASCADE,
                job_id TEXT REFERENCES frontdesk_jobs(id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS frontdesk_events_task_idx
                ON frontdesk_events(task_id, created_at);

            CREATE TABLE IF NOT EXISTS frontdesk_artifacts (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES frontdesk_tasks(id) ON DELETE CASCADE,
                job_id TEXT REFERENCES frontdesk_jobs(id) ON DELETE SET NULL,
                path TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                checksum TEXT,
                size INTEGER,
                import_status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS frontdesk_artifacts_task_idx
                ON frontdesk_artifacts(task_id, created_at);
            """
        )

    class _Transaction:
        def __init__(self, outer: "FrontdeskStore") -> None:
            self.outer = outer

        def __enter__(self) -> None:
            self.outer._lock.acquire()
            try:
                self.outer._conn.execute("BEGIN IMMEDIATE")
            except Exception:
                self.outer._lock.release()
                raise

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            try:
                self.outer._conn.execute("ROLLBACK" if exc_type else "COMMIT")
            finally:
                self.outer._lock.release()

    def _transaction(self) -> "FrontdeskStore._Transaction":
        return FrontdeskStore._Transaction(self)

    # -- row helpers -----------------------------------------------------
    def _task(self, task_id: str) -> FrontdeskTaskRecord:
        row = self._conn.execute(
            "SELECT * FROM frontdesk_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown frontdesk task id: {task_id!r}")
        return FrontdeskTaskRecord.from_row(row)

    def _job(self, job_id: str) -> FrontdeskJobRecord:
        row = self._conn.execute(
            "SELECT * FROM frontdesk_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown frontdesk job id: {job_id!r}")
        return FrontdeskJobRecord.from_row(row)

    def _insert_event(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        job_id: str | None = None,
        payload: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> FrontdeskEventRecord:
        created = _now() if now is None else float(now)
        event_id = _new_id("event")
        payload_json = _json_dump(_json_safe(payload or {}, label="event payload"))
        self._conn.execute(
            """
            INSERT INTO frontdesk_events(id, task_id, job_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, task_id, job_id, event_type, payload_json, created),
        )
        return FrontdeskEventRecord(
            id=event_id,
            task_id=task_id,
            job_id=job_id,
            event_type=event_type,
            payload=json.loads(payload_json),
            created_at=created,
        )

    def _insert_job(
        self,
        *,
        task_id: str,
        kind: str,
        state: str,
        now: float,
        job_id: str | None = None,
    ) -> FrontdeskJobRecord:
        _validate_kind(kind)
        _validate_job_state(state)
        jid = job_id or _new_id(f"{kind}-job")
        self._conn.execute(
            """
            INSERT INTO frontdesk_jobs(
                id, task_id, kind, state, attempt, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (jid, task_id, kind, state, now, now),
        )
        self._insert_event(f"{kind}_job_enqueued", task_id=task_id, job_id=jid, now=now)
        return self._job(jid)

    def _insert_artifacts(
        self,
        *,
        task_id: str,
        job_id: str,
        artifacts: Iterable[dict[str, Any]],
        now: float,
    ) -> list[FrontdeskArtifactRecord]:
        inserted: list[FrontdeskArtifactRecord] = []
        for raw in artifacts:
            if not isinstance(raw, dict):
                raise TypeError("artifact records must be dictionaries")
            artifact = _json_safe(raw, label="artifact")
            path = _string_or_none(artifact.get("path"))
            if not path:
                raise ValueError("artifact path is required")
            artifact_type = _string_or_none(
                artifact.get("type", artifact.get("artifact_type", artifact.get("kind")))
            ) or "artifact"
            checksum = _string_or_none(artifact.get("checksum"))
            size = artifact.get("size")
            if size is not None:
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError("artifact size must be a non-negative integer")
            import_status = _string_or_none(artifact.get("import_status")) or ARTIFACT_PENDING_IMPORT
            artifact_id = _string_or_none(artifact.get("id")) or _new_id("artifact")
            self._conn.execute(
                """
                INSERT INTO frontdesk_artifacts(
                    id, task_id, job_id, path, artifact_type, checksum, size,
                    import_status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    task_id,
                    job_id,
                    path,
                    artifact_type,
                    checksum,
                    size,
                    import_status,
                    now,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM frontdesk_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
            inserted.append(FrontdeskArtifactRecord.from_row(row))
        if inserted:
            self._insert_event(
                "artifacts_recorded",
                task_id=task_id,
                job_id=job_id,
                payload={"artifact_ids": [artifact.id for artifact in inserted]},
                now=now,
            )
        return inserted

    # -- creation / lookup ----------------------------------------------
    def create_task_with_worker_job(
        self,
        user_goal: str,
        *,
        session_key: str | None = None,
        origin: dict[str, Any] | None = None,
        task_id: str | None = None,
        job_id: str | None = None,
    ) -> tuple[FrontdeskTaskRecord, FrontdeskJobRecord]:
        """Atomically create a task and its first queued worker job."""
        now = _now()
        tid = task_id or _new_id("task")
        origin_json = _json_dump(_json_safe(origin or {}, label="task origin"))
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO frontdesk_tasks(
                    id, session_key, origin_json, user_goal, state, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (tid, session_key, origin_json, str(user_goal), FRONTDESK_QUEUED, now, now),
            )
            self._insert_event(
                "task_created",
                task_id=tid,
                payload={"session_key": session_key, "origin": json.loads(origin_json)},
                now=now,
            )
            worker_job = self._insert_job(
                task_id=tid, kind=JOB_WORKER, state=JOB_QUEUED, job_id=job_id, now=now
            )
            return self._task(tid), worker_job

    enqueue_task = create_task_with_worker_job

    def get_task(self, task_id: str) -> FrontdeskTaskRecord | None:
        row = self._conn.execute(
            "SELECT * FROM frontdesk_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return FrontdeskTaskRecord.from_row(row) if row is not None else None

    def get_job(self, job_id: str) -> FrontdeskJobRecord | None:
        row = self._conn.execute(
            "SELECT * FROM frontdesk_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return FrontdeskJobRecord.from_row(row) if row is not None else None

    def list_tasks(self, *, session_key: str | None = None) -> list[FrontdeskTaskRecord]:
        if session_key is None:
            rows = self._conn.execute(
                "SELECT * FROM frontdesk_tasks ORDER BY created_at, id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM frontdesk_tasks WHERE session_key = ? ORDER BY created_at, id",
                (session_key,),
            ).fetchall()
        return [FrontdeskTaskRecord.from_row(row) for row in rows]

    def list_jobs(
        self,
        *,
        task_id: str | None = None,
        kind: str | None = None,
        state: str | None = None,
    ) -> list[FrontdeskJobRecord]:
        clauses = []
        params: list[Any] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(_validate_kind(kind))
        if state is not None:
            clauses.append("state = ?")
            params.append(_validate_job_state(state))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM frontdesk_jobs{where} ORDER BY created_at, id", params
        ).fetchall()
        return [FrontdeskJobRecord.from_row(row) for row in rows]

    def list_events(self, *, task_id: str | None = None) -> list[FrontdeskEventRecord]:
        if task_id is None:
            rows = self._conn.execute(
                "SELECT * FROM frontdesk_events ORDER BY created_at, rowid"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM frontdesk_events WHERE task_id = ? ORDER BY created_at, rowid",
                (task_id,),
            ).fetchall()
        return [FrontdeskEventRecord.from_row(row) for row in rows]

    def list_artifacts(self, *, task_id: str | None = None) -> list[FrontdeskArtifactRecord]:
        if task_id is None:
            rows = self._conn.execute(
                "SELECT * FROM frontdesk_artifacts ORDER BY created_at, id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM frontdesk_artifacts WHERE task_id = ? ORDER BY created_at, id",
                (task_id,),
            ).fetchall()
        return [FrontdeskArtifactRecord.from_row(row) for row in rows]

    def _selected_task_artifacts(
        self,
        task_id: str,
        artifact_ids: Iterable[str] | None,
    ) -> list[FrontdeskArtifactRecord]:
        self._task(task_id)
        if artifact_ids is None:
            return self.list_artifacts(task_id=task_id)
        ids = [str(artifact_id) for artifact_id in artifact_ids]
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT * FROM frontdesk_artifacts
            WHERE task_id = ? AND id IN ({placeholders})
            ORDER BY created_at, id
            """,
            [task_id, *ids],
        ).fetchall()
        artifacts = [FrontdeskArtifactRecord.from_row(row) for row in rows]
        found = {artifact.id for artifact in artifacts}
        missing = [artifact_id for artifact_id in ids if artifact_id not in found]
        if missing:
            raise KeyError(f"unknown frontdesk artifact id for task {task_id!r}: {missing[0]!r}")
        return artifacts

    def _task_has_event(self, task_id: str, event_type: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM frontdesk_events
            WHERE task_id = ? AND event_type = ?
            LIMIT 1
            """,
            (task_id, event_type),
        ).fetchone()
        return row is not None

    # -- worker/reviewer queue operations --------------------------------
    def claim_job(
        self,
        *,
        kind: str | None = None,
        job_id: str | None = None,
        lease_owner: str,
        lease_seconds: float,
        now: float | None = None,
        pid: str | int | None = None,
        session_id: str | None = None,
    ) -> FrontdeskJobRecord | None:
        """Atomically claim the oldest queued job and attach a lease."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        claimed_at = _now() if now is None else float(now)
        expires_at = claimed_at + float(lease_seconds)
        with self._transaction():
            params: list[Any] = [JOB_QUEUED]
            kind_clause = ""
            if kind is not None:
                kind_clause = " AND kind = ?"
                params.append(_validate_kind(kind))
            job_clause = ""
            if job_id is not None:
                job_clause = " AND id = ?"
                params.append(str(job_id))
            row = self._conn.execute(
                f"""
                SELECT * FROM frontdesk_jobs
                WHERE state = ?{kind_clause}{job_clause}
                ORDER BY created_at, id
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            job = FrontdeskJobRecord.from_row(row)
            self._conn.execute(
                """
                UPDATE frontdesk_jobs
                SET state = ?, attempt = attempt + 1, lease_owner = ?,
                    lease_expires_at = ?, pid = ?, session_id = ?, heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    JOB_RUNNING,
                    str(lease_owner),
                    expires_at,
                    _string_or_none(pid),
                    session_id,
                    claimed_at,
                    claimed_at,
                    job.id,
                ),
            )
            if job.kind == JOB_WORKER:
                self._conn.execute(
                    "UPDATE frontdesk_tasks SET state = ?, updated_at = ? WHERE id = ?",
                    (FRONTDESK_RUNNING_WORKER, claimed_at, job.task_id),
                )
            elif job.kind == JOB_REVIEWER:
                self._conn.execute(
                    "UPDATE frontdesk_tasks SET state = ?, updated_at = ? WHERE id = ?",
                    (FRONTDESK_RUNNING_REVIEW, claimed_at, job.task_id),
                )
            self._insert_event(
                "job_claimed",
                task_id=job.task_id,
                job_id=job.id,
                payload={"lease_owner": str(lease_owner), "lease_expires_at": expires_at},
                now=claimed_at,
            )
            return self._job(job.id)

    def heartbeat_job(
        self,
        job_id: str,
        *,
        lease_owner: str,
        attempt: int,
        extend_seconds: float | None = None,
        now: float | None = None,
        pid: str | int | None = None,
        session_id: str | None = None,
    ) -> FrontdeskJobRecord:
        beat_at = _now() if now is None else float(now)
        with self._transaction():
            job = self._job(job_id)
            self._claim_token_matches(job, lease_owner=lease_owner, attempt=attempt)
            lease_expires_at = (
                beat_at + float(extend_seconds)
                if extend_seconds is not None
                else job.lease_expires_at
            )
            self._conn.execute(
                """
                UPDATE frontdesk_jobs
                SET heartbeat_at = ?, lease_expires_at = ?,
                    pid = COALESCE(?, pid),
                    session_id = COALESCE(?, session_id),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    beat_at,
                    lease_expires_at,
                    _string_or_none(pid),
                    session_id,
                    beat_at,
                    job_id,
                ),
            )
            self._insert_event("job_heartbeat", task_id=job.task_id, job_id=job_id, now=beat_at)
            return self._job(job_id)

    def recover_expired_leases(self, *, now: float | None = None) -> list[FrontdeskJobRecord]:
        """Requeue running jobs whose lease expired.

        The explicit recovery policy for this foundation is requeue: stale
        running jobs go back to ``queued`` with lease/process fields cleared, and
        a recovery event records the transition.
        """
        cutoff = _now() if now is None else float(now)
        recovered: list[FrontdeskJobRecord] = []
        with self._transaction():
            rows = self._conn.execute(
                """
                SELECT * FROM frontdesk_jobs
                WHERE state = ? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                ORDER BY lease_expires_at, id
                """,
                (JOB_RUNNING, cutoff),
            ).fetchall()
            for row in rows:
                job = FrontdeskJobRecord.from_row(row)
                self._conn.execute(
                    """
                    UPDATE frontdesk_jobs
                    SET state = ?, lease_owner = NULL, lease_expires_at = NULL,
                        pid = NULL, session_id = NULL, heartbeat_at = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (JOB_QUEUED, cutoff, job.id),
                )
                if job.kind == JOB_WORKER:
                    task_state = FRONTDESK_QUEUED
                else:
                    task_state = FRONTDESK_WORKER_DONE_PENDING_REVIEW
                self._conn.execute(
                    "UPDATE frontdesk_tasks SET state = ?, updated_at = ? WHERE id = ?",
                    (task_state, cutoff, job.task_id),
                )
                self._insert_event(
                    "job_lease_expired_requeued",
                    task_id=job.task_id,
                    job_id=job.id,
                    payload={"previous_lease_owner": job.lease_owner},
                    now=cutoff,
                )
                recovered.append(self._job(job.id))
        return recovered

    def _claim_token_matches(
        self,
        job: FrontdeskJobRecord,
        *,
        lease_owner: str,
        attempt: int,
        require_running: bool = True,
    ) -> None:
        """Validate that a completion/heartbeat belongs to the current lease."""
        if require_running and job.state != JOB_RUNNING:
            raise ValueError("job must be running before it can be completed")
        if job.lease_owner != str(lease_owner):
            raise ValueError("lease owner does not match claimed job")
        if job.attempt != int(attempt):
            raise ValueError("job attempt does not match claimed job")

    def complete_worker_job(
        self,
        job_id: str,
        *,
        success: bool,
        lease_owner: str,
        attempt: int,
        cancelled: bool = False,
        exit_status: str | int | None = None,
        result: dict[str, Any] | None = None,
        artifacts: Iterable[dict[str, Any]] | None = None,
        now: float | None = None,
    ) -> tuple[FrontdeskJobRecord, FrontdeskJobRecord | None]:
        """Complete a worker job and enqueue exactly one reviewer on success."""
        completed_at = _now() if now is None else float(now)
        result_json = _json_dump(_json_safe(result or {}, label="worker result"))
        with self._transaction():
            job = self._job(job_id)
            if job.kind != JOB_WORKER:
                raise ValueError("job is not a worker job")
            if job.state in TERMINAL_JOB_STATES:
                if not (job.state == JOB_CANCELLED and cancelled):
                    self._claim_token_matches(
                        job,
                        lease_owner=lease_owner,
                        attempt=attempt,
                        require_running=False,
                    )
                reviewer = self._existing_reviewer(job.task_id)
                return job, reviewer
            self._claim_token_matches(job, lease_owner=lease_owner, attempt=attempt)

            final_state = JOB_SUCCEEDED if success else (JOB_CANCELLED if cancelled else JOB_FAILED)
            self._conn.execute(
                """
                UPDATE frontdesk_jobs
                SET state = ?, lease_expires_at = NULL,
                    heartbeat_at = ?, exit_status = ?, result_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    final_state,
                    completed_at,
                    _string_or_none(exit_status),
                    result_json,
                    completed_at,
                    job_id,
                ),
            )
            self._insert_artifacts(
                task_id=job.task_id,
                job_id=job_id,
                artifacts=artifacts or (),
                now=completed_at,
            )
            self._insert_event(
                "worker_job_completed",
                task_id=job.task_id,
                job_id=job_id,
                payload={
                    "success": bool(success),
                    "cancelled": bool(cancelled),
                    "exit_status": _string_or_none(exit_status),
                },
                now=completed_at,
            )
            reviewer: FrontdeskJobRecord | None = None
            if success:
                self._conn.execute(
                    "UPDATE frontdesk_tasks SET state = ?, updated_at = ? WHERE id = ?",
                    (FRONTDESK_WORKER_DONE_PENDING_REVIEW, completed_at, job.task_id),
                )
                reviewer = self._ensure_reviewer_job(task_id=job.task_id, now=completed_at)
            else:
                task_state = FRONTDESK_CANCELLED if cancelled else FRONTDESK_ERROR
                self._conn.execute(
                    "UPDATE frontdesk_tasks SET state = ?, updated_at = ? WHERE id = ?",
                    (task_state, completed_at, job.task_id),
                )
            return self._job(job_id), reviewer

    def _existing_reviewer(self, task_id: str) -> FrontdeskJobRecord | None:
        row = self._conn.execute(
            """
            SELECT * FROM frontdesk_jobs
            WHERE task_id = ? AND kind = ?
            ORDER BY created_at, id
            LIMIT 1
            """,
            (task_id, JOB_REVIEWER),
        ).fetchone()
        return FrontdeskJobRecord.from_row(row) if row is not None else None

    def _ensure_reviewer_job(self, *, task_id: str, now: float) -> FrontdeskJobRecord:
        existing = self._existing_reviewer(task_id)
        if existing is not None:
            return existing
        return self._insert_job(task_id=task_id, kind=JOB_REVIEWER, state=JOB_QUEUED, now=now)

    def complete_reviewer_job(
        self,
        job_id: str,
        *,
        review_status: str,
        lease_owner: str,
        attempt: int,
        exit_status: str | int | None = None,
        result: dict[str, Any] | None = None,
        artifacts: Iterable[dict[str, Any]] | None = None,
        now: float | None = None,
    ) -> FrontdeskJobRecord:
        completed_at = _now() if now is None else float(now)
        result_payload = _json_safe(result or {}, label="reviewer result")
        result_payload.setdefault("review_status", review_status)
        result_json = _json_dump(result_payload)
        task_state = self._task_state_for_review(review_status)
        with self._transaction():
            job = self._job(job_id)
            if job.kind != JOB_REVIEWER:
                raise ValueError("job is not a reviewer job")
            if job.state in TERMINAL_JOB_STATES:
                self._claim_token_matches(
                    job,
                    lease_owner=lease_owner,
                    attempt=attempt,
                    require_running=False,
                )
                return job
            self._claim_token_matches(job, lease_owner=lease_owner, attempt=attempt)
            self._conn.execute(
                """
                UPDATE frontdesk_jobs
                SET state = ?, lease_expires_at = NULL,
                    heartbeat_at = ?, exit_status = ?, result_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    JOB_SUCCEEDED,
                    completed_at,
                    _string_or_none(exit_status),
                    result_json,
                    completed_at,
                    job_id,
                ),
            )
            self._insert_artifacts(
                task_id=job.task_id,
                job_id=job_id,
                artifacts=artifacts or (),
                now=completed_at,
            )
            self._conn.execute(
                "UPDATE frontdesk_tasks SET state = ?, updated_at = ? WHERE id = ?",
                (task_state, completed_at, job.task_id),
            )
            self._insert_event(
                "reviewer_job_completed",
                task_id=job.task_id,
                job_id=job_id,
                payload={"review_status": review_status, "task_state": task_state},
                now=completed_at,
            )
            return self._job(job_id)

    def _task_state_for_review(self, review_status: str) -> str:
        if review_status == REVIEW_PASSED:
            return FRONTDESK_REVIEW_PASSED
        if review_status in {REVIEW_FAILED, REVIEW_NEEDS_ITERATION}:
            return FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION
        if review_status == REVIEW_BLOCKED:
            return FRONTDESK_BLOCKED_USER_INPUT
        if review_status in _REJECTED_REVIEW_STATUSES:
            return FRONTDESK_ERROR
        raise ValueError(f"unknown review status {review_status!r}")

    def mark_done_presented_with_status(
        self,
        task_id: str,
        *,
        now: float | None = None,
    ) -> tuple[FrontdeskTaskRecord, bool]:
        """Mark a review-passed task as presented and report idempotence.

        Returns ``(task, already_presented)``.  The review gate and the
        already-presented no-op are evaluated inside one transaction so concurrent
        callers cannot turn an idempotent presentation retry into a spurious
        ``before review passes`` failure.
        """
        marked_at = _now() if now is None else float(now)
        with self._transaction():
            task = self._task(task_id)
            if task.state == FRONTDESK_DONE_PRESENTED:
                return task, True
            if task.state != FRONTDESK_REVIEW_PASSED:
                raise ValueError("task cannot be presented before review passes")
            self._conn.execute(
                "UPDATE frontdesk_tasks SET state = ?, updated_at = ? WHERE id = ?",
                (FRONTDESK_DONE_PRESENTED, marked_at, task_id),
            )
            self._insert_event("task_done_presented", task_id=task_id, now=marked_at)
            return self._task(task_id), False

    def mark_done_presented(self, task_id: str, *, now: float | None = None) -> FrontdeskTaskRecord:
        """Mark a review-passed task as presented to the user.

        Worker completion cannot call this path.  The state must already be
        ``review_passed`` so a successful worker alone can never become
        user-facing completion.  Repeated calls after presentation are idempotent.
        """
        task, _already_presented = self.mark_done_presented_with_status(task_id, now=now)
        return task

    def record_import_decision(
        self,
        task_id: str,
        *,
        artifact_ids: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[FrontdeskArtifactRecord]:
        """Record that reviewed artifact pointers were selected for import.

        This is deliberately a metadata/status transition only.  It never reads,
        copies, executes, shells out with, or deletes the artifact paths stored in
        the database.  Import is allowed only after review has passed; a worker
        success that is still pending review is not a final importable result.
        """
        decided_at = _now() if now is None else float(now)
        event_metadata = _json_safe(metadata or {}, label="import decision metadata")
        with self._transaction():
            task = self._task(task_id)
            if task.state not in {FRONTDESK_REVIEW_PASSED, FRONTDESK_DONE_PRESENTED}:
                raise ValueError("task cannot be imported before review passes")
            artifacts = self._selected_task_artifacts(task_id, artifact_ids)
            changed = [artifact for artifact in artifacts if artifact.import_status != ARTIFACT_IMPORT_REQUESTED]
            event_type = "task_import_decision_recorded"
            should_record_event = bool(changed) or not self._task_has_event(task_id, event_type)
            if changed:
                self._conn.executemany(
                    "UPDATE frontdesk_artifacts SET import_status = ? WHERE id = ?",
                    [(ARTIFACT_IMPORT_REQUESTED, artifact.id) for artifact in changed],
                )
            if should_record_event:
                self._insert_event(
                    event_type,
                    task_id=task_id,
                    payload={
                        "artifact_ids": [artifact.id for artifact in artifacts],
                        "changed_artifact_ids": [artifact.id for artifact in changed],
                        "import_status": ARTIFACT_IMPORT_REQUESTED,
                        "applied": False,
                        "metadata": event_metadata,
                    },
                    now=decided_at,
                )
            return self._selected_task_artifacts(task_id, [artifact.id for artifact in artifacts])

    def record_discard_decision(
        self,
        task_id: str,
        *,
        artifact_ids: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> list[FrontdeskArtifactRecord]:
        """Record a non-destructive decision to discard artifact pointers.

        Discarding only updates durable metadata and records an event.  Artifact
        files are not opened, copied, modified, or deleted.
        """
        decided_at = _now() if now is None else float(now)
        event_metadata = _json_safe(metadata or {}, label="discard decision metadata")
        with self._transaction():
            task = self._task(task_id)
            if task.state not in _DISCARDABLE_TASK_STATES:
                raise ValueError("task cannot be discarded before review completes")
            artifacts = self._selected_task_artifacts(task_id, artifact_ids)
            changed = [artifact for artifact in artifacts if artifact.import_status != ARTIFACT_DISCARDED]
            event_type = "task_discard_decision_recorded"
            should_record_event = bool(changed) or not self._task_has_event(task_id, event_type)
            if changed:
                self._conn.executemany(
                    "UPDATE frontdesk_artifacts SET import_status = ? WHERE id = ?",
                    [(ARTIFACT_DISCARDED, artifact.id) for artifact in changed],
                )
            if should_record_event:
                self._insert_event(
                    event_type,
                    task_id=task_id,
                    payload={
                        "artifact_ids": [artifact.id for artifact in artifacts],
                        "changed_artifact_ids": [artifact.id for artifact in changed],
                        "import_status": ARTIFACT_DISCARDED,
                        "deleted": False,
                        "metadata": event_metadata,
                    },
                    now=decided_at,
                )
            return self._selected_task_artifacts(task_id, [artifact.id for artifact in artifacts])

    def request_cancel(self, task_id: str, *, reason: str | None = None) -> FrontdeskTaskRecord:
        """Idempotently request cancellation for a task and its live jobs."""
        requested_at = _now()
        with self._transaction():
            task = self._task(task_id)
            if task.state in {FRONTDESK_CANCELLED, FRONTDESK_DONE_PRESENTED}:
                return task
            if task.state == FRONTDESK_CANCEL_REQUESTED:
                return task
            self._conn.execute(
                """
                UPDATE frontdesk_tasks
                SET state = ?, cancel_requested_at = COALESCE(cancel_requested_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (FRONTDESK_CANCELLED, requested_at, requested_at, task_id),
            )
            self._conn.execute(
                """
                UPDATE frontdesk_jobs
                SET state = ?, lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE task_id = ? AND state NOT IN (?, ?, ?)
                """,
                (JOB_CANCELLED, requested_at, task_id, JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED),
            )
            self._insert_event(
                "task_cancel_requested",
                task_id=task_id,
                payload={"reason": reason} if reason else {},
                now=requested_at,
            )
            return self._task(task_id)
