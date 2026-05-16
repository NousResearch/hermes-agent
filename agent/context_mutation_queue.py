"""Durable opt-in mutation queue for DAG context maintenance.

The queue uses the existing ``context_mutation_log`` table as a local durable job
log.  It is intentionally inert unless callers explicitly enqueue and run a
worker/sidecar entrypoint; no daemon is started by importing this module.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import socket
import time
from typing import Any, Callable, Dict, Iterable, Optional

from agent.context_dag_models import MutationLogEntry
from agent.context_dag_store import ContextDAGStore, _json_dumps, _json_loads
from agent.context_dag_reconcile import reconcile_full_transcript

QUEUE_STATUSES = {"queued", "running", "succeeded", "dead"}
RETRYABLE_STATUSES = {"queued"}
Handler = Callable[[MutationLogEntry], Any]


class ContextMutationQueue:
    """Small SQLite-backed per-session DAG mutation queue."""

    def __init__(self, store: ContextDAGStore):
        self.store = store
        self.db = store.db

    @staticmethod
    def default_idempotency_key(session_id: str, operation: str, payload: Dict[str, Any]) -> str:
        digest = hashlib.sha256(_json_dumps(payload or {}).encode("utf-8")).hexdigest()
        return f"{session_id}:{operation}:{digest}"

    def _row_to_mutation(self, row) -> MutationLogEntry:
        return ContextDAGStore._row_to_mutation(row)

    def enqueue(
        self,
        session_id: str,
        operation: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        idempotency_key: Optional[str] = None,
        max_attempts: int = 3,
        available_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MutationLogEntry:
        """Insert a queued mutation if its idempotency key is not present."""

        if not session_id:
            raise ValueError("session_id is required")
        if not operation:
            raise ValueError("operation is required")
        payload = dict(payload or {})
        max_attempts = max(1, int(max_attempts or 1))
        key = idempotency_key or self.default_idempotency_key(session_id, operation, payload)
        now = time.time()
        effective_available_at = float(available_at if available_at is not None else now)
        metadata_json = _json_dumps(metadata or {})
        payload_json = _json_dumps(payload)

        def _do(conn):
            conn.execute(
                """
                INSERT INTO context_mutation_log (
                    session_id, operation, status, idempotency_key, payload_json,
                    error, attempts, max_attempts, claimed_at, claimed_by,
                    available_at, created_at, updated_at, metadata_json
                ) VALUES (?, ?, 'queued', ?, ?, NULL, 0, ?, NULL, NULL, ?, ?, ?, ?)
                ON CONFLICT(session_id, idempotency_key) DO NOTHING
                """,
                (
                    session_id,
                    operation,
                    key,
                    payload_json,
                    max_attempts,
                    effective_available_at,
                    now,
                    now,
                    metadata_json,
                ),
            )
            return conn.execute(
                "SELECT * FROM context_mutation_log WHERE session_id = ? AND idempotency_key = ?",
                (session_id, key),
            ).fetchone()

        return self._row_to_mutation(self.db._execute_write(_do))

    def get(self, job_id: int) -> Optional[MutationLogEntry]:
        with self.db._lock:
            row = self.db._conn.execute("SELECT * FROM context_mutation_log WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_mutation(row) if row else None

    def list_jobs(self, *, session_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50) -> list[MutationLogEntry]:
        clauses = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, int(limit or 50)))
        with self.db._lock:
            rows = self.db._conn.execute(
                f"SELECT * FROM context_mutation_log {where} ORDER BY created_at, id LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._row_to_mutation(row) for row in rows]

    def claim_next(self, *, worker_id: Optional[str] = None, now: Optional[float] = None) -> Optional[MutationLogEntry]:
        """Atomically claim the next queued job, serializing by session.

        A queued job is claimable only when no job for the same session is
        currently ``running``.  This gives the PR7 per-session mutation lock
        without any external service.
        """

        ts = float(now if now is not None else time.time())
        worker_id = worker_id or f"{socket.gethostname()}:{id(self)}"

        def _do(conn):
            row = conn.execute(
                """
                SELECT * FROM context_mutation_log q
                WHERE q.status = 'queued'
                  AND COALESCE(q.available_at, q.created_at) <= ?
                  AND q.attempts < q.max_attempts
                  AND NOT EXISTS (
                      SELECT 1 FROM context_mutation_log r
                      WHERE r.session_id = q.session_id AND r.status = 'running'
                  )
                ORDER BY q.created_at, q.id
                LIMIT 1
                """,
                (ts,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE context_mutation_log
                SET status = 'running', attempts = attempts + 1,
                    claimed_at = ?, claimed_by = ?, updated_at = ?, error = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (ts, worker_id, ts, row["id"]),
            )
            return conn.execute("SELECT * FROM context_mutation_log WHERE id = ?", (row["id"],)).fetchone()

        row = self.db._execute_write(_do)
        return self._row_to_mutation(row) if row else None

    def complete(self, job_id: int, *, result: Any = None) -> MutationLogEntry:
        now = time.time()

        def _do(conn):
            row = conn.execute("SELECT metadata_json FROM context_mutation_log WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"mutation job {job_id} not found")
            metadata = _json_loads(row["metadata_json"], {})
            if result is not None:
                metadata["result"] = result
            conn.execute(
                """
                UPDATE context_mutation_log
                SET status = 'succeeded', error = NULL, claimed_at = NULL,
                    claimed_by = NULL, updated_at = ?, metadata_json = ?
                WHERE id = ?
                """,
                (now, _json_dumps(metadata), job_id),
            )
            return conn.execute("SELECT * FROM context_mutation_log WHERE id = ?", (job_id,)).fetchone()

        return self._row_to_mutation(self.db._execute_write(_do))

    def fail(self, job_id: int, error: str, *, retry_delay: float = 0) -> MutationLogEntry:
        now = time.time()

        def _do(conn):
            row = conn.execute("SELECT attempts, max_attempts FROM context_mutation_log WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"mutation job {job_id} not found")
            status = "dead" if int(row["attempts"] or 0) >= int(row["max_attempts"] or 1) else "queued"
            conn.execute(
                """
                UPDATE context_mutation_log
                SET status = ?, error = ?, claimed_at = NULL, claimed_by = NULL,
                    available_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, str(error), now + max(0.0, float(retry_delay or 0)), now, job_id),
            )
            return conn.execute("SELECT * FROM context_mutation_log WHERE id = ?", (job_id,)).fetchone()

        return self._row_to_mutation(self.db._execute_write(_do))


def default_dag_mutation_handlers(store: ContextDAGStore) -> Dict[str, Handler]:
    """Return local, no-LLM default handlers for supported PR7 job types.

    ``compact_leaf`` remains dependency-injected because it needs a summarizer;
    callers should provide their own fake/real handler explicitly.  The included
    handlers cover safe reconciliation and projection persistence writes.
    """

    def _reconcile(job: MutationLogEntry) -> Dict[str, Any]:
        result = reconcile_full_transcript(
            store,
            job.session_id,
            job.payload.get("messages") or [],
            source=job.payload.get("source") or "mutation_queue",
        )
        return {
            "scanned": result.scanned,
            "inserted": result.inserted,
            "duplicates_skipped": result.duplicates_skipped,
            "matched": result.matched,
            "checkpoint_advanced": result.checkpoint_advanced,
            "last_ingested_message_id": result.last_ingested_message_id,
            "warnings": result.warnings,
        }

    def _rebuild_projection(job: MutationLogEntry) -> Dict[str, Any]:
        projection = store.write_active_projection(
            session_id=job.session_id,
            engine_version=job.payload.get("engine_version") or "dag-v1",
            projection=job.payload.get("projection") or [],
            fresh_tail_start_message_id=job.payload.get("fresh_tail_start_message_id"),
            latest_raw_message_id=job.payload.get("latest_raw_message_id"),
            token_estimate=job.payload.get("token_estimate"),
            metadata=job.payload.get("metadata") or {"source": "mutation_queue"},
        )
        return {"projection_id": projection.id, "status": projection.status}

    return {
        "reconcile_transcript": _reconcile,
        "rebuild_projection": _rebuild_projection,
    }


class MutationWorker:
    """Explicit sidecar/worker runner for a bounded number of DAG mutations."""

    def __init__(
        self,
        queue: ContextMutationQueue,
        *,
        handlers: Optional[Dict[str, Handler]] = None,
        worker_id: Optional[str] = None,
        retry_delay: float = 0,
    ) -> None:
        self.queue = queue
        self.handlers = dict(handlers or {})
        self.worker_id = worker_id or f"mutation-worker:{socket.gethostname()}"
        self.retry_delay = retry_delay

    def run_once(self) -> Optional[MutationLogEntry]:
        job = self.queue.claim_next(worker_id=self.worker_id)
        if job is None:
            return None
        handler = self.handlers.get(job.operation)
        if handler is None:
            return self.queue.fail(job.id, f"No handler registered for DAG mutation operation {job.operation!r}", retry_delay=0)
        try:
            result = handler(job)
        except Exception as exc:
            return self.queue.fail(job.id, str(exc), retry_delay=self.retry_delay)
        return self.queue.complete(job.id, result=result)

    def run(self, *, limit: int = 1) -> list[MutationLogEntry]:
        processed: list[MutationLogEntry] = []
        for _ in range(max(1, int(limit or 1))):
            result = self.run_once()
            if result is None:
                break
            processed.append(result)
        return processed


def process_next_mutation(
    queue: ContextMutationQueue,
    *,
    enabled: bool = False,
    handlers: Optional[Dict[str, Handler]] = None,
    worker_id: Optional[str] = None,
) -> Optional[MutationLogEntry]:
    """Opt-in sidecar entrypoint; disabled by default for safety."""

    if not enabled:
        return None
    return MutationWorker(queue, handlers=handlers, worker_id=worker_id).run_once()
