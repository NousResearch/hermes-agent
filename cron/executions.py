"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

EXECUTIONS_FILE = get_hermes_home().resolve() / "cron" / "executions.db"
_IMPORT_EXECUTIONS_FILE = EXECUTIONS_FILE
MAX_TERMINAL_EXECUTIONS = 1000
_TERMINAL_STATES = ("completed", "failed", "unknown")
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


def _current_executions_file():
    """Resolve the ledger beside the active task-local cron store."""
    if EXECUTIONS_FILE != _IMPORT_EXECUTIONS_FILE:
        return EXECUTIONS_FILE
    try:
        from cron.jobs import _current_cron_store

        return _current_cron_store().cron_dir / "executions.db"
    except Exception:
        return get_hermes_home().resolve() / "cron" / "executions.db"


def _connect() -> sqlite3.Connection:
    path = _current_executions_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=5)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    """Configure and migrate the ledger under a cross-process SQLite lock."""
    from hermes_state import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    apply_wal_with_fallback(conn, db_label="cron/executions.db")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS executions (
                 id TEXT PRIMARY KEY,
                 job_id TEXT NOT NULL,
                 source TEXT NOT NULL,
                 process_id TEXT NOT NULL,
                 pid INTEGER NOT NULL,
                 process_started_at INTEGER,
                 status TEXT NOT NULL CHECK(status IN
                   ('claimed','running','completed','failed','unknown')),
                 claimed_at TEXT NOT NULL,
                 started_at TEXT,
                 finished_at TEXT,
                 error TEXT,
                 session_key TEXT,
                 admitted_binding_version INTEGER,
                 admitted_route_instance_id TEXT,
                 admitted_session_id TEXT,
                 admitted_routing_revision INTEGER,
                 admitted_at TEXT,
                 outcome TEXT,
                 result_json TEXT,
                 phase TEXT,
                 delivery_state TEXT,
                 delivery_claimed_at TEXT,
                 delivery_finished_at TEXT,
                 delivery_error TEXT,
                 retry_count INTEGER DEFAULT 0,
                 next_retry_at TEXT,
                 delivery_target_json TEXT,
                 job_accounted INTEGER DEFAULT 0,
                 requires_job_accounting INTEGER DEFAULT 0,
                 transcript_session_id TEXT,
                 transcript_json TEXT,
                 transcript_state TEXT DEFAULT 'not_applicable',
                 transcript_base_message_count INTEGER,
                 transcript_base_revision INTEGER,
                 transcript_last_prompt_tokens INTEGER
               )"""
        )
        # Additive, versioned migration. BEGIN IMMEDIATE serializes the
        # PRAGMA-table-info → ALTER sequence across gateway/desktop/scheduler
        # processes so two starters cannot race the same ADD COLUMN.
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(executions)")
        }
        for name, sql_type in (
            ("session_key", "TEXT"),
            ("admitted_binding_version", "INTEGER"),
            ("admitted_route_instance_id", "TEXT"),
            ("admitted_session_id", "TEXT"),
            ("admitted_routing_revision", "INTEGER"),
            ("admitted_at", "TEXT"),
            ("outcome", "TEXT"),
            ("result_json", "TEXT"),
            ("phase", "TEXT"),
            ("delivery_state", "TEXT"),
            ("delivery_claimed_at", "TEXT"),
            ("delivery_finished_at", "TEXT"),
            ("delivery_error", "TEXT"),
            ("retry_count", "INTEGER DEFAULT 0"),
            ("next_retry_at", "TEXT"),
            ("delivery_target_json", "TEXT"),
            ("job_accounted", "INTEGER DEFAULT 0"),
            ("requires_job_accounting", "INTEGER DEFAULT 0"),
            ("transcript_session_id", "TEXT"),
            ("transcript_json", "TEXT"),
            ("transcript_state", "TEXT DEFAULT 'not_applicable'"),
            ("transcript_base_message_count", "INTEGER"),
            ("transcript_base_revision", "INTEGER"),
            ("transcript_last_prompt_tokens", "INTEGER"),
        ):
            if name not in columns:
                conn.execute(f"ALTER TABLE executions ADD COLUMN {name} {sql_type}")
        conn.execute(
            """UPDATE executions SET phase=CASE status
                   WHEN 'claimed' THEN 'claimed'
                   WHEN 'running' THEN 'running'
                   WHEN 'completed' THEN 'completed'
                   WHEN 'failed' THEN 'failed'
                   ELSE 'unknown' END
               WHERE phase IS NULL"""
        )
        conn.execute(
            "UPDATE executions SET delivery_state='not_applicable' "
            "WHERE delivery_state IS NULL"
        )
        conn.execute(
            "UPDATE executions SET requires_job_accounting=1 "
            "WHERE delivery_target_json IS NOT NULL"
        )
        conn.execute(
            "UPDATE executions SET transcript_state='not_applicable' "
            "WHERE transcript_state IS NULL"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
            "ON executions(job_id, claimed_at DESC, id DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
            "ON executions(status, claimed_at DESC, id DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_executions_pending_delivery "
            "ON executions(delivery_state, phase, claimed_at)"
        )
        conn.execute("PRAGMA user_version=8")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, always close.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back
    the transaction; it does not close the connection. Relying on that alone
    leaks a connection (and its WAL/SHM file descriptors) on every call,
    since closing then depends on the garbage collector. Schema init runs
    inside the ``try`` too, so a PRAGMA/DDL failure after a successful
    ``connect()`` still closes the connection instead of leaking it.
    """
    with _lock:
        conn = _connect()
        try:
            _initialize_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()


def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[Dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        # The PID exists, but the durable owner fingerprint was unavailable.
        # Treat that ambiguity as live; recovery may rewrite state only after
        # proving owner death or PID reuse.
        return True
    current = _process_start_time(pid)
    if current is None:
        return True
    return current == started_at


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    limit = max(0, int(MAX_TERMINAL_EXECUTIONS))
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
               AND (requires_job_accounting=0 OR job_accounted=1)
               AND COALESCE(transcript_state, 'not_applicable') != 'pending'
             ORDER BY claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (limit,),
    )


def create_execution(
    job_id: str, *, source: str, requires_job_accounting: bool = False
) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, phase, delivery_state,
                requires_job_accounting)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, 'claimed',
                       'not_applicable', ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now, int(requires_job_accounting)),
        )
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
    record = _record(row)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='running', phase='running', started_at=?
               WHERE id=? AND status='claimed'""",
            (now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record)
    return record


def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Return one durable execution occurrence by its stable identity."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (str(execution_id),)
        ).fetchone()
    return _record(row)


def seal_contextual_admission(
    execution_id: str,
    *,
    session_key: str,
    admitted_session_id: str,
    admitted_routing_revision: int = 0,
    admitted_route_instance_id: Optional[str] = None,
    admitted_binding_version: int = 1,
) -> bool:
    """Durably bind an occurrence to the exact live session before queueing.

    The stable routing key remains the job target. The id is an occurrence
    fence only, making the reset-before/reset-after admission boundary durable.
    """
    key = str(session_key or "").strip()
    session_id = str(admitted_session_id or "").strip()
    route_instance_id = str(admitted_route_instance_id or "").strip() or None
    binding_version = int(admitted_binding_version)
    if binding_version not in (1, 2):
        raise ValueError("unsupported contextual binding version")
    if binding_version == 2 and route_instance_id is None:
        raise ValueError("v2 contextual admission requires a route instance")
    if not key or not session_id:
        return False
    admitted_at = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET session_key=?, admitted_binding_version=?,
                   admitted_route_instance_id=?, admitted_session_id=?,
                   admitted_routing_revision=?, admitted_at=?, phase='admitted'
               WHERE id=? AND status IN ('claimed','running')
                 AND session_key IS NULL AND admitted_session_id IS NULL""",
            (
                key,
                binding_version,
                route_instance_id,
                session_id,
                int(admitted_routing_revision),
                admitted_at,
                execution_id,
            ),
        )
        if cur.rowcount == 1:
            return True
        existing = conn.execute(
            """SELECT status, session_key, admitted_binding_version,
                      admitted_route_instance_id,
                      admitted_session_id, admitted_routing_revision
               FROM executions WHERE id=?""",
            (execution_id,),
        ).fetchone()
        return bool(
            existing is not None
            and existing["status"] in {"claimed", "running"}
            and existing["session_key"] == key
            and int(existing["admitted_binding_version"] or 1) == binding_version
            and (existing["admitted_route_instance_id"] or None)
            == route_instance_id
            and existing["admitted_session_id"] == session_id
            and int(existing["admitted_routing_revision"] or 0)
            == int(admitted_routing_revision)
        )


def seal_contextual_delivery_target(
    execution_id: str,
    *,
    target: Dict[str, Any],
) -> bool:
    """Immutably snapshot the destination before contextual execution starts."""
    payload = json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET delivery_target_json=?, requires_job_accounting=1
               WHERE id=? AND status IN ('claimed','running')
                 AND delivery_target_json IS NULL""",
            (payload, str(execution_id)),
        )
        if cur.rowcount == 1:
            return True
        row = conn.execute(
            "SELECT status, delivery_target_json FROM executions WHERE id=?",
            (str(execution_id),),
        ).fetchone()
        return bool(
            row
            and row["status"] in {"claimed", "running"}
            and row["delivery_target_json"] == payload
        )


def claim_contextual_job_accounting(execution_id: str) -> bool:
    """Persist that idempotent job accounting completed for this occurrence."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET job_accounted=1
               WHERE id=? AND COALESCE(job_accounted, 0)=0""",
            (str(execution_id),),
        )
        return cur.rowcount == 1


def list_unaccounted_contextual_executions(*, limit: int = 1000) -> List[Dict[str, Any]]:
    """Return terminal contextual occurrences whose job-file accounting is pending."""
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT * FROM executions
               WHERE delivery_target_json IS NOT NULL
                 AND COALESCE(job_accounted, 0)=0
                 AND status IN ('completed','failed','unknown')
               ORDER BY claimed_at ASC, id ASC
               LIMIT ?""",
            (max(1, min(int(limit), 10000)),),
        ).fetchall()
        return [dict(row) for row in rows]


_CONTEXTUAL_OUTCOMES = {
    "notify",
    "no_action",
    "retryable",
    "rejected",
    "stale",
    "failure",
    "unknown",
}


def persist_contextual_agent_result(
    execution_id: str,
    *,
    outcome: str,
    final_response: str = "",
    error: Optional[str] = None,
    transcript_session_id: Optional[str] = None,
    transcript_entries: Optional[List[Dict[str, Any]]] = None,
    transcript_base_message_count: Optional[int] = None,
    transcript_base_revision: Optional[int] = None,
    transcript_last_prompt_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Durably persist the agent result before any external delivery attempt.

    ``notify`` remains non-terminal in ``agent_completed/pending``. All other
    outcomes are terminal except ``retryable``, which stays in ``retry_wait``
    for bounded scheduler-owned re-dispatch of the same occurrence.
    """
    kind = str(outcome or "unknown")
    if kind not in _CONTEXTUAL_OUTCOMES:
        kind = "unknown"
        error = error or "Gateway returned an unrecognized contextual cron outcome."
    response = str(final_response or "")
    if kind == "notify" and not response.strip():
        kind = "failure"
        error = error or "Contextual cron notify outcome had no final response."

    if kind == "notify":
        status, phase, delivery_state, finished_at = (
            "running",
            "agent_completed",
            "pending",
            None,
        )
        detail = None
    elif kind == "retryable":
        status, phase, delivery_state, finished_at = (
            "running",
            "retry_wait",
            "not_applicable",
            None,
        )
        detail = str(error or "Contextual cron requested a retry.")
    elif kind == "no_action":
        status, phase, delivery_state = "completed", "completed", "not_applicable"
        finished_at = _hermes_now().isoformat()
        detail = None
    elif kind == "unknown":
        status, phase, delivery_state = "unknown", "unknown", "unknown"
        finished_at = _hermes_now().isoformat()
        detail = str(error or "Contextual cron terminal state is unknown.")
    else:
        status, phase, delivery_state = "failed", "failed", "not_applicable"
        finished_at = _hermes_now().isoformat()
        detail = str(error or f"Contextual cron ended with {kind}.")

    result_json = json.dumps(
        {"kind": kind, "final_response": response, "error": error},
        ensure_ascii=False,
        sort_keys=True,
    )
    transcript_payload: Optional[str] = None
    transcript_session: Optional[str] = None
    if transcript_entries is not None or transcript_session_id is not None:
        transcript_session = str(transcript_session_id or "").strip()
        if not transcript_session or transcript_entries is None:
            raise ValueError(
                "contextual transcript outbox requires a session id and entries"
            )
        transcript_payload = json.dumps(
            transcript_entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            not isinstance(transcript_base_message_count, int)
            or isinstance(transcript_base_message_count, bool)
            or transcript_base_message_count < 0
        ):
            raise ValueError(
                "contextual transcript outbox requires a valid base message count"
            )
        transcript_base_message_count = int(transcript_base_message_count)
        if (
            not isinstance(transcript_base_revision, int)
            or isinstance(transcript_base_revision, bool)
            or transcript_base_revision < 0
        ):
            raise ValueError(
                "contextual transcript outbox requires a valid base revision"
            )
        transcript_base_revision = int(transcript_base_revision)
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, phase=?, delivery_state=?, finished_at=?, error=?,
                   outcome=?, result_json=?, next_retry_at=NULL,
                   transcript_session_id=COALESCE(?, transcript_session_id),
                   transcript_json=COALESCE(?, transcript_json),
                   transcript_state=CASE WHEN ? IS NULL THEN transcript_state
                                         ELSE 'pending' END,
                   transcript_base_message_count=
                       COALESCE(?, transcript_base_message_count),
                   transcript_base_revision=
                       COALESCE(?, transcript_base_revision),
                   transcript_last_prompt_tokens=
                       COALESCE(?, transcript_last_prompt_tokens)
               WHERE id=? AND status IN ('claimed','running')
                 AND COALESCE(phase, status) IN
                   ('claimed','running','admitted','retry_wait')""",
            (
                status,
                phase,
                delivery_state,
                finished_at,
                detail,
                kind,
                result_json,
                transcript_session,
                transcript_payload,
                transcript_payload,
                transcript_base_message_count,
                transcript_base_revision,
                (
                    int(transcript_last_prompt_tokens)
                    if transcript_last_prompt_tokens is not None
                    else None
                ),
                execution_id,
            ),
        )
        record = _record(
            conn.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
        )
        if cur.rowcount != 1:
            # Idempotent join: gateway and scheduler may both acknowledge the
            # same result, but neither may rewrite a different terminal value.
            if not record or record.get("outcome") != kind:
                return None
            if record.get("result_json") != result_json:
                return None
            if transcript_payload is not None and (
                record.get("transcript_session_id") != transcript_session
                or record.get("transcript_json") != transcript_payload
                or record.get("transcript_base_message_count")
                != transcript_base_message_count
                or record.get("transcript_base_revision")
                != transcript_base_revision
            ):
                return None
            if (
                transcript_last_prompt_tokens is not None
                and record.get("transcript_last_prompt_tokens")
                != int(transcript_last_prompt_tokens)
            ):
                return None
        if record and record.get("status") in _TERMINAL_STATES:
            _prune_unlocked(conn)
    _emit_execution_state(record)
    return record


def list_pending_contextual_transcripts(*, limit: int = 1000) -> List[Dict[str, Any]]:
    """Return durable transcript outboxes that still need idempotent application."""
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT * FROM executions
               WHERE transcript_state='pending'
                 AND transcript_session_id IS NOT NULL
                 AND transcript_json IS NOT NULL
               ORDER BY claimed_at ASC, id ASC
               LIMIT ?""",
            (max(1, min(int(limit), 10000)),),
        ).fetchall()
        return [dict(row) for row in rows]


def mark_contextual_transcript_conflict(
    execution_id: str, *, error: str
) -> bool:
    """Terminalize a pending outbox that lost its immutable causal position."""
    now = _hermes_now().isoformat()
    detail = str(error or "Contextual transcript causal position is unknown.")
    result_json = json.dumps(
        {"kind": "unknown", "final_response": "", "error": detail},
        ensure_ascii=False,
        sort_keys=True,
    )
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='unknown', phase='unknown', outcome='unknown',
                   result_json=?, error=?, finished_at=?,
                   transcript_state='conflict', delivery_state='unknown',
                   delivery_finished_at=?, delivery_error=?
               WHERE id=? AND transcript_state='pending'
                 AND status IN ('claimed','running','completed','failed','unknown')""",
            (result_json, detail, now, now, detail, str(execution_id)),
        )
        if cur.rowcount == 1:
            _prune_unlocked(conn)
            record = _record(
                conn.execute(
                    "SELECT * FROM executions WHERE id=?", (str(execution_id),)
                ).fetchone()
            )
        else:
            record = None
    if record is not None:
        _emit_execution_state(record, delivery_outcome="unknown")
    return cur.rowcount == 1


def mark_contextual_transcript_applied(execution_id: str) -> bool:
    """Acknowledge one fully applied transcript outbox exactly once."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET transcript_state='applied'
               WHERE id=? AND transcript_state='pending'""",
            (str(execution_id),),
        )
        if cur.rowcount == 1:
            _prune_unlocked(conn)
        return cur.rowcount == 1


def prepare_contextual_retry(execution_id: str) -> Optional[Dict[str, Any]]:
    """Re-arm the same admitted occurrence after a typed transient outcome."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET phase=CASE WHEN admitted_session_id IS NULL
                              THEN 'running' ELSE 'admitted' END,
                   outcome=NULL, result_json=NULL, error=NULL,
                   delivery_state='not_applicable', retry_count=retry_count+1,
                   next_retry_at=NULL
               WHERE id=? AND status='running' AND phase='retry_wait'
                 AND outcome='retryable'""",
            (execution_id,),
        )
        if cur.rowcount != 1:
            return None
        return _record(
            conn.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
        )


def claim_contextual_delivery(execution_id: str) -> Optional[Dict[str, Any]]:
    """CAS-claim the one scheduler-owned external delivery attempt."""
    now = _hermes_now().isoformat()
    pid = os.getpid()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET phase='delivering', delivery_state='claimed',
                   delivery_claimed_at=?, process_id=?, pid=?,
                   process_started_at=?
               WHERE id=? AND status='running' AND phase='agent_completed'
                 AND delivery_state='pending' AND outcome='notify'
                 AND COALESCE(transcript_state, 'not_applicable') != 'pending'""",
            (
                now,
                _PROCESS_ID,
                pid,
                _process_start_time(pid),
                execution_id,
            ),
        )
        if cur.rowcount != 1:
            return None
        record = _record(
            conn.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
        )
    _emit_execution_state(record, delivery_outcome="claimed")
    return record


def finish_contextual_delivery(
    execution_id: str,
    *,
    delivery_state: str,
    error: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Durably acknowledge the claimed delivery without automatic resend."""
    state = str(delivery_state or "unknown")
    if state not in {"sent", "failed", "unknown"}:
        state = "unknown"
        error = error or "Unrecognized contextual delivery result."
    if state == "sent":
        status, phase, detail = "completed", "completed", None
    elif state == "failed":
        status, phase = "failed", "failed"
        detail = str(error or "Contextual cron delivery failed.")
    else:
        status, phase = "unknown", "unknown"
        detail = str(
            error
            or "Contextual cron delivery may have occurred without durable acknowledgement."
        )
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, phase=?, delivery_state=?,
                   delivery_finished_at=?, finished_at=?,
                   delivery_error=?, error=?
               WHERE id=? AND status='running' AND phase='delivering'
                 AND delivery_state='claimed'""",
            (
                status,
                phase,
                state,
                now,
                now,
                detail,
                detail,
                execution_id,
            ),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _record(
            conn.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
        )
    _emit_execution_state(record, delivery_outcome=state)
    return record


def suppress_contextual_delivery(
    execution_id: str, *, error: str
) -> Optional[Dict[str, Any]]:
    """Fail a pending delivery before ownership is claimed or transport begins."""
    now = _hermes_now().isoformat()
    detail = str(error or "Contextual cron delivery was suppressed.")
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='failed', phase='failed', delivery_state='failed',
                   delivery_finished_at=?, finished_at=?,
                   delivery_error=?, error=?
               WHERE id=? AND status='running' AND phase='agent_completed'
                 AND delivery_state='pending'
                 AND COALESCE(transcript_state, 'not_applicable') != 'pending'""",
            (now, now, detail, detail, execution_id),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _record(
            conn.execute(
                "SELECT * FROM executions WHERE id=?", (execution_id,)
            ).fetchone()
        )
    _emit_execution_state(record, delivery_outcome="suppressed")
    return record


def list_pending_contextual_deliveries() -> List[Dict[str, Any]]:
    """Return agent-complete occurrences that have not begun delivery."""
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT * FROM executions
               WHERE status='running' AND phase='agent_completed'
                 AND delivery_state='pending' AND outcome='notify'
                 AND COALESCE(transcript_state, 'not_applicable') != 'pending'
               ORDER BY claimed_at, id"""
        ).fetchall()
    return [dict(row) for row in rows]


def finish_contextual_execution(
    execution_id: str,
    *,
    outcome: str,
    final_response: str = "",
    error: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compatibility wrapper around result persistence and delivery ack."""
    record = persist_contextual_agent_result(
        execution_id,
        outcome=outcome,
        final_response=final_response,
        error=error,
    )
    if record is None or record.get("outcome") != "notify" or not delivery_outcome:
        return record
    claimed = claim_contextual_delivery(execution_id)
    if claimed is None:
        return get_execution(execution_id)
    if delivery_outcome == "failed":
        return finish_contextual_delivery(
            execution_id,
            delivery_state="failed",
            error=error or "Contextual cron delivery failed.",
        )
    return finish_contextual_delivery(execution_id, delivery_state="sent")


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten."""
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, phase=?, delivery_state='not_applicable',
                   finished_at=?, error=?
               WHERE id=? AND status IN ('claimed','running')""",
            (status, status, now, detail, execution_id),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    return record


def recover_interrupted_executions() -> int:
    """Terminalize abandoned attempts without consuming unclaimed occurrences."""
    now = _hermes_now().isoformat()
    recovery_error = (
        "Scheduler restarted after this execution's owner exited before a durable "
        "terminal state; whether side effects ran is unknown."
    )
    changed = 0
    recovered: List[Dict[str, Any]] = []

    # Never acquire the jobs-file lock while holding the execution DB write
    # transaction. Accounting takes those authorities in the opposite order.
    with _transaction() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """SELECT * FROM executions
                   WHERE status IN ('claimed','running')"""
            ).fetchall()
        ]

    candidates: List[tuple[Dict[str, Any], bool]] = []
    for row in rows:
        if row["process_id"] == _PROCESS_ID:
            continue
        if _owner_is_live(int(row["pid"]), row["process_started_at"]):
            continue
        if (
            row["phase"] == "agent_completed"
            and row["delivery_state"] == "pending"
            and row["outcome"] == "notify"
        ):
            continue

        before_occurrence_claim = False
        if (
            row["status"] == "claimed"
            and row["phase"] == "claimed"
            and bool(row["requires_job_accounting"])
        ):
            from cron.jobs import contextual_occurrence_claim_state

            claim_state = contextual_occurrence_claim_state(
                str(row["job_id"]), execution_id=str(row["id"])
            )
            if claim_state is None:
                logger.error(
                    "Deferred recovery for contextual execution %s because "
                    "the jobs authority lock is unavailable",
                    row["id"],
                )
                continue
            before_occurrence_claim = claim_state is False
        candidates.append((row, before_occurrence_claim))

    with _transaction() as conn:
        for row, before_occurrence_claim in candidates:
            if before_occurrence_claim:
                row_error = (
                    "Scheduler owner exited before this occurrence acquired its "
                    "jobs-store claim; the occurrence remains eligible to run."
                )
                row_result = json.dumps(
                    {"kind": "rejected", "final_response": "", "error": row_error},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                cur = conn.execute(
                    """UPDATE executions
                       SET status='failed', finished_at=?, error=?,
                           outcome='rejected', result_json=?, phase='rejected',
                           delivery_state='not_applicable', job_accounted=1
                       WHERE id=? AND status='claimed' AND phase='claimed'""",
                    (now, row_error, row_result, row["id"]),
                )
            else:
                delivery_uncertain = (
                    row["phase"] == "delivering"
                    or row["delivery_state"] == "claimed"
                )
                row_error = (
                    "Scheduler restarted after a contextual delivery began; the "
                    "message may have been sent without durable acknowledgement."
                    if delivery_uncertain
                    else recovery_error
                )
                row_result = json.dumps(
                    {"kind": "unknown", "final_response": "", "error": row_error},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                cur = conn.execute(
                    """UPDATE executions
                       SET status='unknown', finished_at=?, error=?,
                           outcome='unknown', result_json=?, phase='unknown',
                           delivery_state=CASE WHEN ? THEN 'unknown'
                                               ELSE delivery_state END,
                           delivery_finished_at=CASE WHEN ? THEN ?
                                                      ELSE delivery_finished_at END,
                           delivery_error=CASE WHEN ? THEN ?
                                               ELSE delivery_error END
                       WHERE id=? AND status IN ('claimed','running')""",
                    (
                        now,
                        row_error,
                        row_result,
                        delivery_uncertain,
                        delivery_uncertain,
                        now,
                        delivery_uncertain,
                        row_error,
                        row["id"],
                    ),
                )
            changed += cur.rowcount
            if cur.rowcount:
                record = _record(
                    conn.execute(
                        "SELECT * FROM executions WHERE id=?", (row["id"],)
                    ).fetchone()
                )
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50,
    before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(str(before_claimed_at))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
