"""Job-scoped native outbound messages for opted-in cron runs.

This is the HERMES-022 ledger. Cron agents never send through a shell or
provider CLI. They enqueue or execute outbound intents that the scheduler
owns, then the existing send engine delivers through the active Hermes
adapter identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

OUTBOUND_FILE = get_hermes_home().resolve() / "cron" / "outbound.db"
_MESSAGE_KEY_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,200}$")
_lock = threading.RLock()


def _connect() -> sqlite3.Connection:
    OUTBOUND_FILE.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(OUTBOUND_FILE, timeout=5)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    apply_wal_with_fallback(conn, db_label="cron/outbound.db")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS outbound_messages (
             job_id TEXT NOT NULL,
             run_id TEXT NOT NULL,
             message_key TEXT NOT NULL,
             target TEXT NOT NULL,
             body_hash TEXT NOT NULL,
             status TEXT NOT NULL CHECK(status IN
               ('queued','sent','verified','ambiguous','failed')),
             platform TEXT,
             chat_id TEXT,
             thread_id TEXT,
             transport_message_id TEXT,
             error TEXT,
             created_at TEXT NOT NULL,
             updated_at TEXT NOT NULL,
             PRIMARY KEY (job_id, run_id, message_key)
           )"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    with _lock:
        conn = _connect()
        try:
            _initialize_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()


def normalize_message_key(message_key: Any) -> str:
    key = str(message_key or "").strip()
    if not key or not _MESSAGE_KEY_RE.fullmatch(key):
        raise ValueError(
            "message_key must be 1-200 characters using letters, digits, "
            "'.', '_', ':', '/', or '-'."
        )
    return key


def hash_body(body: str) -> str:
    return hashlib.sha256((body or "").encode("utf-8")).hexdigest()


def job_allows_messaging(job: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(job, dict):
        return False
    return bool(job.get("allow_messaging"))


def current_cron_run_id() -> str:
    from gateway.session_context import get_session_env

    return str(get_session_env("HERMES_CRON_RUN_ID", "") or "").strip()


def current_cron_job_id() -> str:
    from gateway.session_context import get_session_env

    return str(get_session_env("HERMES_CRON_JOB_ID", "") or "").strip()


def is_cron_messaging_session() -> bool:
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_CRON_ALLOW_MESSAGING", "") == "1"


def get_record(job_id: str, run_id: str, message_key: str) -> Optional[Dict[str, Any]]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM outbound_messages WHERE job_id=? AND run_id=? AND message_key=?",
            (job_id, run_id, message_key),
        ).fetchone()
    return dict(row) if row else None


def claim_or_reuse(
    *,
    job_id: str,
    run_id: str,
    message_key: str,
    target: str,
    body: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
) -> Dict[str, Any]:
    """Atomically claim a send or return a previous terminal/in-flight record."""
    now = _hermes_now().isoformat()
    body_hash = hash_body(body)
    with _transaction() as conn:
        existing = conn.execute(
            "SELECT * FROM outbound_messages WHERE job_id=? AND run_id=? AND message_key=?",
            (job_id, run_id, message_key),
        ).fetchone()
        if existing:
            record = dict(existing)
            if record["body_hash"] != body_hash or record["target"] != target:
                raise ValueError(
                    f"message_key '{message_key}' was already used in this run "
                    "with a different body or target."
                )
            if record["status"] == "failed":
                conn.execute(
                    """UPDATE outbound_messages
                       SET status='queued', error=NULL, updated_at=?
                       WHERE job_id=? AND run_id=? AND message_key=?""",
                    (now, job_id, run_id, message_key),
                )
                row = conn.execute(
                    "SELECT * FROM outbound_messages WHERE job_id=? AND run_id=? AND message_key=?",
                    (job_id, run_id, message_key),
                ).fetchone()
                return {"action": "claim", "record": dict(row)}
            return {"action": "reuse", "record": record}
        conn.execute(
            """INSERT INTO outbound_messages (
                   job_id, run_id, message_key, target, body_hash, status,
                   platform, chat_id, thread_id, transport_message_id, error,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, NULL, NULL, ?, ?)""",
            (
                job_id,
                run_id,
                message_key,
                target,
                body_hash,
                platform,
                chat_id,
                thread_id,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM outbound_messages WHERE job_id=? AND run_id=? AND message_key=?",
            (job_id, run_id, message_key),
        ).fetchone()
    return {"action": "claim", "record": dict(row)}


def mark_result(
    *,
    job_id: str,
    run_id: str,
    message_key: str,
    status: str,
    transport_message_id: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    if status not in {"sent", "verified", "ambiguous", "failed"}:
        raise ValueError(f"unsupported outbound status: {status}")
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        conn.execute(
            """UPDATE outbound_messages
               SET status=?, transport_message_id=?, error=?, updated_at=?
               WHERE job_id=? AND run_id=? AND message_key=? AND status='queued'""",
            (
                status,
                transport_message_id,
                error,
                now,
                job_id,
                run_id,
                message_key,
            ),
        )
        row = conn.execute(
            "SELECT * FROM outbound_messages WHERE job_id=? AND run_id=? AND message_key=?",
            (job_id, run_id, message_key),
        ).fetchone()
    if not row:
        raise LookupError("outbound message record disappeared")
    # A send attempt has exactly one terminal result.  If a late callback or
    # duplicate completion races this update, retain the first result rather
    # than allowing a verified send to be downgraded to failed/ambiguous.
    return dict(row)


def count_successful(job_id: str, run_id: str) -> int:
    with _transaction() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM outbound_messages
               WHERE job_id=? AND run_id=? AND status IN ('sent','verified')""",
            (job_id, run_id),
        ).fetchone()
    return int(row["n"] if row else 0)


def classify_send_result(result: Any) -> Dict[str, Any]:
    """Map a send-engine result onto the outbound ledger states."""
    if not isinstance(result, dict):
        return {
            "status": "ambiguous",
            "transport_message_id": None,
            "error": "send engine returned a non-dict result",
        }
    if result.get("error"):
        return {
            "status": "failed",
            "transport_message_id": result.get("message_id"),
            "error": str(result.get("error")),
        }
    if result.get("success") and result.get("message_id"):
        return {
            "status": "verified",
            "transport_message_id": str(result.get("message_id")),
            "error": None,
        }
    if result.get("success"):
        return {
            "status": "sent",
            "transport_message_id": None,
            "error": None,
        }
    return {
        "status": "ambiguous",
        "transport_message_id": result.get("message_id"),
        "error": "send engine returned an unconfirmed result",
    }


def reuse_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": record.get("status") in {"sent", "verified"},
        "status": record.get("status"),
        "skipped": True,
        "reason": "cron_outbound_idempotent_reuse",
        "job_id": record.get("job_id"),
        "run_id": record.get("run_id"),
        "message_key": record.get("message_key"),
        "target": record.get("target"),
        "message_id": record.get("transport_message_id"),
        "error": record.get("error"),
        "note": (
            f"Reused outbound message '{record.get('message_key')}' "
            f"already recorded as {record.get('status')}."
        ),
    }


def success_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": record.get("status") in {"sent", "verified"},
        "status": record.get("status"),
        "job_id": record.get("job_id"),
        "run_id": record.get("run_id"),
        "message_key": record.get("message_key"),
        "target": record.get("target"),
        "message_id": record.get("transport_message_id"),
        "error": record.get("error"),
    }


def dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
