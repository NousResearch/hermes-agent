"""Durable cron failure incidents with signature dedup and ack.

The executions ledger records every attempt; this module groups the *failures* into incidents keyed
by ``(job_id, error signature)`` so the same job failing with the same error does not re-ping the
operator every run once acknowledged. Lifecycle: ``detected`` → ``alerted`` → ``closed``. The same
job + same normalized error resolves to the SAME incident id, so a closed incident stays closed
until the error text changes and mints a new one. ``alerted`` means a failure ping actually reached
the operator. Incidents share ``cron/executions.db`` with ``cron.executions`` (one ledger file).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from cron import executions as _executions
from cron.executions import ledger_transaction, open_ledger, prepare_ledger
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override (mirrors ``cron.executions.EXECUTIONS_FILE``).
EXECUTIONS_FILE: Optional[Path] = None

INCIDENT_STATES = ("detected", "alerted", "closed")
_FAILURE_TYPE_ORDER = (
    ("rate_limit", (r"\b429\b", "rate limit", "usage limit", "quota")),
    ("timeout", ("timeout", "timed out")),
    ("auth", (r"\b401\b", "unauthorized", "authentication", "auth")),
    ("delivery", ("delivery", "deliver", "delivering")),
    ("config", ("config", "configuration", "validation")),
    ("script", ("script", "no_agent")),
    ("agent", ("agent", "model", "provider", "inference")),
)
MAX_ERROR_CHARS = 500
_MAX_SIGNATURE_ERROR_CHARS = 200

_lock = threading.RLock()


def _db_path() -> Path:
    """Shared cron DB path. The ``cron.executions`` override wins when installed so redirecting the
    executions ledger also redirects the incident table (they must stay in the SAME database); then
    this module's own override, then the canonical profile home."""
    for override in (_executions.EXECUTIONS_FILE, EXECUTIONS_FILE):
        if override is not None:
            return Path(override)
    return get_hermes_home().resolve() / "cron" / "executions.db"


def _connect() -> sqlite3.Connection:
    return open_ledger(_db_path())


def _initialize_schema(conn: sqlite3.Connection) -> None:
    prepare_ledger(conn, db_label="cron/executions.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cron_incidents (
             id            TEXT PRIMARY KEY,
             job_id        TEXT NOT NULL,
             error_sig     TEXT NOT NULL,
             state         TEXT NOT NULL,
             failure_type  TEXT NOT NULL DEFAULT 'unknown',
             first_seen_at TEXT NOT NULL,
             last_seen_at  TEXT NOT NULL,
             acked_at      TEXT,
             closed_at     TEXT,
             error         TEXT NOT NULL,
             output_file   TEXT
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cron_incidents_job "
        "ON cron_incidents(job_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cron_incidents_state "
        "ON cron_incidents(state)"
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    with ledger_transaction(_lock, _connect, _initialize_schema) as conn:
        yield conn


def _normalize_error(error: str) -> str:
    """Strip whitespace and lowercase before signing (dedup normalization)."""
    return re.sub(r"\s+", " ", str(error or "")).strip().lower()


def _redact_error(error: str) -> str:
    """Redact secrets (best-effort; the scheduler path never fails on it) then bound the length."""
    text = str(error or "")
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text)
    except Exception:
        pass
    return text[:MAX_ERROR_CHARS]


def _error_signature(job_id: str, error: str) -> str:
    """Dedup key: stable for same job + same normalized error prefix."""
    normalized = _normalize_error(error)[:_MAX_SIGNATURE_ERROR_CHARS]
    return hashlib.sha256(job_id.encode() + normalized.encode()).hexdigest()[:12]


def _incident_id(job_id: str, error_sig: str) -> str:
    return f"{job_id[:6]}_{error_sig}"


def _classify_failure_type(error: str) -> str:
    """Classify a failure from error-text keywords; ``unknown`` is the default."""
    text = _normalize_error(error)
    if not text:
        return "unknown"
    for kind, patterns in _FAILURE_TYPE_ORDER:
        for pattern in patterns:
            if pattern.startswith("\\b") and pattern.endswith("\\b"):
                if re.search(pattern, text):
                    return kind
            elif pattern in text:
                return kind
    return "unknown"


def upsert_incident(
    job_id: str, error: str, *, job_name: Optional[str] = None, failure_type: Optional[str] = None,
    output_file: Optional[str] = None,
) -> tuple[str, bool]:
    """Record (or refresh) the incident for ``job_id`` + ``error``; returns ``(incident_id,
    is_new)``. An existing row for the signature refreshes
    ``last_seen_at``/``error``/``output_file`` and keeps its state — a ``closed`` incident stays
    closed. A changed error text mints a new incident."""
    job_id = str(job_id or "")
    sig = _error_signature(job_id, error)
    stored_error = _redact_error(error)
    incident_id = _incident_id(job_id, sig)
    now = _hermes_now().isoformat()
    failure_type = failure_type or _classify_failure_type(error)
    output_file = str(output_file) if output_file is not None else None

    with _transaction() as conn:
        row = conn.execute(
            "SELECT id FROM cron_incidents WHERE id=?", (incident_id,)
        ).fetchone()
        if row is not None:
            conn.execute(
                """UPDATE cron_incidents
                   SET last_seen_at=?, error=?, output_file=?
                   WHERE id=?""",
                (now, stored_error, output_file, incident_id),
            )
            return incident_id, False
        conn.execute(
            """INSERT INTO cron_incidents
               (id, job_id, error_sig, state, failure_type,
                first_seen_at, last_seen_at, error, output_file)
               VALUES (?, ?, ?, 'detected', ?, ?, ?, ?, ?)""",
            (incident_id, job_id, sig, failure_type, now, now,
             stored_error, output_file),
        )
        return incident_id, True


def set_incident_state(incident_id: str, state: str) -> bool:
    """Transition an incident's lifecycle state; return whether it changed. ``closed`` is terminal
    for that signature (re-open happens by a changed error minting a NEW incident). Unknown states
    are rejected (no-op, ``False``)."""
    if state not in INCIDENT_STATES:
        return False
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = conn.execute(
            "SELECT state FROM cron_incidents WHERE id=?", (incident_id,)
        ).fetchone()
        if row is None or row["state"] in (state, "closed"):
            return False
        if state == "closed":
            conn.execute(
                """UPDATE cron_incidents
                   SET state='closed', closed_at=?, acked_at=?
                   WHERE id=? AND state != 'closed'""",
                (now, now, incident_id),
            )
        else:
            conn.execute(
                "UPDATE cron_incidents SET state=? WHERE id=?",
                (state, incident_id),
            )
        return True


def ack_incident(incident_id: str) -> bool:
    """Acknowledge (close) an incident; ``False`` when missing or already closed."""
    return set_incident_state(incident_id, "closed")


def _state_filter(state: Optional[str]) -> tuple[str, tuple]:
    return ("", ()) if state is None else (" WHERE state=?", (state,))


def list_incidents(state: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return incidents, newest-activity first, optionally filtered by state."""
    if state is not None and state not in INCIDENT_STATES:
        return []
    where, params = _state_filter(state)
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM cron_incidents" + where + " ORDER BY last_seen_at DESC, id DESC", params
        ).fetchall()
    return [dict(row) for row in rows]


def get_incident(incident_id: str) -> Optional[Dict[str, Any]]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM cron_incidents WHERE id=?", (incident_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def count_incidents(state: Optional[str] = None) -> int:
    if state is not None and state not in INCIDENT_STATES:
        return 0
    where, params = _state_filter(state)
    with _transaction() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM cron_incidents" + where, params).fetchone()
    return int(row["n"]) if row is not None else 0
