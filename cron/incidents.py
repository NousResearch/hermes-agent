"""Durable cron failure incidents with signature dedup and ack.

The executions ledger (``cron.executions``) records every attempt; this module
groups the *failures* into durable incidents keyed by ``(job_id, error
signature)`` so the same job failing with the same error does not re-ping the
operator every run once they have acknowledged it.

Lifecycle: ``detected`` → ``alerted`` → ``closed``. Closing
(acking) an incident is per-signature: the same job + same normalized error
keeps resolving to the SAME incident id, so a closed incident stays closed (no
re-alert) until the error text changes, which mints a brand-new incident.
``detected`` means the failure was recorded; ``alerted`` means at least one
failure ping for the signature actually reached the operator. Richer states
(e.g. a dv9.6 ``reviewed``) are deliberately NOT reserved here — state
validity lives in ``INCIDENT_STATES`` (Python), not a SQLite CHECK, exactly
so a future slice can add states without a table rebuild.

Incidents live in the SAME ``cron/executions.db`` as ``cron.executions`` so
there is one durable cron store per profile. The schema is lazily created on
connect and a missing database never raises (directories are created).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=5)


def _db_path() -> Path:
    """Resolve the shared cron DB path.

    Prefer the ``cron.executions`` override when one is installed so an
    operator/test that redirects the executions ledger also redirects the
    incident table — they must stay in the SAME database. Falls back to this
    module's own override, then the canonical profile home.
    """
    try:
        from cron.executions import EXECUTIONS_FILE as _EXEC_OVERRIDE

        if _EXEC_OVERRIDE is not None:
            return Path(_EXEC_OVERRIDE)
    except Exception:
        pass
    if EXECUTIONS_FILE is not None:
        return Path(EXECUTIONS_FILE)
    return get_hermes_home().resolve() / "cron" / "executions.db"


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    apply_wal_with_fallback(conn, db_label="cron/executions.db")
    conn.execute("PRAGMA synchronous=FULL")
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
    """Open a connection, commit/rollback on exit, always close.

    Mirrors ``cron.executions._transaction``: schema init runs inside the
    ``try`` so a PRAGMA/DDL failure after a successful ``connect()`` still
    closes the connection instead of leaking it.
    """
    with _lock:
        conn = _connect()
        try:
            _initialize_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()


def _normalize_error(error: str) -> str:
    """Strip whitespace and lowercase before signing (dedup normalization)."""
    return re.sub(r"\s+", " ", str(error or "")).strip().lower()


def _redact_error(error: str) -> str:
    """Redact secrets then bound the stored error length."""
    text = str(error or "")
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text)
    except Exception:
        # Redaction is best-effort; the scheduler path never fails on it.
        pass
    return text[:MAX_ERROR_CHARS]


def _error_signature(job_id: str, error: str) -> str:
    """Dedup key: stable for same job + same normalized error prefix."""
    normalized = _normalize_error(error)[:_MAX_SIGNATURE_ERROR_CHARS]
    digest = hashlib.sha256(job_id.encode() + normalized.encode()).hexdigest()
    return digest[:12]


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
    job_id: str,
    error: str,
    *,
    job_name: Optional[str] = None,
    failure_type: Optional[str] = None,
    output_file: Optional[str] = None,
) -> tuple[str, bool]:
    """Record (or refresh) the incident for ``job_id`` + ``error``.

    Returns ``(incident_id, is_new)``. A row for the same signature already
    existing refreshes ``last_seen_at``/``error``/``output_file`` and keeps its
    current state — a ``closed`` (acked) incident stays closed for the same
    signature. A changed error text mints a new incident automatically.
    """
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
    """Transition an incident's lifecycle state; return whether it changed.

    ``closed`` is terminal for that signature: no transition (including back
    to ``alerted``) leaves it — re-open happens by the error changing and
    minting a NEW incident. Unknown states are rejected (no-op, ``False``).
    """
    if state not in INCIDENT_STATES:
        return False
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = conn.execute(
            "SELECT state FROM cron_incidents WHERE id=?", (incident_id,)
        ).fetchone()
        if row is None or row["state"] == state:
            return False
        if row["state"] == "closed":
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
    """Acknowledge (close) an incident; return whether the state changed.

    A no-op (``False``) when the incident does not exist or is already closed.
    """
    return set_incident_state(incident_id, "closed")


def list_incidents(state: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return incidents, newest-activity first, optionally filtered by state."""
    if state is not None and state not in INCIDENT_STATES:
        return []
    with _transaction() as conn:
        if state is None:
            rows = conn.execute(
                "SELECT * FROM cron_incidents "
                "ORDER BY last_seen_at DESC, id DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cron_incidents WHERE state=? "
                "ORDER BY last_seen_at DESC, id DESC",
                (state,),
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
    with _transaction() as conn:
        if state is None:
            row = conn.execute("SELECT COUNT(*) AS n FROM cron_incidents").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM cron_incidents WHERE state=?",
                (state,),
            ).fetchone()
    return int(row["n"]) if row is not None else 0
