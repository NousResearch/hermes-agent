"""Shared project-finalization contract, schema, data objects and public interfaces.

HOF-002: defines persistence boundaries, identity, states, locking, membership,
and queryable migration marker ONLY. No evaluation, reporting, delivery sending,
cleanup execution, repair routing, or gateway logic.

All operations use the normal kanban connection path (WAL + FULL + FK).
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, List

from hermes_cli.sqlite_util import add_column_if_missing as _add_column_if_missing
from hermes_cli.sqlite_util import write_txn

# ---------------------------------------------------------------------------
# Constants / Enums (externally distinguishable)
# ---------------------------------------------------------------------------

PROJECT_FINALIZATION_STATES: tuple[str, ...] = (
    "open",
    "waiting",
    "evaluating",
    "repairing",
    "complete",
    "blocked",
    "failed",
    "delivery_pending",
    "delivery_failed",
    "cleanup_scheduled",
    "cleaned",
)

TERMINAL_OUTCOMES: tuple[str, ...] = ("COMPLETE", "BLOCKED", "FAILED")

CHECKER_VERDICTS: tuple[str, ...] = ("PASS", "FAIL_REPAIRABLE", "FAIL_TERMINAL")

NOTIFICATION_POLICIES: tuple[str, ...] = ("project_summary", "verbose", "silent")

MEMBERSHIP_KINDS: tuple[str, ...] = ("required", "support", "repair", "checker")

SCHEMA_VERSION = "1"
MIGRATION_MARKER = "hof002-v1"

# Regex for SHA-256 (64 lowercase hex)
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

# Full column list for repair (used for partial migration tolerance)
PROJECT_FINALIZATIONS_COLS = [
    ("board_id", "TEXT NOT NULL"),
    ("root_task_id", "TEXT NOT NULL"),
    ("generation", "INTEGER NOT NULL"),
    ("state", "TEXT NOT NULL"),
    ("terminal_outcome", "TEXT"),
    ("final_checker_task_id", "TEXT NOT NULL"),
    ("checker_verdict", "TEXT"),
    ("repair_generation", "INTEGER NOT NULL DEFAULT 0"),
    ("repair_budget", "INTEGER NOT NULL DEFAULT 1"),
    ("notification_policy", "TEXT NOT NULL"),
    ("retention_days", "INTEGER NOT NULL"),
    ("final_report_path", "TEXT"),
    ("final_report_sha256", "TEXT"),
    ("manifest_path", "TEXT"),
    ("manifest_sha256", "TEXT"),
    ("usage_summary_json", "TEXT"),
    ("blocker_json", "TEXT"),
    ("created_at", "INTEGER NOT NULL"),
    ("updated_at", "INTEGER NOT NULL"),
    ("evaluated_at", "INTEGER"),
    ("finalized_at", "INTEGER"),
    ("cleanup_after", "TEXT"),
    ("cleaned_at", "INTEGER"),
    ("lock_owner", "TEXT"),
    ("lock_expires_at", "INTEGER"),
    ("version", "INTEGER NOT NULL DEFAULT 1"),
]


# ---------------------------------------------------------------------------
# Data objects (stable public)
# ---------------------------------------------------------------------------

@dataclass
class ProjectFinalization:
    """Canonical in-memory representation of a project finalization row."""
    board_id: str
    root_task_id: str
    generation: int
    state: str
    terminal_outcome: Optional[str]
    final_checker_task_id: str
    checker_verdict: Optional[str]
    repair_generation: int
    repair_budget: int
    notification_policy: str
    retention_days: int
    final_report_path: Optional[str]
    final_report_sha256: Optional[str]
    manifest_path: Optional[str]
    manifest_sha256: Optional[str]
    usage_summary_json: Optional[str]
    blocker_json: Optional[str]
    created_at: int
    updated_at: int
    evaluated_at: Optional[int]
    finalized_at: Optional[int]
    cleanup_after: Optional[str]
    cleaned_at: Optional[int]
    lock_owner: Optional[str]
    lock_expires_at: Optional[int]
    version: int = 1


@dataclass
class ProjectMember:
    """Explicit project membership (support/repair/checker etc)."""
    board_id: str
    root_task_id: str
    generation: int
    task_id: str
    membership_kind: str
    required: bool
    created_at: int


@dataclass
class ProjectDeliveryAttempt:
    """Delivery attempt ledger boundary (persistence only)."""
    id: Optional[int]
    board_id: str
    root_task_id: str
    generation: int
    idempotency_key: str
    platform: str
    destination_reference: Optional[str]
    thread_reference: Optional[str]
    attempt_number: int
    delivery_state: str
    accepted: Optional[bool]
    provider_message_id: Optional[str]
    redacted_error: Optional[str]
    created_at: int
    completed_at: Optional[int]
    next_retry_at: Optional[int]


@dataclass
class ProjectFailureEnvelope:
    """Provider failure envelope boundary (persistence only)."""
    id: Optional[int]
    board_id: str
    root_task_id: str
    generation: int
    task_id: str
    run_id: Optional[int]
    provider: Optional[str]
    model: Optional[str]
    failure_class: Optional[str]
    status_code: Optional[int]
    retry_after: Optional[int]
    redacted_error: Optional[str]
    error_fingerprint: Optional[str]
    created_at: int


@dataclass
class ProjectCleanupJournal:
    """Cleanup journal boundary (persistence only)."""
    id: Optional[int]
    board_id: str
    root_task_id: str
    generation: int
    plan_sha256: Optional[str]
    mode: Optional[str]
    status: str
    retention_cutoff: Optional[int]
    eligible_task_count: int
    excluded_task_count: int
    deleted_task_count: int
    archived_task_count: int
    evidence_path: Optional[str]
    created_at: int
    executed_at: Optional[int]
    redacted_error: Optional[str]


# ---------------------------------------------------------------------------
# Schema (additive, idempotent, with migration marker)
# ---------------------------------------------------------------------------

PROJECT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_finalizations (
    board_id              TEXT NOT NULL,
    root_task_id          TEXT NOT NULL,
    generation            INTEGER NOT NULL,
    state                 TEXT NOT NULL,
    terminal_outcome      TEXT,
    final_checker_task_id TEXT NOT NULL,
    checker_verdict       TEXT,
    repair_generation     INTEGER NOT NULL DEFAULT 0,
    repair_budget         INTEGER NOT NULL DEFAULT 1,
    notification_policy   TEXT NOT NULL,
    retention_days        INTEGER NOT NULL,
    final_report_path     TEXT,
    final_report_sha256   TEXT,
    manifest_path         TEXT,
    manifest_sha256       TEXT,
    usage_summary_json    TEXT,
    blocker_json          TEXT,
    created_at            INTEGER NOT NULL,
    updated_at            INTEGER NOT NULL,
    evaluated_at          INTEGER,
    finalized_at          INTEGER,
    cleanup_after         TEXT,
    cleaned_at            INTEGER,
    lock_owner            TEXT,
    lock_expires_at       INTEGER,
    version               INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (board_id, root_task_id, generation)
);

CREATE TABLE IF NOT EXISTS project_finalization_members (
    board_id        TEXT NOT NULL,
    root_task_id    TEXT NOT NULL,
    generation      INTEGER NOT NULL,
    task_id         TEXT NOT NULL,
    membership_kind TEXT NOT NULL,
    required        INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (board_id, root_task_id, generation, task_id)
);

CREATE TABLE IF NOT EXISTS project_delivery_attempts (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id              TEXT NOT NULL,
    root_task_id          TEXT NOT NULL,
    generation            INTEGER NOT NULL,
    idempotency_key       TEXT NOT NULL,
    platform              TEXT NOT NULL,
    destination_reference TEXT,
    thread_reference      TEXT,
    attempt_number        INTEGER NOT NULL,
    delivery_state        TEXT NOT NULL,
    accepted              INTEGER,
    provider_message_id   TEXT,
    redacted_error        TEXT,
    created_at            INTEGER NOT NULL,
    completed_at          INTEGER,
    next_retry_at         INTEGER,
    UNIQUE (idempotency_key, attempt_number)
);

CREATE TABLE IF NOT EXISTS project_failure_envelopes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id          TEXT NOT NULL,
    root_task_id      TEXT NOT NULL,
    generation        INTEGER NOT NULL,
    task_id           TEXT NOT NULL,
    run_id            INTEGER,
    provider          TEXT,
    model             TEXT,
    failure_class     TEXT,
    status_code       INTEGER,
    retry_after       INTEGER,
    redacted_error    TEXT,
    error_fingerprint TEXT,
    created_at        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS project_cleanup_journal (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id              TEXT NOT NULL,
    root_task_id          TEXT NOT NULL,
    generation            INTEGER NOT NULL,
    plan_sha256           TEXT,
    mode                  TEXT,
    status                TEXT NOT NULL,
    retention_cutoff      INTEGER,
    eligible_task_count   INTEGER NOT NULL DEFAULT 0,
    excluded_task_count   INTEGER NOT NULL DEFAULT 0,
    deleted_task_count    INTEGER NOT NULL DEFAULT 0,
    archived_task_count   INTEGER NOT NULL DEFAULT 0,
    evidence_path         TEXT,
    created_at            INTEGER NOT NULL,
    executed_at           INTEGER,
    redacted_error        TEXT
);

-- Queryable migration identity for HOF-002 (named marker + version row)
CREATE TABLE IF NOT EXISTS project_finalization_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Useful indexes for common queries (idempotent)
CREATE INDEX IF NOT EXISTS idx_pfinal_board_state ON project_finalizations(board_id, state);
CREATE INDEX IF NOT EXISTS idx_pfinal_root_gen ON project_finalizations(board_id, root_task_id, generation);
CREATE INDEX IF NOT EXISTS idx_pmembers_root ON project_finalization_members(board_id, root_task_id, generation);
CREATE INDEX IF NOT EXISTS idx_pdelivery_key ON project_delivery_attempts(idempotency_key, attempt_number);
CREATE INDEX IF NOT EXISTS idx_pfailure_root ON project_failure_envelopes(board_id, root_task_id);
CREATE INDEX IF NOT EXISTS idx_pcleanup_root ON project_cleanup_journal(board_id, root_task_id);
"""


def ensure_project_finalization_schema(conn: sqlite3.Connection) -> None:
    """Idempotent creation of project finalization tables + migration marker.

    Uses normal connection pragmas (caller must have set WAL/FK/FULL).
    Detects/repairs partial column sets on the new tables via additive ALTER.
    Sets queryable marker in project_finalization_meta.
    """
    conn.executescript(PROJECT_SCHEMA_SQL)

    # Ensure meta marker rows exist (idempotent)
    conn.execute(
        "INSERT OR IGNORE INTO project_finalization_meta (key, value) VALUES (?, ?)",
        ("version", SCHEMA_VERSION),
    )
    conn.execute(
        "INSERT OR IGNORE INTO project_finalization_meta (key, value) VALUES (?, ?)",
        ("migration", MIGRATION_MARKER),
    )

    # Robust repair for any pre-existing partial table definition
    _ensure_full_project_finalization_columns(conn)


def _ensure_full_project_finalization_columns(conn: sqlite3.Connection) -> None:
    """Ensure all required columns exist on project_finalizations (for partial migration cases)."""
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(project_finalizations)")}
    except Exception:
        return
    for col_name, ddl_type in PROJECT_FINALIZATIONS_COLS:
        if col_name not in cols:
            try:
                _add_column_if_missing(conn, "project_finalizations", col_name, f"{col_name} {ddl_type}")
            except Exception:
                pass  # best effort; CREATE path covers fresh


def get_project_finalization_migration_marker(conn: sqlite3.Connection) -> str | None:
    """Return the queryable migration identity (e.g. 'hof002-v1')."""
    row = conn.execute(
        "SELECT value FROM project_finalization_meta WHERE key = ?",
        ("migration",),
    ).fetchone()
    return row[0] if row else None


def get_project_finalization_schema_version(conn: sqlite3.Connection) -> str | None:
    """Return the schema version marker."""
    row = conn.execute(
        "SELECT value FROM project_finalization_meta WHERE key = ?",
        ("version",),
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Row <-> dataclass helpers
# ---------------------------------------------------------------------------

def _row_to_project_finalization(row: sqlite3.Row) -> ProjectFinalization:
    return ProjectFinalization(
        board_id=row["board_id"],
        root_task_id=row["root_task_id"],
        generation=int(row["generation"]),
        state=row["state"],
        terminal_outcome=row["terminal_outcome"],
        final_checker_task_id=row["final_checker_task_id"],
        checker_verdict=row["checker_verdict"],
        repair_generation=int(row["repair_generation"]),
        repair_budget=int(row["repair_budget"]),
        notification_policy=row["notification_policy"],
        retention_days=int(row["retention_days"]),
        final_report_path=row["final_report_path"],
        final_report_sha256=row["final_report_sha256"],
        manifest_path=row["manifest_path"],
        manifest_sha256=row["manifest_sha256"],
        usage_summary_json=row["usage_summary_json"],
        blocker_json=row["blocker_json"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        evaluated_at=int(row["evaluated_at"]) if row["evaluated_at"] is not None else None,
        finalized_at=int(row["finalized_at"]) if row["finalized_at"] is not None else None,
        cleanup_after=row["cleanup_after"],
        cleaned_at=int(row["cleaned_at"]) if row["cleaned_at"] is not None else None,
        lock_owner=row["lock_owner"],
        lock_expires_at=int(row["lock_expires_at"]) if row["lock_expires_at"] is not None else None,
        version=int(row["version"]) if row["version"] is not None else 1,
    )


def _row_to_project_member(row: sqlite3.Row) -> ProjectMember:
    return ProjectMember(
        board_id=row["board_id"],
        root_task_id=row["root_task_id"],
        generation=int(row["generation"]),
        task_id=row["task_id"],
        membership_kind=row["membership_kind"],
        required=bool(row["required"]),
        created_at=int(row["created_at"]),
    )


def _row_to_delivery_attempt(row: sqlite3.Row) -> ProjectDeliveryAttempt:
    return ProjectDeliveryAttempt(
        id=row["id"],
        board_id=row["board_id"],
        root_task_id=row["root_task_id"],
        generation=int(row["generation"]),
        idempotency_key=row["idempotency_key"],
        platform=row["platform"],
        destination_reference=row["destination_reference"],
        thread_reference=row["thread_reference"],
        attempt_number=int(row["attempt_number"]),
        delivery_state=row["delivery_state"],
        accepted=bool(row["accepted"]) if row["accepted"] is not None else None,
        provider_message_id=row["provider_message_id"],
        redacted_error=row["redacted_error"],
        created_at=int(row["created_at"]),
        completed_at=int(row["completed_at"]) if row["completed_at"] is not None else None,
        next_retry_at=int(row["next_retry_at"]) if row["next_retry_at"] is not None else None,
    )


def _row_to_failure_envelope(row: sqlite3.Row) -> ProjectFailureEnvelope:
    return ProjectFailureEnvelope(
        id=row["id"],
        board_id=row["board_id"],
        root_task_id=row["root_task_id"],
        generation=int(row["generation"]),
        task_id=row["task_id"],
        run_id=int(row["run_id"]) if row["run_id"] is not None else None,
        provider=row["provider"],
        model=row["model"],
        failure_class=row["failure_class"],
        status_code=int(row["status_code"]) if row["status_code"] is not None else None,
        retry_after=int(row["retry_after"]) if row["retry_after"] is not None else None,
        redacted_error=row["redacted_error"],
        error_fingerprint=row["error_fingerprint"],
        created_at=int(row["created_at"]),
    )


def _row_to_cleanup_journal(row: sqlite3.Row) -> ProjectCleanupJournal:
    return ProjectCleanupJournal(
        id=row["id"],
        board_id=row["board_id"],
        root_task_id=row["root_task_id"],
        generation=int(row["generation"]),
        plan_sha256=row["plan_sha256"],
        mode=row["mode"],
        status=row["status"],
        retention_cutoff=int(row["retention_cutoff"]) if row["retention_cutoff"] is not None else None,
        eligible_task_count=int(row["eligible_task_count"]),
        excluded_task_count=int(row["excluded_task_count"]),
        deleted_task_count=int(row["deleted_task_count"]),
        archived_task_count=int(row["archived_task_count"]),
        evidence_path=row["evidence_path"],
        created_at=int(row["created_at"]),
        executed_at=int(row["executed_at"]) if row["executed_at"] is not None else None,
        redacted_error=row["redacted_error"],
    )


# ---------------------------------------------------------------------------
# Validation helpers (strict per contract)
# ---------------------------------------------------------------------------

def validate_notification_policy(policy: str) -> None:
    if policy not in NOTIFICATION_POLICIES:
        raise ValueError(f"invalid notification_policy: {policy!r}")


def validate_retention_days(days: int) -> None:
    if not isinstance(days, int) or days < 2:
        raise ValueError(f"retention_days must be >= 2, got {days}")


def validate_repair_budget(budget: int) -> None:
    if budget not in (0, 1):
        raise ValueError(f"repair_budget must be 0 or 1, got {budget}")


def validate_terminal_outcome(outcome: Optional[str]) -> None:
    if outcome is not None and outcome not in TERMINAL_OUTCOMES:
        raise ValueError(f"invalid terminal_outcome: {outcome!r}")


def validate_checker_verdict(verdict: str) -> None:
    if verdict not in CHECKER_VERDICTS:
        raise ValueError(f"invalid checker_verdict: {verdict!r}")


def validate_state(state: str) -> None:
    if state not in PROJECT_FINALIZATION_STATES:
        raise ValueError(f"invalid state: {state!r}")


def validate_sha256(sha: Optional[str]) -> None:
    if sha is not None and not _SHA256_RE.match(sha):
        raise ValueError(f"invalid sha256 (must be 64 lowercase hex): {sha!r}")


def validate_generation(generation: int) -> None:
    if not isinstance(generation, int) or generation < 1:
        raise ValueError(f"generation must be >= 1, got {generation}")


def validate_membership_kind(kind: str) -> None:
    if kind not in MEMBERSHIP_KINDS:
        raise ValueError(f"invalid membership_kind: {kind!r}")


# ---------------------------------------------------------------------------
# Internal row helpers
# ---------------------------------------------------------------------------

def _get_project_finalization_row(
    conn: sqlite3.Connection, board_id: str, root_task_id: str, generation: int
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM project_finalizations WHERE board_id=? AND root_task_id=? AND generation=?",
        (board_id, root_task_id, generation),
    ).fetchone()


# ---------------------------------------------------------------------------
# Public CRUD + locking + membership (per contract section 13)
# ---------------------------------------------------------------------------

def create_project_finalization(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    final_checker_task_id: str,
    notification_policy: str = "project_summary",
    retention_days: int = 3,
    repair_budget: int = 1,
) -> ProjectFinalization:
    """Create (or return existing) project finalization for generation 1.

    Idempotent on (board, root, gen=1). Generation 2+ via explicit reopen (future).
    """
    validate_notification_policy(notification_policy)
    validate_retention_days(retention_days)
    validate_repair_budget(repair_budget)
    if not board_id or not root_task_id or not final_checker_task_id:
        raise ValueError("board_id, root_task_id and final_checker_task_id are required")

    generation = 1
    validate_generation(generation)

    now = int(time.time())
    with write_txn(conn):
        existing = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        if existing:
            return _row_to_project_finalization(existing)

        conn.execute(
            """
            INSERT INTO project_finalizations (
                board_id, root_task_id, generation, state, terminal_outcome,
                final_checker_task_id, checker_verdict,
                repair_generation, repair_budget,
                notification_policy, retention_days,
                final_report_path, final_report_sha256,
                manifest_path, manifest_sha256,
                usage_summary_json, blocker_json,
                created_at, updated_at,
                evaluated_at, finalized_at,
                cleanup_after, cleaned_at,
                lock_owner, lock_expires_at, version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                board_id, root_task_id, generation, "open", None,
                final_checker_task_id, None,
                0, repair_budget,
                notification_policy, retention_days,
                None, None, None, None, None, None,
                now, now, None, None, None, None, None, None, 1,
            ),
        )
        row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        assert row is not None
        return _row_to_project_finalization(row)


def get_project_finalization(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
) -> Optional[ProjectFinalization]:
    """Return specific generation or the latest (highest generation)."""
    if generation is None:
        row = conn.execute(
            """
            SELECT * FROM project_finalizations
            WHERE board_id = ? AND root_task_id = ?
            ORDER BY generation DESC LIMIT 1
            """,
            (board_id, root_task_id),
        ).fetchone()
    else:
        validate_generation(generation)
        row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
    return _row_to_project_finalization(row) if row else None


def list_project_finalizations(
    conn: sqlite3.Connection,
    *,
    board_id: str | None = None,
    state: str | None = None,
) -> list[ProjectFinalization]:
    """List finalizations, optionally filtered by board and/or state."""
    where = []
    params: list[Any] = []
    if board_id is not None:
        where.append("board_id = ?")
        params.append(board_id)
    if state is not None:
        validate_state(state)
        where.append("state = ?")
        params.append(state)
    sql = "SELECT * FROM project_finalizations"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY board_id, root_task_id, generation"
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_project_finalization(r) for r in rows]


def acquire_finalization_lock(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    owner: str,
    lease_seconds: int,
    now: str | None = None,
) -> bool:
    """CAS acquire/renew of the durable lease lock on the project finalization row.

    Returns True on success (acquired or renewed by same owner).
    """
    validate_generation(generation)
    if not owner:
        raise ValueError("owner required for lock")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")

    now_ts = int(now) if now is not None else int(time.time())
    expires = now_ts + lease_seconds

    with write_txn(conn):
        cur = conn.execute(
            """
            UPDATE project_finalizations
               SET lock_owner = ?,
                   lock_expires_at = ?,
                   updated_at = ?
             WHERE board_id = ?
               AND root_task_id = ?
               AND generation = ?
               AND (lock_owner IS NULL
                    OR lock_owner = ?
                    OR COALESCE(lock_expires_at, 0) < ?)
            """,
            (owner, expires, now_ts, board_id, root_task_id, generation, owner, now_ts),
        )
        return cur.rowcount > 0


def release_finalization_lock(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    owner: str,
) -> None:
    """Release only if the current owner matches. Repeated release is safe."""
    validate_generation(generation)
    now_ts = int(time.time())
    with write_txn(conn):
        conn.execute(
            """
            UPDATE project_finalizations
               SET lock_owner = NULL,
                   lock_expires_at = NULL,
                   updated_at = ?
             WHERE board_id = ?
               AND root_task_id = ?
               AND generation = ?
               AND lock_owner = ?
            """,
            (now_ts, board_id, root_task_id, generation, owner),
        )


def record_checker_verdict(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    checker_task_id: str,
    verdict: str,
) -> ProjectFinalization:
    """Record checker verdict. Only the designated final_checker_task_id may write it.

    Idempotent if same verdict already recorded.
    """
    validate_generation(generation)
    validate_checker_verdict(verdict)

    now = int(time.time())
    with write_txn(conn):
        row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        if row is None:
            raise ValueError(f"no project finalization for {board_id}/{root_task_id}/{generation}")
        if row["final_checker_task_id"] != checker_task_id:
            raise ValueError("checker_task_id does not match designated final_checker_task_id")
        if row["checker_verdict"] == verdict:
            return _row_to_project_finalization(row)

        conn.execute(
            """
            UPDATE project_finalizations
               SET checker_verdict = ?,
                   evaluated_at = ?,
                   updated_at = ?
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
            """,
            (verdict, now, now, board_id, root_task_id, generation),
        )
        new_row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        return _row_to_project_finalization(new_row)


def record_final_artifacts(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    report_path: str,
    report_sha256: str,
    manifest_path: str,
    manifest_sha256: str,
    usage_summary_json: str | None = None,
) -> ProjectFinalization:
    """Persist final report and manifest identity. Idempotent for identical values."""
    validate_generation(generation)
    validate_sha256(report_sha256)
    validate_sha256(manifest_sha256)

    now = int(time.time())
    with write_txn(conn):
        row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        if row is None:
            raise ValueError("project finalization does not exist")
        # Idempotent if identical
        if (
            row["final_report_path"] == report_path
            and row["final_report_sha256"] == report_sha256
            and row["manifest_path"] == manifest_path
            and row["manifest_sha256"] == manifest_sha256
        ):
            return _row_to_project_finalization(row)

        conn.execute(
            """
            UPDATE project_finalizations
               SET final_report_path = ?,
                   final_report_sha256 = ?,
                   manifest_path = ?,
                   manifest_sha256 = ?,
                   usage_summary_json = COALESCE(?, usage_summary_json),
                   finalized_at = ?,
                   updated_at = ?
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
            """,
            (
                report_path, report_sha256, manifest_path, manifest_sha256,
                usage_summary_json, now, now, board_id, root_task_id, generation,
            ),
        )
        new_row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        return _row_to_project_finalization(new_row)


def record_terminal_outcome(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    outcome: str,
    blocker_json: str | None = None,
) -> ProjectFinalization:
    """Record the terminal project outcome (COMPLETE / BLOCKED / FAILED).

    Sets matching state. Idempotent on same outcome.
    """
    validate_generation(generation)
    validate_terminal_outcome(outcome)

    now = int(time.time())
    # Map outcome to a terminal-ish state (keeps state and outcome separate per spec)
    state_map = {
        "COMPLETE": "complete",
        "BLOCKED": "blocked",
        "FAILED": "failed",
    }
    target_state = state_map[outcome]

    with write_txn(conn):
        row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        if row is None:
            raise ValueError("project finalization does not exist")
        if row["terminal_outcome"] == outcome:
            return _row_to_project_finalization(row)

        conn.execute(
            """
            UPDATE project_finalizations
               SET terminal_outcome = ?,
                   state = ?,
                   blocker_json = COALESCE(?, blocker_json),
                   finalized_at = ?,
                   updated_at = ?
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
            """,
            (outcome, target_state, blocker_json, now, now, board_id, root_task_id, generation),
        )
        new_row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        return _row_to_project_finalization(new_row)


def schedule_project_cleanup(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    cleanup_after: str,
) -> ProjectFinalization:
    """Schedule cleanup for a finalized project. Requires cleanup_after."""
    validate_generation(generation)
    if not cleanup_after:
        raise ValueError("cleanup_after is required")

    now = int(time.time())
    with write_txn(conn):
        row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        if row is None:
            raise ValueError("project finalization does not exist")
        if row["cleanup_after"] == cleanup_after and row["state"] == "cleanup_scheduled":
            return _row_to_project_finalization(row)

        conn.execute(
            """
            UPDATE project_finalizations
               SET cleanup_after = ?,
                   state = 'cleanup_scheduled',
                   updated_at = ?
             WHERE board_id = ? AND root_task_id = ? AND generation = ?
            """,
            (cleanup_after, now, board_id, root_task_id, generation),
        )
        new_row = _get_project_finalization_row(conn, board_id, root_task_id, generation)
        return _row_to_project_finalization(new_row)


def register_project_member(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    task_id: str,
    membership_kind: str,
    required: bool,
) -> None:
    """Register (idempotent) an explicit project member."""
    validate_generation(generation)
    validate_membership_kind(membership_kind)
    if not task_id:
        raise ValueError("task_id required")

    now = int(time.time())
    with write_txn(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO project_finalization_members
                (board_id, root_task_id, generation, task_id, membership_kind, required, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (board_id, root_task_id, generation, task_id, membership_kind, 1 if required else 0, now),
        )


def list_project_members(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
) -> list[ProjectMember]:
    """List registered members for a project generation."""
    validate_generation(generation)
    rows = conn.execute(
        """
        SELECT * FROM project_finalization_members
        WHERE board_id = ? AND root_task_id = ? AND generation = ?
        ORDER BY created_at
        """,
        (board_id, root_task_id, generation),
    ).fetchall()
    return [_row_to_project_member(r) for r in rows]


# ---------------------------------------------------------------------------
# Convenience boundary recorders for delivery/failure/cleanup (schema boundaries)
# These are intentionally minimal; full semantics owned by later HOFs.
# ---------------------------------------------------------------------------

def record_delivery_attempt(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    idempotency_key: str,
    platform: str,
    attempt_number: int,
    delivery_state: str,
    destination_reference: str | None = None,
    thread_reference: str | None = None,
    accepted: bool | None = None,
    provider_message_id: str | None = None,
    redacted_error: str | None = None,
) -> ProjectDeliveryAttempt:
    """Insert (idempotent by key+attempt) a delivery attempt row."""
    validate_generation(generation)
    if not idempotency_key or not platform:
        raise ValueError("idempotency_key and platform required")
    now = int(time.time())
    with write_txn(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO project_delivery_attempts
            (board_id, root_task_id, generation, idempotency_key, platform,
             destination_reference, thread_reference, attempt_number, delivery_state,
             accepted, provider_message_id, redacted_error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                board_id, root_task_id, generation, idempotency_key, platform,
                destination_reference, thread_reference, attempt_number, delivery_state,
                1 if accepted else 0 if accepted is not None else None,
                provider_message_id, redacted_error, now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM project_delivery_attempts WHERE idempotency_key=? AND attempt_number=?",
            (idempotency_key, attempt_number),
        ).fetchone()
        return _row_to_delivery_attempt(row)


def record_failure_envelope(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    task_id: str,
    redacted_error: str | None = None,
    **kwargs: Any,
) -> ProjectFailureEnvelope:
    """Persist a failure envelope (minimal fields required for boundary)."""
    validate_generation(generation)
    now = int(time.time())
    with write_txn(conn):
        cur = conn.execute(
            """
            INSERT INTO project_failure_envelopes
            (board_id, root_task_id, generation, task_id, run_id, provider, model,
             failure_class, status_code, retry_after, redacted_error, error_fingerprint, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                board_id, root_task_id, generation, task_id,
                kwargs.get("run_id"),
                kwargs.get("provider"),
                kwargs.get("model"),
                kwargs.get("failure_class"),
                kwargs.get("status_code"),
                kwargs.get("retry_after"),
                redacted_error,
                kwargs.get("error_fingerprint"),
                now,
            ),
        )
        row = conn.execute("SELECT * FROM project_failure_envelopes WHERE id=?", (cur.lastrowid,)).fetchone()
        return _row_to_failure_envelope(row)


def record_cleanup_journal(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    plan_sha256: str | None = None,
    mode: str | None = None,
    status: str = "scheduled",
    **kwargs: Any,
) -> ProjectCleanupJournal:
    """Persist a cleanup journal entry (boundary only)."""
    validate_generation(generation)
    validate_sha256(plan_sha256) if plan_sha256 else None
    now = int(time.time())
    with write_txn(conn):
        cur = conn.execute(
            """
            INSERT INTO project_cleanup_journal
            (board_id, root_task_id, generation, plan_sha256, mode, status,
             retention_cutoff, eligible_task_count, excluded_task_count,
             deleted_task_count, archived_task_count, evidence_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                board_id, root_task_id, generation, plan_sha256, mode, status,
                kwargs.get("retention_cutoff"),
                kwargs.get("eligible_task_count", 0),
                kwargs.get("excluded_task_count", 0),
                kwargs.get("deleted_task_count", 0),
                kwargs.get("archived_task_count", 0),
                kwargs.get("evidence_path"),
                now,
            ),
        )
        row = conn.execute("SELECT * FROM project_cleanup_journal WHERE id=?", (cur.lastrowid,)).fetchone()
        return _row_to_cleanup_journal(row)
