"""Isolated persistence for state-update candidates awaiting approval."""

from __future__ import annotations

import sqlite3
from typing import Any

from .state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_state_candidates (
    candidate_id TEXT PRIMARY KEY NOT NULL,
    state_key TEXT NOT NULL,
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    scope TEXT NOT NULL,
    evidence_ref TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'deferred', 'applied')),
    decided_by TEXT,
    decision_reason TEXT,
    applied_by TEXT,
    applied_reason TEXT
)
"""


_APPLY_SAVEPOINT = "apply_approved_to_current"


def _require_scope(scope: str) -> str:
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope is required")
    return scope


def ensure_schema(conn: sqlite3.Connection) -> None:
    existing = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'pending_state_candidates'"
    ).fetchone()
    if existing is not None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(pending_state_candidates)")}
        required = {
            "candidate_id", "state_key", "old_value", "new_value", "scope",
            "evidence_ref", "reason", "status", "decided_by", "decision_reason",
            "applied_by", "applied_reason",
        }
        info = {row[1]: row for row in conn.execute("PRAGMA table_info(pending_state_candidates)")}
        normalized_sql = " ".join((existing[0] or "").lower().split())
        expected_check = "status text not null check (status in ('pending', 'approved', 'rejected', 'deferred', 'applied'))"
        required_not_null = required - {"candidate_id", "decided_by", "decision_reason", "applied_by", "applied_reason"}
        if (
            not required.issubset(columns)
            or info.get("candidate_id", (None, None, None, None, None, 0))[5] != 1
            or any(info.get(name, (None, None, None, None, 0, 0))[3] != 1 for name in required_not_null)
                or "candidate_id text primary key not null" not in normalized_sql
            or expected_check not in normalized_sql
        ):
            raise RuntimeError("pending_state_candidates schema migration required")
        return
    # The caller owns commit/rollback. This helper must compose with an outer transaction.
    conn.execute(_SCHEMA)


def save_pending(conn: sqlite3.Connection, result: StateCandidateResult) -> bool:
    """Persist one pending candidate idempotently; never mutate current state.

    The caller owns the transaction and must commit or roll back explicitly.
    """
    if not isinstance(result, StateCandidateResult) or result.status is not CandidateStatus.PENDING:
        raise ValueError("only pending candidates may be persisted")
    fields = (
        result.candidate_id, result.state_key, result.old_value, result.new_value,
        result.scope, result.evidence_ref, result.reason,
    )
    if not all(isinstance(value, str) and value.strip() for value in fields):
        raise ValueError("pending candidate fields must be nonblank strings")
    _require_scope(result.scope)
    ensure_schema(conn)
    existing = conn.execute(
        "SELECT state_key, old_value, new_value, scope, evidence_ref, reason, status "
        "FROM pending_state_candidates WHERE candidate_id = ?",
        (result.candidate_id,),
    ).fetchone()
    if existing is not None:
        expected = (
            result.state_key, result.old_value, result.new_value, result.scope,
            result.evidence_ref, result.reason, "pending",
        )
        if tuple(existing) != expected:
            raise ValueError("candidate_id already belongs to a different candidate")
        return True
    cursor = conn.execute(
        """INSERT OR IGNORE INTO pending_state_candidates
        (candidate_id, state_key, old_value, new_value, scope, evidence_ref, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (
            result.candidate_id, result.state_key, result.old_value, result.new_value,
            result.scope, result.evidence_ref, result.reason,
        ),
    )
    if cursor.rowcount != 1:
        existing = conn.execute(
            "SELECT state_key, old_value, new_value, scope, evidence_ref, reason, status "
            "FROM pending_state_candidates WHERE candidate_id = ?",
            (result.candidate_id,),
        ).fetchone()
        expected = (
            result.state_key, result.old_value, result.new_value, result.scope,
            result.evidence_ref, result.reason, "pending",
        )
        if existing is None or tuple(existing) != expected:
            raise RuntimeError("pending candidate was not persisted")
    return True


def decide_pending(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    scope: str,
    decision: str,
    decided_by: str,
    reason: str,
) -> bool:
    """Record one explicit decision; this never applies current state."""
    if decision not in {"approved", "rejected", "deferred"}:
        raise ValueError("decision must be approved, rejected, or deferred")
    if not decided_by or not reason:
        raise ValueError("decided_by and reason are required")
    scope = _require_scope(scope)
    ensure_schema(conn)
    cursor = conn.execute(
        """UPDATE pending_state_candidates
           SET status = ?, decided_by = ?, decision_reason = ?
         WHERE candidate_id = ? AND scope = ? AND status = 'pending'""",
        (decision, decided_by, reason, candidate_id, scope),
    )
    return cursor.rowcount == 1


def apply_approved_to_current(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    scope: str,
    applied_by: str,
    reason: str,
) -> bool:
    """Apply an approved candidate to caller-provisioned current_state.

    The operation uses a savepoint so it composes with a caller-owned transaction.
    It never creates or migrates the authoritative current_state table.
    """
    if not applied_by or not reason:
        raise ValueError("applied_by and reason are required")
    scope = _require_scope(scope)
    ensure_schema(conn)
    conn.execute(f"SAVEPOINT {_APPLY_SAVEPOINT}")
    try:
        candidate = conn.execute(
            """SELECT state_key, old_value, new_value, scope, status
                 FROM pending_state_candidates
                WHERE candidate_id = ? AND scope = ?""",
            (candidate_id, scope),
        ).fetchone()
        if candidate is None:
            raise KeyError(f"unknown candidate: {candidate_id}")
        state_key, old_value, new_value, candidate_scope, status = candidate
        if status != "approved":
            raise ValueError("only approved candidates may be applied")
        current = conn.execute(
            """SELECT value FROM current_state
                 WHERE state_key = ? AND scope = ?""",
            (state_key, candidate_scope),
        ).fetchone()
        if current is None:
            raise LookupError("current state for candidate scope is missing")
        if current[0] != old_value:
            raise RuntimeError("current state conflict; candidate old_value no longer matches")
        updated = conn.execute(
            """UPDATE current_state SET value = ?
                 WHERE state_key = ? AND scope = ? AND value = ?""",
            (new_value, state_key, candidate_scope, old_value),
        )
        if updated.rowcount != 1:
            raise RuntimeError("current state update was not applied")
        marked = conn.execute(
            """UPDATE pending_state_candidates
                  SET status = 'applied', applied_by = ?, applied_reason = ?
                WHERE candidate_id = ? AND scope = ? AND status = 'approved'""",
            (applied_by, reason, candidate_id, scope),
        )
        if marked.rowcount != 1:
            raise RuntimeError("approved candidate was not marked applied")
        conn.execute(f"RELEASE SAVEPOINT {_APPLY_SAVEPOINT}")
        return True
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {_APPLY_SAVEPOINT}")
        conn.execute(f"RELEASE SAVEPOINT {_APPLY_SAVEPOINT}")
        raise


def read_decision(conn: sqlite3.Connection, candidate_id: str, *, scope: str) -> dict[str, Any] | None:
    """Read status and approval metadata within the caller's scope."""
    scope = _require_scope(scope)
    ensure_schema(conn)
    row = conn.execute(
        """SELECT candidate_id, status, decided_by, decision_reason
             FROM pending_state_candidates
            WHERE candidate_id = ? AND scope = ?""",
        (candidate_id, scope),
    ).fetchone()
    if row is None:
        return None
    return {"candidate_id": row[0], "status": row[1], "decided_by": row[2], "reason": row[3]}


def _candidate_from_row(row) -> StateCandidateResult:
    return StateCandidateResult(
        candidate_id=row[0], state_key=row[1], old_value=row[2], new_value=row[3],
        scope=row[4], evidence_ref=row[5], reason=row[6],
        delta_type=DeltaType.REPLACE, status=CandidateStatus(row[7]),
    )


def read_pending(conn: sqlite3.Connection, candidate_id: str, *, scope: str) -> StateCandidateResult | None:
    """Read a pending candidate only within the caller's scope."""
    scope = _require_scope(scope)
    ensure_schema(conn)
    row = conn.execute(
        """SELECT candidate_id, state_key, old_value, new_value, scope,
                  evidence_ref, reason, status
             FROM pending_state_candidates
            WHERE candidate_id = ? AND scope = ? AND status = 'pending'""",
        (candidate_id, scope),
    ).fetchone()
    return _candidate_from_row(row) if row is not None else None


def list_pending(conn: sqlite3.Connection, *, scope: str) -> list[StateCandidateResult]:
    ensure_schema(conn)
    scope = _require_scope(scope)
    rows = conn.execute(
        """SELECT candidate_id, state_key, old_value, new_value, scope,
                  evidence_ref, reason, status
             FROM pending_state_candidates
            WHERE scope = ? AND status = 'pending' ORDER BY candidate_id""",
        (scope,),
    ).fetchall()
    return [_candidate_from_row(row) for row in rows]
