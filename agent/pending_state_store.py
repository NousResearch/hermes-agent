"""Isolated persistence for state-update candidates awaiting approval."""

from __future__ import annotations

import sqlite3
from typing import Any

from .state_candidate_evaluator import CandidateStatus, DeltaType, StateCandidateResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_state_candidates (
    candidate_id TEXT PRIMARY KEY,
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
        if not required.issubset(columns) or "check (status = 'pending')" in (existing[0] or "").lower():
            raise RuntimeError("pending_state_candidates schema migration required")
        return
    conn.execute(_SCHEMA)
    conn.commit()


def save_pending(conn: sqlite3.Connection, result: StateCandidateResult) -> bool:
    """Persist one pending candidate idempotently; never mutate current state."""
    if result.status is not CandidateStatus.PENDING:
        raise ValueError("only pending candidates may be persisted")
    ensure_schema(conn)
    conn.execute(
        """INSERT OR IGNORE INTO pending_state_candidates
        (candidate_id, state_key, old_value, new_value, scope, evidence_ref, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (
            result.candidate_id, result.state_key, result.old_value, result.new_value,
            result.scope, result.evidence_ref, result.reason,
        ),
    )
    conn.commit()
    return True


def decide_pending(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    decision: str,
    decided_by: str,
    reason: str,
) -> bool:
    """Record one explicit decision; this never applies current state."""
    if decision not in {"approved", "rejected", "deferred"}:
        raise ValueError("decision must be approved, rejected, or deferred")
    if not decided_by or not reason:
        raise ValueError("decided_by and reason are required")
    ensure_schema(conn)
    cursor = conn.execute(
        """UPDATE pending_state_candidates
           SET status = ?, decided_by = ?, decision_reason = ?
         WHERE candidate_id = ? AND status = 'pending'""",
        (decision, decided_by, reason, candidate_id),
    )
    conn.commit()
    return cursor.rowcount == 1



def apply_approved_to_current(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    applied_by: str,
    reason: str,
) -> bool:
    """Atomically apply an approved candidate to an explicit caller-provisioned current_state table.

    The caller must provide ``current_state(state_key, scope, value)``; this function
    intentionally does not create or migrate that authoritative table. It only changes
    a row after approval and an old-value compare-and-swap, then marks the candidate
    applied in the pending store.
    """
    if not applied_by or not reason:
        raise ValueError("applied_by and reason are required")
    ensure_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        candidate = conn.execute(
            """SELECT state_key, old_value, new_value, scope, status
                 FROM pending_state_candidates WHERE candidate_id = ?""",
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            raise KeyError(f"unknown candidate: {candidate_id}")
        state_key, old_value, new_value, scope, status = candidate
        if status != "approved":
            raise ValueError("only approved candidates may be applied")
        current = conn.execute(
            """SELECT value FROM current_state
                 WHERE state_key = ? AND scope = ?""",
            (state_key, scope),
        ).fetchone()
        if current is None:
            raise LookupError("current state for candidate scope is missing")
        if current[0] != old_value:
            raise RuntimeError("current state conflict; candidate old_value no longer matches")
        updated = conn.execute(
            """UPDATE current_state SET value = ?
                 WHERE state_key = ? AND scope = ? AND value = ?""",
            (new_value, state_key, scope, old_value),
        )
        if updated.rowcount != 1:
            raise RuntimeError("current state update was not applied")
        conn.execute(
            """UPDATE pending_state_candidates
                  SET status = 'applied', applied_by = ?, applied_reason = ?
                WHERE candidate_id = ? AND status = 'approved'""",
            (applied_by, reason, candidate_id),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def read_decision(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any] | None:
    """Read status and approval metadata without promoting any state."""
    ensure_schema(conn)
    row = conn.execute(
        """SELECT candidate_id, status, decided_by, decision_reason
             FROM pending_state_candidates WHERE candidate_id = ?""",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return None
    return {"candidate_id": row[0], "status": row[1], "decided_by": row[2], "reason": row[3]}


def read_pending(conn: sqlite3.Connection, candidate_id: str) -> StateCandidateResult | None:
    ensure_schema(conn)
    row = conn.execute(
        """SELECT candidate_id, state_key, old_value, new_value, scope,
                  evidence_ref, reason, status
           FROM pending_state_candidates
          WHERE candidate_id = ? AND status = 'pending'""",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return None
    return StateCandidateResult(
        candidate_id=row[0], state_key=row[1], old_value=row[2], new_value=row[3],
        scope=row[4], evidence_ref=row[5], reason=row[6],
        delta_type=DeltaType.REPLACE, status=CandidateStatus(row[7]),
    )


def list_pending(conn: sqlite3.Connection, *, scope: str | None = None) -> list[StateCandidateResult]:
    ensure_schema(conn)
    if scope is None:
        rows = conn.execute(
            "SELECT candidate_id FROM pending_state_candidates WHERE status = 'pending' ORDER BY candidate_id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT candidate_id FROM pending_state_candidates WHERE scope = ? AND status = 'pending' ORDER BY candidate_id",
            (scope,),
        ).fetchall()
    return [candidate for row in rows if (candidate := read_pending(conn, row[0])) is not None]
