"""Lifecycle authorization and the claim-liveness/lease proof for the Kanban kernel.

Split out of the :mod:`hermes_cli.kanban_db` facade, which may only shrink.  Two things
live here because the kernel has exactly one of each:

* :func:`authorize_lifecycle` — the single owner/admin enforcement point every public
  mutator calls before its first write and again inside its own ``write_txn``;
  :func:`_append_admin_audit` records a permitted administrative mutation in the SAME
  transaction, on the board's existing event stream.
* the claim proof — :func:`_claim_is_live` (process still owns the claim),
  :func:`heartbeat_claim` / :func:`_extend_run_claim` / :func:`_extend_live_stale_claim`
  (lease extension) and :func:`_record_reclaim` (the ``reclaimed`` event).

Origin-resident names are reached late-bound through ``_kb`` so monkeypatching
``hermes_cli.kanban_db.<name>`` keeps working for every caller.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

import hermes_cli.kanban_db as _kb


def _resolve_claim_ttl_seconds(ttl_seconds: Optional[int] = None) -> int:
    """Explicit ``ttl_seconds`` > ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` > default."""
    if ttl_seconds is not None:
        return max(1, int(ttl_seconds))

    return _kb._env_int("HERMES_KANBAN_CLAIM_TTL_SECONDS", _kb.DEFAULT_CLAIM_TTL_SECONDS, minimum=1)


def heartbeat_claim(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> bool:
    """Extend a running claim; True if we still own it."""
    expires = int(time.time()) + _kb._resolve_claim_ttl_seconds(ttl_seconds)
    lock = claimer or _kb._claimer_id()
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ?", (expires, task_id, lock),
        )
        if cur.rowcount != 1:
            return False
        _kb._extend_run_claim(conn, task_id, expires)
        return True


def _extend_run_claim(conn: sqlite3.Connection, task_id: str, expires: int) -> Optional[int]:
    """Mirror a task claim extension onto its active run row; returns that run id."""
    run_id = _kb._current_run_id(conn, task_id)
    if run_id is not None:
        conn.execute("UPDATE task_runs SET claim_expires = ? WHERE id = ?", (expires, run_id))
    return run_id


def _record_reclaim(
    conn: sqlite3.Connection, task_id: str, termination: dict, *, error: str, payload: dict,
) -> Optional[int]:
    """Close the active run as ``reclaimed`` and emit the ``reclaimed`` event
    (payload merged with the termination report). Caller holds the txn."""
    run_id = _kb._end_run(
        conn, task_id, outcome="reclaimed", status="reclaimed", error=error, metadata=termination,
    )
    payload.update(termination)
    _kb._append_event(conn, task_id, "reclaimed", payload, run_id=run_id)
    return run_id


def _extend_live_stale_claim(conn: sqlite3.Connection, row: sqlite3.Row, now: int) -> None:
    """TTL-expired claim whose host-local worker is alive: extend instead of
    reclaiming (``claim_extended`` event). CAS on the same expired lock so a
    concurrent reclaimer wins cleanly."""
    new_expires = now + _kb._resolve_claim_ttl_seconds()
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' "
            "  AND claim_lock IS ? "
            "  AND claim_expires IS NOT NULL "
            "  AND claim_expires < ?", (new_expires, row["id"], row["claim_lock"], now),
        )
        if cur.rowcount != 1:
            return
        run_id = _kb._extend_run_claim(conn, row["id"], new_expires)
        _kb._append_event(
            conn, row["id"], "claim_extended",
            {
                "reason": "pid_alive",
                "worker_pid": int(row["worker_pid"]),
                "claim_lock": row["claim_lock"],
                "claim_expires_was": _kb._to_epoch(row["claim_expires"]),
                "claim_expires_now": new_expires,
                "last_heartbeat_at": _kb._to_epoch(row["last_heartbeat_at"]),
            },
            run_id=run_id,
        )


def _claim_is_live(trow) -> bool:
    """True when a ``running`` task's claim still protects a run: the worker process
    it spawned exists (PID + start-time fingerprint). A claim whose worker is gone,
    or a library/CLI claim that never spawned one, has no run to protect. TTL expiry
    is deliberately not consulted: ``reclaim_stale_tasks`` extends, not reclaims, the
    claim of a live worker, so the process is the liveness authority here too."""
    return bool(
        trow["status"] == "running"
        and trow["claim_lock"] is not None
        and trow["worker_pid"]
        and _kb._worker_alive(trow["worker_pid"], trow["worker_started_at"])
    )


def authorize_lifecycle(
    conn: sqlite3.Connection, *, task_id: str, operation: str, now: Optional[int] = None,
):
    """THE single owner/admin authorization enforcement point for lifecycle writes.

    Every public mutator below calls this before its first write, and again inside
    its own ``write_txn`` (the authoritative re-check: reconciliation and mutation
    serialize, so the loser of a race re-reads the resulting run before writing).
    An unbound caller is denied outright — there is deliberately no
    "no identity, therefore administrator" fallback. ``expected_run_id``/``force``
    remain compatibility assertions inside the mutators and are never grants.

    The trusted record itself lives in :mod:`agent.kanban_execution_identity`;
    this wrapper exists so the kernel has one named enforcement point.
    """
    from agent.kanban_execution_identity import authorize_lifecycle_write

    return authorize_lifecycle_write(conn, task_id=task_id, operation=operation, now=now)


def _append_admin_audit(
    conn: sqlite3.Connection, auth, *, task_id: str, run_id: Optional[int] = None,
    new: Optional[dict] = None,
) -> None:
    """Record one administrative lifecycle mutation in the SAME transaction.

    Principal, transport, reason, target board/task and the old/new run
    coordinates land in the existing ``task_events`` stream. Bearer/session
    tokens and claim-lock strings are never written.
    """
    audit = getattr(auth, "audit", None)
    if not isinstance(audit, dict):
        return
    payload = dict(audit)
    if new:
        payload["current"] = dict(new)
    _kb._append_event(conn, task_id, "admin_recovery", payload, run_id=run_id)

__all__ = [
    "_append_admin_audit",
    "_claim_is_live",
    "_extend_live_stale_claim",
    "_extend_run_claim",
    "_record_reclaim",
    "_resolve_claim_ttl_seconds",
    "authorize_lifecycle",
    "heartbeat_claim",
]
