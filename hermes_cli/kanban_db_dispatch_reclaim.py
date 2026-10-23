"""Running-card reclaim passes (sibling of ``kanban_db_dispatch``): heartbeat
staleness and orphaned-claim reconciliation. Every
dispatch-level helper is read through ``_kbd()`` at call time so the facade
stays the patch seam.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional


def _kb():
    from hermes_cli import kanban_db
    return kanban_db


def _kbd():
    from hermes_cli import kanban_db_dispatch
    return kanban_db_dispatch


def detect_stale_running(
    conn: sqlite3.Connection,
    *,
    stale_timeout_seconds: int = 0,
    signal_fn=None,
) -> list[str]:
    """Reclaim ``running`` tasks with no heartbeat progress; returns their ids.

    Stale = running longer than ``stale_timeout_seconds`` (active run's
    ``started_at``, else ``tasks.started_at``) AND ``last_heartbeat_at`` NULL or
    older than ``_kbd()._STALE_HEARTBEAT_GAP_SECONDS``. Task returns to its source
    phase, run closes ``outcome='stale'``, a live host-local worker is killed.
    ``0`` disables the check; ``signal_fn`` is a test hook. Deliberately NOT
    counted via ``_kbd()._record_task_failure``: an absent heartbeat is not a worker
    failure, and counting it would let long-running tasks trip the breaker.
    """
    if stale_timeout_seconds <= 0:
        return []

    now = int(time.time())
    reclaimed: list[str] = []

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.worker_started_at, t.last_heartbeat_at, t.claim_lock, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running'"
    ).fetchall()

    for row in rows:
        if row["active_started_at"] is None:
            continue
        elapsed = now - int(row["active_started_at"])
        if elapsed < stale_timeout_seconds:
            continue

        last_hb = row["last_heartbeat_at"]
        hb_age = (now - int(last_hb)) if last_hb is not None else None
        if hb_age is not None and hb_age < _kbd()._STALE_HEARTBEAT_GAP_SECONDS:
            continue

        pid = row["worker_pid"]
        tid = row["id"]
        lock = row["claim_lock"] or ""

        termination = _kb()._terminate_reclaimed_worker(
            pid, lock, signal_fn=signal_fn, started_at=_kb()._row_get(row, "worker_started_at"))

        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _kbd()._worker_survived_termination(termination):
            _kbd()._defer_reclaim_for_live_worker(
                conn, tid, lock, now, termination,
                reason="heartbeat_stale_worker_alive",
            )
            continue

        with _kb().write_txn(conn):
            retry_status = _kb()._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (retry_status, tid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue

            payload = {
                "elapsed_seconds": int(elapsed),
                "last_heartbeat_at": _kb()._opt_int(last_hb),
                "heartbeat_age_seconds": _kb()._opt_int(hb_age),
                "timeout_seconds": stale_timeout_seconds,
                "pid": int(pid) if pid else None,
                "retry_status": retry_status,
            }
            payload.update(termination)

            run_id = _kb()._end_run(
                conn, tid,
                outcome="stale", status="stale",
                error=(
                    f"no heartbeat for {int(hb_age)}s "
                    if hb_age is not None
                    else "no heartbeat ever"
                ) + f" after {int(elapsed)}s running",
                metadata=payload,
            )
            _kb()._append_event(conn, tid, "stale", payload, run_id=run_id)
            reclaimed.append(tid)

    return reclaimed


def reconcile_orphaned_running(conn: sqlite3.Connection) -> list[str]:
    """Requeue ``running`` cards with broken claim bookkeeping; returns their ids.

    A task ``running`` with NULL ``claim_lock``/``claim_expires`` (crash
    mid-claim, manual SQL, DB restore) is a zombie forever: ``release_stale_claims``
    needs ``claim_expires``, ``detect_crashed_workers`` needs a host-local lock +
    pid, ``detect_stale_running`` is off by default. Orphans go back to ``ready``
    with a comment, leaked run closed, ``reconciled`` event; a row with a live
    host-local PID is deferred so no duplicate spawns beside it.
    """
    now = int(time.time())
    reconciled: list[str] = []
    rows = conn.execute(
        "SELECT id, claim_lock, claim_expires, worker_pid, worker_started_at FROM tasks "
        "WHERE status = 'running' "
        "  AND (claim_lock IS NULL OR claim_expires IS NULL)"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        pid = row["worker_pid"]
        if pid and _kbd()._worker_alive(pid, _kb()._row_get(row, "worker_started_at")):
            # Never requeue beside a live process. Retry next tick.
            _kb()._log.debug(
                "kanban reconcile: task %s has broken claim bookkeeping but "
                "pid %s is alive on this host — deferring", tid, pid,
            )
            continue
        with _kb().write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ? AND claim_expires IS ?",
                (tid, row["claim_lock"], row["claim_expires"]),
            )
            if cur.rowcount != 1:
                continue
            payload = {
                "reason": "orphaned_running",
                "claim_lock": row["claim_lock"],
                "claim_expires": _kb()._opt_int(row["claim_expires"]),
                "worker_pid": int(pid) if pid else None,
                "now": now,
            }
            run_id = _kb()._end_run(
                conn, tid,
                outcome="reclaimed", status="reclaimed",
                error="orphaned running card (broken claim bookkeeping)",
                metadata=payload,
            )
            _kb()._insert_comment(
                conn, tid, "dispatcher",
                "reconciliation: card was 'running' with no valid claim "
                "(dead/gone worker) — requeued to ready",
                now,
            )
            _kb()._append_event(conn, tid, "reconciled", payload, run_id=run_id)
            reconciled.append(tid)
        _kb()._log.info(
            "kanban reconcile: requeued orphaned running task %s "
            "(claim_lock=%r, worker_pid=%r)", tid, row["claim_lock"], pid,
        )
    return reconciled
