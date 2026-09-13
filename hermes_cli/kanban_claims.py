"""Run claims, heartbeat leases and reclaim reservations."""
from __future__ import annotations

import hermes_cli.kanban_db_boards as _owner_kanban_boards
import hermes_cli.kanban_transitions as _owner_kanban_transitions
import sqlite3
import time
import logging
from typing import Optional, Any
from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_db_models import Task
_log = logging.getLogger(__name__)

def _claim_and_open_run(
    conn: sqlite3.Connection, task_id: str, source_status: str, lock: str, expires: int, now: int,
    *, event_extra: Optional[dict] = None,
) -> Optional[int]:
    """CAS ``source_status -> running``, open a run row, emit ``claimed``; None
    when the CAS lost. Caller holds the txn."""
    cur = conn.execute(
        f"""
        UPDATE tasks
           SET status        = 'running',
               claim_lock    = ?,
               claim_expires = ?,
               worker_pid = NULL,
               worker_scope = NULL,
               worker_pid_started_at = NULL,
               worker_registered_at = NULL,
               last_heartbeat_at = NULL,
               reclaim_reserved_at = NULL,
               started_at    = COALESCE(started_at, ?)
         WHERE id = ?
           AND status = '{source_status}'
           AND claim_lock IS NULL
           AND worker_scope IS NULL
        """,
        (lock, expires, now, task_id),
    )
    if cur.rowcount != 1:
        return None
    trow = conn.execute(
        "SELECT assignee, max_runtime_seconds, current_step_key "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    run_cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key, status,
            claim_lock, claim_expires, max_runtime_seconds,
            started_at
        ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
        """,
        (
            task_id, trow["assignee"] if trow else None, trow["current_step_key"] if trow else None,
            lock, expires, trow["max_runtime_seconds"] if trow else None, now,
        ),
    )
    run_id = run_cur.lastrowid
    conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, task_id))
    _kanban_db._append_event(
        conn, task_id, "claimed",
        {"lock": lock, "expires": expires, "run_id": run_id, **(event_extra or {})}, run_id=run_id,
    )
    return run_id


def claim_task(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``ready -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``ready`` status).
    """
    now = int(time.time())
    lock = claimer or _kanban_db._claimer_id()
    expires = now + _kanban_db._resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        if not _claim_scope_clear(conn, task_id):
            return None
        # Single enforcement point: never ready -> running with an undone
        # parent, whichever writer set 'ready'. Demote to 'todo';
        # recompute_ready re-promotes when the parents finish.
        if not _owner_kanban_transitions._parents_satisfied(conn, task_id):
            conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'ready'", (task_id,),
            )
            _kanban_db._append_event(conn, task_id, "claim_rejected", {"reason": "parents_not_done"})
            return None
        # Close a leaked prior run so the CAS below doesn't strand it.
        _owner_kanban_transitions._reclaim_dangling_run(
            conn, task_id, statuses=("ready",), now=now, note="invariant recovery on re-claim",
        )
        run_id = _claim_and_open_run(conn, task_id, "ready", lock, expires, now)
        if run_id is None:
            return None
        claimed = _kanban_db.get_task(conn, task_id)
    _kanban_db._fire_task_hook("kanban_task_claimed", claimed, task_id, run_id)
    return claimed


def claim_review_task(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomic ``review -> running`` (None when lost). Parents are re-checked
    (one may have reopened meanwhile) and a NEW run tracks the reviewer
    separately from the implementer."""
    now = int(time.time())
    lock = claimer or _kanban_db._claimer_id()
    expires = now + _kanban_db._resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        if not _claim_scope_clear(conn, task_id):
            return None
        if not _owner_kanban_transitions._parents_satisfied(conn, task_id):
            demoted = conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'review' AND claim_lock IS NULL", (task_id,),
            )
            if demoted.rowcount == 1:
                _kanban_db._append_event(
                    conn, task_id, "dependency_wait",
                    {"reason": "parent_reopened", "source_status": "review"},
                )
            return None
        run_id = _claim_and_open_run(
            conn, task_id, "review", lock, expires, now, event_extra={"source_status": "review"},
        )
        if run_id is None:
            return None
        return _kanban_db.get_task(conn, task_id)


def _retry_status_for_run(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int] = None,
) -> str:
    """``review`` when the run's ``claimed`` event says ``source_status=review``,
    else ``ready`` — one place, so crash/timeout/reclaim can't silently turn a
    reviewer run into an implementation run."""
    if run_id is None:
        run_id = _kanban_db._current_run_id(conn, task_id)
    if run_id is None:
        return "ready"
    event = _kanban_db._latest_event(conn, task_id, "claimed", run_id)
    payload = _kanban_db._json_dict(_kanban_db._row_get(event, "payload"))
    return "review" if payload.get("source_status") == "review" else "ready"





def heartbeat_claim(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Extend a running claim.  Returns True if we still own it.

    Workers that know they'll exceed 15 minutes should call this every
    few minutes to keep ownership.

    Pass 12 (AP): a row the stale-claim sweep has reserved for reclaim
    (``reclaim_reserved_at`` set with ``claim_expires`` still inside the
    reservation's defer grace) is never extended — the heartbeat returns
    False (claim lost), so the worker's heartbeat bridge stops keeping
    the doomed run alive and the sweep terminates a row that provably
    stopped heartbeating instead of killing a live one.
    """
    expires = int(time.time()) + _kanban_db._resolve_claim_ttl_seconds(ttl_seconds)
    lock = claimer or _kanban_db._claimer_id()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ? "
            "AND (? IS NULL OR current_run_id = ?) "
            # Never resurrect a claim the stale sweep has reserved for
            # termination (pass 12, AP). The marker only means anything
            # while claim_expires still sits inside the reservation's
            # grace window; any later writer (adoption, a fresh claim)
            # pushes claim_expires beyond it and the guard self-releases.
            "AND reclaim_reserved_at IS NULL",
            (expires, task_id, lock, expected_run_id, expected_run_id),
        )
        if cur.rowcount == 1:
            run_id = _kanban_db._current_run_id(conn, task_id)
            if run_id is not None:
                conn.execute(
                    "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                    (expires, run_id),
                )
            return True
        # Cold path: distinguish a reservation refusal from the ordinary
        # not-running / wrong-lock misses so the log names the cause.
        row = conn.execute(
            "SELECT reclaim_reserved_at, claim_expires FROM tasks "
            "WHERE id = ?",
            (task_id,),
        ).fetchone()
        if (
            row is not None
            and row["reclaim_reserved_at"] is not None
            and row["claim_expires"] is not None
            and int(row["claim_expires"])
            <= int(row["reclaim_reserved_at"]) + _kanban_db.RECLAIM_DEFER_GRACE_SECONDS
        ):
            _kanban_db._log.info(
                "kanban: heartbeat for %s refused — the claim is "
                "reserved for reclaim; treating it as claim lost",
                task_id,
            )
        return False



def _extend_run_claim(conn: sqlite3.Connection, task_id: str, expires: int) -> Optional[int]:
    """Mirror a task claim extension onto its active run row; returns that run id."""
    run_id = _kanban_db._current_run_id(conn, task_id)
    if run_id is not None:
        conn.execute("UPDATE task_runs SET claim_expires = ? WHERE id = ?", (expires, run_id))
    return run_id


def release_stale_claims(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> int:
    """Reset any ``running`` task whose claim has expired.

    A stale-by-TTL claim whose host-local worker PID is still alive is
    *extended* (with a ``claim_extended`` event) instead of being
    reclaimed. Reclaiming a live worker mid-flight produces the spawn-
    then-immediately-reclaim loop seen on slow models that spend longer
    than ``DEFAULT_CLAIM_TTL_SECONDS`` inside a single tool-free LLM
    call (#23025): no tool calls means no ``kanban_heartbeat``, even
    though the subprocess is healthy.

    Backstop (#29747 gap 3): if the worker's PID is still alive but its
    ``last_heartbeat_at`` is stale by more than
    ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (1h), the worker has
    been making no observable progress and we reclaim anyway — even if
    ``_pid_alive`` is still true. This catches the wedged-in-a-logic-loop
    case where the process is technically running but accomplishing
    nothing. ``_touch_activity`` (run_agent.py) bridges chunk-level
    liveness into ``last_heartbeat_at`` via #31752, so any genuinely
    active worker keeps its heartbeat fresh as a side effect of normal
    API traffic. ``enforce_max_runtime`` and ``detect_crashed_workers``
    remain the upper bounds for genuinely wedged or dead workers.

    Returns the number of stale claims actually reclaimed (live-pid
    extensions don't count). Safe to call often.
    """
    now = int(time.time())
    reclaimed = 0
    host_prefix = f"{_kanban_db._claimer_id().split(':', 1)[0]}:"
    stale = conn.execute(
        "SELECT * FROM tasks "
        "WHERE status = 'running' AND claim_expires IS NOT NULL "
        "  AND claim_expires < ?",
        (now,),
    ).fetchall()
    for row in stale:
        lock = row["claim_lock"] or ""
        host_local = lock.startswith(host_prefix)
        hb = row["last_heartbeat_at"]
        # Heartbeat staleness backstop: if we have a heartbeat at all
        # and it's older than the max-stale threshold, the worker is
        # not making observable progress.  Reclaim instead of extending,
        # even if the PID is still alive (it's likely in a logic loop).
        heartbeat_stale = (
            hb is not None
            and (now - int(hb)) > _kanban_db.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        )
        # Liveness via the authoritative signal for this row (scope
        # cgroup state, else pid+fingerprint) — never bare PID liveness,
        # which PID reuse turns into endless extensions for a dead run.
        worker_alive, alive_reason = _kanban_worker_identity._run_worker_alive(row)
        if (
            host_local
            and row["worker_pid"]
            and worker_alive
            and not heartbeat_stale
        ):
            new_expires = now + _kanban_db._resolve_claim_ttl_seconds()
            with write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET claim_expires = ? "
                    "WHERE id = ? AND status = 'running' "
                    "  AND claim_lock IS ? "
                    "  AND claim_expires IS NOT NULL "
                    "  AND claim_expires < ? AND current_run_id IS ? AND worker_pid IS ? "
                    "AND worker_pid_started_at IS ? AND worker_scope IS ? "
                    "AND last_heartbeat_at IS ? AND reclaim_reserved_at IS NULL",
                    (new_expires, row["id"], row["claim_lock"], now, row["current_run_id"],
                     row["worker_pid"], row["worker_pid_started_at"], row["worker_scope"], row["last_heartbeat_at"]),
                )
                if cur.rowcount != 1:
                    continue
                run_id = _kanban_db._current_run_id(conn, row["id"])
                if run_id is not None:
                    conn.execute(
                        "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                        (new_expires, run_id),
                    )
                _kanban_db._append_event(
                    conn, row["id"], "claim_extended",
                    {
                        "reason": alive_reason,
                        "worker_pid": int(row["worker_pid"]),
                        "claim_lock": row["claim_lock"],
                        "claim_expires_was": int(row["claim_expires"]),
                        "claim_expires_now": new_expires,
                        "last_heartbeat_at": (
                            int(row["last_heartbeat_at"])
                            if row["last_heartbeat_at"] is not None
                            else None
                        ),
                    },
                    run_id=run_id,
                )
            continue

        # Pass 9 (AF): a pending own-worker handoff must never be eaten
        # by the generic stale reclaim — this sweep runs BEFORE crash
        # detection, so without this branch a drain outlasting the defer
        # grace silently dropped the worker's requested transition.
        if _kanban_worker_handoff._handle_stale_own_worker_handoff(conn, row, now=now):
            continue

        # Pass 11 (AN): the reclaim decision is atomic, the signal is
        # post-commit. AK's fresh read stood the sweep down for heartbeats
        # that landed before it, but the read and the termination were
        # still separate operations — a heartbeat committing in between
        # signalled a live worker. The decision now happens inside ONE
        # write transaction: the row is re-read and reserved by an UPDATE
        # with an optimistic CAS on the exact values just read
        # (claim_expires + worker_pid_started_at + worker_scope). A CAS
        # miss means a heartbeat landed between the re-read and the
        # UPDATE — the row is alive: no signal, no reclaim this tick. A
        # hit holds the claim for one defer grace (a worker that then
        # survives the signal cannot be duplicated beside it), and only
        # then is the termination tuple signalled, strictly post-commit
        # and keyed to the fresh values, never the scan snapshot. The
        # reservation also stamps ``reclaim_reserved_at`` so heartbeats
        # are refused while it holds, and is re-verified under the write
        # lock right before the signal (pass 12, AP) — a heartbeat that
        # raced in anyway releases it and stands the row down.
        fresh = reserve_reclaim(conn, row["id"], row, now=now, expired_only=True)
        if fresh is None:
            continue
        grace = fresh["claim_expires"]

        # Pass 12 (AP): the reservation has committed but the signal has
        # not fired — a heartbeat already in flight can still land in
        # that window. ``heartbeat_claim`` refuses reserved rows (the
        # worker side of the guard); this re-check is the sweep side:
        # verify under the write lock that the reservation still holds
        # (same grace sentinel, same fingerprint, no heartbeat since the
        # reservation) and stand down — releasing the reservation — when
        # anything moved, so the signal only ever targets a row that
        # provably stopped heartbeating.
        if not _recheck_reclaim_reservation(
            conn, row["id"], row["claim_lock"], grace, fresh,
        ):
            continue
        termination = _kanban_worker_identity._terminate_reclaimed_worker(
            fresh["worker_pid"], fresh["claim_lock"], signal_fn=signal_fn,
            scope_unit=fresh["worker_scope"] or None,
            pid_started_at=fresh["worker_pid_started_at"],
            task_id=row["id"], run_id=fresh["current_run_id"],
            expected_db=_reclaim_db_path(conn),
        )
        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _kanban_worker_identity._worker_survived_termination(termination):
            _kanban_worker_identity._defer_reclaim_for_live_worker(
                conn, row["id"], row["claim_lock"], now, termination,
                reason="ttl_expired_worker_alive",
                expected_run_id=fresh["current_run_id"],
            )
            continue
        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, row["id"])
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "worker_pid_started_at = NULL, "
                "worker_registered_at = NULL, worker_scope = NULL, "
                "reclaim_reserved_at = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
                "AND claim_expires IS ? AND worker_pid_started_at IS ? "
                "AND worker_scope IS ? AND current_run_id IS ? AND worker_pid IS ? "
                "AND last_heartbeat_at IS ? AND reclaim_reserved_at IS ?",
                (retry_status, row["id"], row["claim_lock"], grace,
                 fresh["worker_pid_started_at"], fresh["worker_scope"], fresh["current_run_id"],
                 fresh["worker_pid"], fresh["last_heartbeat_at"], fresh["reclaim_reserved_at"]),
            )
            if cur.rowcount != 1:
                # CAS miss: the row moved after the reservation — under
                # our held claim only a live worker's heartbeat can
                # rewrite claim_expires. Stand down; the reservation's
                # grace keeps the row un-spawnable until the next tick
                # re-scans.
                continue
            run_id = _kanban_db._end_run(
                conn, row["id"],
                outcome="reclaimed", status="reclaimed",
                error=f"stale_lock={row['claim_lock']}",
                metadata=termination,
            )
            payload = {
                "stale_lock": row["claim_lock"],
                "worker_pid": (
                    int(row["worker_pid"])
                    if row["worker_pid"] is not None else None
                ),
                "claim_expires": int(row["claim_expires"]),
                "last_heartbeat_at": (
                    int(row["last_heartbeat_at"])
                    if row["last_heartbeat_at"] is not None else None
                ),
                "now": now,
                "host_local": host_local,
                "heartbeat_stale": bool(heartbeat_stale),
                "retry_status": retry_status,
            }
            payload.update(termination)
            _kanban_db._append_event(
                conn, row["id"], "reclaimed",
                payload,
                run_id=run_id,
            )
            reclaimed += 1
        # Worker-lifecycle observer (RFC #58548): the reclaim txn above has
        # committed. The ``continue`` branches (rowcount mismatch, claim
        # extension, deferred reclaim) never reach this point, so only a
        # genuinely reclaimed stale claim fires.
        if _kanban_db._kanban_observer_consumed("on_kanban_worker_stale_claim"):
            _kanban_db._fire_kanban_lifecycle_hook(
                "on_kanban_worker_stale_claim",
                row["id"],
                board=_owner_kanban_boards.get_current_board(),
                assignee=row["assignee"],
                run_id=run_id,
                worker_pid=(
                    int(row["worker_pid"])
                    if row["worker_pid"] is not None else None
                ),
                heartbeat_stale=bool(heartbeat_stale),
                retry_status=retry_status,
            )
    return reclaimed



def _record_reclaim(
    conn: sqlite3.Connection, task_id: str, termination: dict, *, error: str, payload: dict,
) -> Optional[int]:
    """Close the active run as ``reclaimed`` and emit the ``reclaimed`` event
    (payload merged with the termination report). Caller holds the txn."""
    run_id = _kanban_db._end_run(
        conn, task_id, outcome="reclaimed", status="reclaimed", error=error, metadata=termination,
    )
    payload.update(termination)
    _kanban_db._append_event(conn, task_id, "reclaimed", payload, run_id=run_id)
    return run_id


def _extend_live_stale_claim(conn: sqlite3.Connection, row: sqlite3.Row, now: int) -> None:
    """TTL-expired claim whose host-local worker is alive: extend instead of
    reclaiming (``claim_extended`` event). CAS on the same expired lock so a
    concurrent reclaimer wins cleanly."""
    new_expires = now + _kanban_db._resolve_claim_ttl_seconds()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' "
            "  AND claim_lock IS ? "
            "  AND claim_expires IS NOT NULL "
            "  AND claim_expires < ?", (new_expires, row["id"], row["claim_lock"], now),
        )
        if cur.rowcount != 1:
            return
        run_id = _extend_run_claim(conn, row["id"], new_expires)
        _kanban_db._append_event(
            conn, row["id"], "claim_extended",
            {
                "reason": "pid_alive",
                "worker_pid": int(row["worker_pid"]),
                "claim_lock": row["claim_lock"],
                "claim_expires_was": int(row["claim_expires"]),
                "claim_expires_now": new_expires,
                "last_heartbeat_at": _kanban_db._opt_int(row["last_heartbeat_at"]),
            },
            run_id=run_id,
        )


def reclaim_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    signal_fn=None, expected_run_id: Optional[int] = None,
) -> bool:
    """Reclaim one execution after verified stop; never reclaim its successor."""
    snapshot = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if snapshot is None or (snapshot["status"] != "running" and snapshot["claim_lock"] is None):
        return False
    if expected_run_id is not None and snapshot["current_run_id"] != expected_run_id:
        return False
    reserved = reserve_reclaim(conn, task_id, snapshot)
    if reserved is None:
        return False
    if not _recheck_reclaim_reservation(
        conn, task_id, reserved["claim_lock"], reserved["claim_expires"], reserved,
    ):
        return False
    termination = _kanban_worker_identity._terminate_reclaimed_worker(
        reserved["worker_pid"], reserved["claim_lock"], signal_fn=signal_fn,
        scope_unit=reserved["worker_scope"] or None,
        pid_started_at=reserved["worker_pid_started_at"],
        task_id=task_id, run_id=reserved["current_run_id"],
        expected_db=_reclaim_db_path(conn),
    )
    if _kanban_worker_identity._worker_survived_termination(termination):
        _kanban_worker_identity._defer_reclaim_for_live_worker(
            conn, task_id, reserved["claim_lock"], int(time.time()), termination,
            reason="manual_reclaim_scope_still_stopping",
            expected_run_id=reserved["current_run_id"],
        )
        return False
    with write_txn(conn):
        retry_status = _retry_status_for_run(conn, task_id, reserved["current_run_id"])
        predicate, values = _snapshot_predicate(reserved)
        cur = conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_pid_started_at = NULL, worker_registered_at = NULL, "
            "worker_scope = NULL, reclaim_reserved_at = NULL, last_heartbeat_at = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND " + predicate,
            (retry_status, task_id, *values),
        )
        if cur.rowcount != 1:
            return False
        _record_reclaim(
            conn, task_id, termination,
            error=f"manual_reclaim: {reason}" if reason else f"manual_reclaim lock={reserved['claim_lock']}",
            payload={"manual": True, "reason": reason, "prev_lock": reserved["claim_lock"],
                     "retry_status": retry_status},
        )
    return True



def reassign_task(
    conn: sqlite3.Connection, task_id: str, profile: Optional[str], *, reclaim_first: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Reassign (None unassigns); a running task is refused unless
    ``reclaim_first`` releases its claim — the "this profile's model is broken" path."""
    if reclaim_first:
        # Safe to call even if nothing to reclaim.
        reclaim_task(conn, task_id, reason=reason or "reassign")
    # assign_task handles its own txn + the still-running guard.
    try:
        return _kanban_db.assign_task(conn, task_id, profile)
    except RuntimeError:
        # Task is still running and reclaim_first was False; caller
        # needs to decide whether to retry with reclaim.
        return False


def _reread_stale_claim_for_reclaim(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
) -> Any:
    """In-transaction re-read of a stale row right before its reservation.

    Pass 11 (AN): the scan that fed :func:`release_stale_claims` is a
    snapshot, so the decision to terminate is made against a row read
    INSIDE the sweep's write transaction — the reservation UPDATE that
    follows is an optimistic CAS on exactly these values. A dedicated
    seam (not an inline ``conn.execute``) so the read→update
    interleaving, a heartbeat landing between the two, is injectable in
    tests. Returns ``None`` when the row is no longer a running claim
    under ``claim_lock``.
    """
    return conn.execute(
        "SELECT * FROM tasks "
        "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
        (task_id, claim_lock),
    ).fetchone()


def _recheck_reclaim_reservation(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    grace: int,
    fresh: Any,
) -> bool:
    """Verify, in one short write transaction, that a reclaim reservation
    still holds immediately before its worker is signalled (pass 12, AP).

    The reservation must commit before the signal (the termination runs
    strictly post-commit), so a heartbeat that was already in flight can
    still land in that window — ``heartbeat_claim`` now refuses reserved
    rows, closing the window from the worker's side; this seam closes it
    from the sweep's side for any writer that rewrote the row anyway (an
    older worker binary, a direct write). The row is re-read under the
    write lock and the reservation confirmed intact: the grace sentinel
    still on ``claim_expires``, the marker still set, and the worker
    fingerprint and heartbeat timestamp unchanged since the reservation
    read. When anything moved, the reservation is released to the live
    values (marker dropped; whatever claim state the racing writer left
    stands) and the caller must NOT signal. A dedicated seam for the
    same reason as :func:`_reread_stale_claim_for_reclaim` — the
    reservation→re-check interleaving is injectable in tests.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, claim_expires, "
            "worker_pid_started_at, worker_scope, reclaim_reserved_at, "
            "last_heartbeat_at, current_run_id, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if (
            row is not None
            and row["status"] == "running"
            and row["claim_lock"] == claim_lock
            and row["current_run_id"] == fresh["current_run_id"]
            and row["worker_pid"] == fresh["worker_pid"]
            and row["reclaim_reserved_at"] is not None
            and row["reclaim_reserved_at"] == fresh["reclaim_reserved_at"]
            and row["claim_expires"] is not None
            and int(row["claim_expires"]) == int(grace)
            and row["worker_pid_started_at"] == fresh["worker_pid_started_at"]
            and row["worker_scope"] == fresh["worker_scope"]
            and row["last_heartbeat_at"] == fresh["last_heartbeat_at"]
        ):
            return True
        # Something moved between the reservation commit and this
        # re-check: a heartbeat raced in (the row is alive) or the row
        # moved on entirely. Never signal on stale evidence — release
        # the reservation and let the live claim state stand.
        cur = conn.execute(
            "UPDATE tasks SET reclaim_reserved_at = NULL "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
            "AND reclaim_reserved_at IS ? AND current_run_id IS ? "
            "AND worker_pid IS ? AND worker_pid_started_at IS ? AND worker_scope IS ?",
            (task_id, claim_lock, fresh["reclaim_reserved_at"], fresh["current_run_id"], fresh["worker_pid"],
             fresh["worker_pid_started_at"], fresh["worker_scope"]),
        )
        if cur.rowcount == 1:
            _kanban_db._log.info(
                "kanban: stale reclaim reservation for %s released "
                "without signalling — the row moved (a heartbeat or a "
                "takeover landed) between the reservation and the "
                "re-check",
                task_id,
            )
    return False


_RECLAIM_IDENTITY_COLUMNS = (
    "status", "current_run_id", "claim_lock", "worker_pid", "worker_pid_started_at",
    "worker_scope", "last_heartbeat_at", "claim_expires", "reclaim_reserved_at",
)


def _snapshot_predicate(snapshot):
    """SQL and bindings for an exact attempt snapshot, including lease motion."""
    return (
        " AND ".join(f"{column} IS ?" for column in _RECLAIM_IDENTITY_COLUMNS),
        tuple(snapshot[column] for column in _RECLAIM_IDENTITY_COLUMNS),
    )


def reserve_reclaim(conn, task_id, snapshot, *, now=None, expired_only=False):
    """Reserve observed ownership before any signal, without holding SQLite during stop.

    An old scan cannot reserve a successor even when both use the same dispatcher
    lock, PID or timestamp. A concurrent heartbeat cancels this attempt. A crashed
    controller leaves an expiring, recoverable reservation, never a spawnable task.
    """
    now = int(time.time()) if now is None else int(now)
    with write_txn(conn):
        fresh = _reread_stale_claim_for_reclaim(conn, task_id, snapshot["claim_lock"])
        if fresh is None:
            return None
        if any(fresh[column] != snapshot[column] for column in _RECLAIM_IDENTITY_COLUMNS):
            return None
        if (fresh["reclaim_reserved_at"] is not None
                and fresh["claim_expires"] is not None and fresh["claim_expires"] > now):
            return None
        if expired_only and (fresh["claim_expires"] is None or fresh["claim_expires"] >= now):
            return None
        predicate, values = _snapshot_predicate(fresh)
        expires = now + _kanban_db.RECLAIM_DEFER_GRACE_SECONDS
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ?, reclaim_reserved_at = ? "
            "WHERE id = ? AND " + predicate, (expires, now, task_id, *values),
        )
        if cur.rowcount != 1:
            return None
        if fresh["current_run_id"] is not None:
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ? AND task_id = ? AND ended_at IS NULL",
                (expires, fresh["current_run_id"], task_id),
            )
        return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()

from hermes_cli import kanban_db as _kanban_db
from hermes_cli import kanban_worker_handoff as _kanban_worker_handoff
from hermes_cli import kanban_worker_identity as _kanban_worker_identity


def _reclaim_db_path(conn):
    return next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), None)


def _claim_scope_clear(conn, task_id):
    """A parked attempt retains ownership until its verified scope sweep."""
    row = conn.execute("SELECT worker_scope FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row is not None and row["worker_scope"] is None
