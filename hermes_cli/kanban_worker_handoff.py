"""Kanban worker lifecycle; integrated from PR #101911 by danashburn.

Attempt ownership and persistence use the current main Kanban modules.
"""

from __future__ import annotations

import hermes_cli.kanban_transitions as _owner_kanban_transitions
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

_log = logging.getLogger(__name__)


def _mark_run_scope_stopping(
    conn: sqlite3.Connection,
    task_id: str,
    scope: str,
    *,
    reason: str = "terminal_stop_unconfirmed",
    expected_run_id: Optional[int] = None,
) -> bool:
    """Record the run's "stopping" state — durable, once per run.

        A tick that wants to close a run but could not verify its scope dead
        defers the close and marks this event instead; the row keeps its
        claim and the next tick re-checks the service. One event per run id
        so a long teardown doesn't flood the timeline.  Nested-safe: the
    ancestor-reopen invalidation composes this under the dashboard's outer
    commit."""
    with _kanban_db_connect.write_txn(conn, allow_nested=True):
        owner = conn.execute(
            "SELECT status,current_run_id,worker_scope FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        if (
            owner is None
            or owner["status"] != "running"
            or owner["worker_scope"] != scope
        ):
            return False
        run_id = owner["current_run_id"]
        if expected_run_id is not None and run_id != expected_run_id:
            return False
        already = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND kind = 'scope_stopping' "
            "  AND run_id IS ? LIMIT 1",
            (task_id, run_id),
        ).fetchone()
        if already is not None:
            return True
        _kanban_db._append_event(
            conn,
            task_id,
            "scope_stopping",
            {"scope": scope, "reason": reason},
            run_id=run_id,
        )
    return True


def _defer_own_worker_handoff(
    conn: sqlite3.Connection,
    task_id: str,
    scope: str,
    handoff: dict,
    *,
    claim_lock: Optional[str],
    expected_run_id: Optional[int] = None,
) -> bool:
    """Durably defer an own-worker terminal handoff (pass 8, V).

    The run's own worker cannot verify its scope dead before its
    terminal ``review``/``ready`` write — the caller is alive inside the
    cgroup — so writing the spawnable row first (the pre-V behaviour)
    exposed the review/ready lane beside a still-draining scope. The
    whole transition is instead recorded here and applied by the crash
    sweep once the verified stop lands
    (:func:`_apply_pending_own_worker_handoff`). The claim is held and
    extended like any other deferral so nothing respawns beside it.
    """
    now = int(time.time())
    grace = now + _kanban_db.RECLAIM_DEFER_GRACE_SECONDS
    with _kanban_db_connect.write_txn(conn, allow_nested=True):
        owner = conn.execute(
            "SELECT status,current_run_id,worker_scope,claim_lock FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        if (
            owner is None
            or owner["status"] != "running"
            or owner["worker_scope"] != scope
            or owner["claim_lock"] != claim_lock
        ):
            return False
        run_id = owner["current_run_id"]
        if expected_run_id is not None and run_id != expected_run_id:
            return False
        already = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND kind = 'own_worker_handoff' "
            "  AND run_id IS ? LIMIT 1",
            (task_id, run_id),
        ).fetchone()
        if already is None:
            _kanban_db._append_event(
                conn,
                task_id,
                "own_worker_handoff",
                {"scope": scope, **handoff},
                run_id=run_id,
            )
        conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
            (grace, task_id, claim_lock),
        )
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                (grace, run_id),
            )
    return True


def _apply_pending_own_worker_handoff(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: Optional[int],
) -> bool:
    """Flip a drained run's row per its deferred own-worker handoff.

    Called by the crash sweep's teardown phase once the run's scope is
    verified empty: the worker already did its paperwork (the
    ``own_worker_handoff`` marker), so the row moves to its requested
    lane instead of crash-requeueing — no ``crashed`` event, no protocol
    violation, no failure count. CAS on the marker's run id: a row that
    moved on (requeued, adopted, reopened) keeps its new state and the
    marker stays inert history. The one deliberate divergence: a
    ``review_requested`` marker whose parents are no longer satisfied is
    NOT inert — the row is demoted to ``todo`` (the ancestor-reopen
    outcome) so it can never respawn beside a reopened parent, and the
    payload is preserved in a discard event for audit (pass 9, AG).
    """
    if run_id is None:
        return False
    marker = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'own_worker_handoff' "
        "  AND run_id IS ? ORDER BY id DESC LIMIT 1",
        (task_id, int(run_id)),
    ).fetchone()
    if marker is None:
        return False
    try:
        payload = json.loads(marker["payload"]) if marker["payload"] else {}
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        return False
    kind = payload.get("handoff")
    if kind not in ("review_requested", "changes_requested"):
        return False
    with _kanban_db_connect.write_txn(conn):
        row = conn.execute(
            "SELECT status, current_run_id, worker_scope FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] != "running"
            or int(row["current_run_id"] or 0) != int(run_id)
            or row["worker_scope"] != payload.get("scope")
        ):
            return False
        if kind == "review_requested":
            if not _owner_kanban_transitions._parents_satisfied(conn, task_id):
                # An ancestor reopened while the scope drained. The
                # ancestor-reopen invalidation cannot retract this run —
                # it holds the marker past its own worker's exit — and
                # returning False here would drop the row into the
                # generic reclaim, whose _retry_status_for_run restores
                # ready/review beside a reopened parent (pass 9, AG).
                # Close it the way the reopen invalidation would:
                # demote to todo, never back to a spawnable lane, with
                # the discarded handoff payload kept in the event for
                # audit.
                cur = conn.execute(
                    "UPDATE tasks "
                    "SET status = 'todo', claim_lock = NULL, "
                    "    claim_expires = NULL, worker_pid = NULL, "
                    "    worker_pid_started_at = NULL, "
                    "    last_heartbeat_at = NULL, reclaim_reserved_at = NULL, "
                    "    worker_registered_at = NULL, worker_scope = NULL "
                    "WHERE id = ? AND status = 'running' "
                    "  AND current_run_id = ?",
                    (task_id, int(run_id)),
                )
                if cur.rowcount != 1:
                    return False
                new_run = _kanban_db._end_run(
                    conn,
                    task_id,
                    outcome="reclaimed",
                    status="todo",
                    summary=(
                        "own-worker handoff discarded: a parent reopened "
                        "while the scope drained"
                    ),
                )
                _kanban_db._append_event(
                    conn,
                    task_id,
                    "own_worker_handoff_discarded",
                    {
                        "reason": "parents_unsatisfied",
                        "handoff": kind,
                        "handoff_payload": payload,
                    },
                    run_id=new_run,
                )
                _log.info(
                    "kanban: discarded deferred %s handoff for %s (run %s) "
                    "— a parent reopened while the scope drained; row "
                    "demoted to todo",
                    kind,
                    task_id,
                    run_id,
                )
                return True
            reviewer = payload.get("reviewer")
            assignee_sql = ", assignee = ?" if reviewer else ""
            params: tuple[Any, ...] = (
                (reviewer, task_id, int(run_id)) if reviewer else (task_id, int(run_id))
            )
            cur = conn.execute(
                "UPDATE tasks "
                "SET status = 'review', claim_lock = NULL, "
                "    claim_expires = NULL, worker_pid = NULL, "
                "    worker_pid_started_at = NULL, "
                "    last_heartbeat_at = NULL, reclaim_reserved_at = NULL, "
                "    worker_registered_at = NULL, worker_scope = NULL"
                + assignee_sql
                + " WHERE id = ? AND status = 'running' "
                "  AND current_run_id = ?",
                params,
            )
            if cur.rowcount != 1:
                return False
            lines = (payload.get("summary") or "").strip().splitlines()
            new_run = _kanban_db._end_run(
                conn,
                task_id,
                outcome="review_requested",
                status="review",
                summary=payload.get("summary"),
                metadata=payload.get("metadata"),
            )
            _kanban_db._append_event(
                conn,
                task_id,
                "review_requested",
                {
                    "summary": lines[0][:400] if lines else None,
                    "implementer": payload.get("implementer"),
                    "reviewer": reviewer,
                    "deferred_handoff": True,
                },
                run_id=new_run,
            )
        else:
            implementer = payload.get("implementer")
            if not isinstance(implementer, str) or not implementer.strip():
                return False
            new_status = _owner_kanban_transitions._landing_status_after_parents(
                conn, task_id
            )
            cur = conn.execute(
                "UPDATE tasks "
                "SET status = ?, assignee = COALESCE(?, assignee), "
                "    claim_lock = NULL, claim_expires = NULL, "
                "    worker_pid = NULL, worker_pid_started_at = NULL, "
                "    last_heartbeat_at = NULL, reclaim_reserved_at = NULL, "
                "    worker_registered_at = NULL, worker_scope = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND current_run_id = ?",
                (new_status, implementer, task_id, int(run_id)),
            )
            if cur.rowcount != 1:
                return False
            new_run = _kanban_db._end_run(
                conn,
                task_id,
                outcome="changes_requested",
                status=new_status,
                summary=payload.get("reason"),
            )
            _kanban_db._append_event(
                conn,
                task_id,
                "changes_requested",
                {
                    "reason": payload.get("reason"),
                    "implementer": implementer,
                    "reviewer": payload.get("reviewer"),
                    "status": new_status,
                    "deferred_handoff": True,
                },
                run_id=new_run,
            )
        _log.info(
            "kanban: applied deferred %s handoff for %s (run %s) after "
            "verified scope teardown",
            kind,
            task_id,
            run_id,
        )
        return True


def _own_worker_handoff_ceiling_ticks(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
) -> int:
    """Drain-ceiling ticks this run has already survived (pass 11, AO).

    Counts the ``claim_extended`` holds the handoff branch wrote with
    reason ``own_worker_handoff_draining`` at or beyond the drain
    ceiling (payload ``drain_age``). ``drain_age`` is
    ``now - marker_created_at`` and therefore monotonic, so once a tick
    reaches the ceiling every later tick does too: the plain count IS
    the run of consecutive ceiling ticks, with no reset to miss.
    """
    ticks = 0
    for ev in conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'claim_extended' AND run_id IS ?",
        (task_id, run_id),
    ).fetchall():
        try:
            payload = json.loads(ev["payload"]) if ev["payload"] else {}
        except (TypeError, ValueError):
            continue
        drain_age = payload.get("drain_age")
        if (
            payload.get("reason") == "own_worker_handoff_draining"
            and drain_age is not None
            and int(drain_age) >= _OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS
        ):
            ticks += 1
    return ticks


def _handle_stale_own_worker_handoff(
    conn: sqlite3.Connection,
    row: Any,
    *,
    now: int,
) -> bool:
    """Stale-sweep branch for a run with a pending own-worker handoff.

    ``release_stale_claims`` runs BEFORE ``detect_crashed_workers``, and
    a deferred handoff extends the claim only by one defer grace — a
    scope drain outlasting it used to fall into the generic stale
    reclaim here, ending the run as ``reclaimed`` and silently dropping
    the worker's requested review/changes transition (pass 9, AF).

    Contract per the deferred marker instead (pass 10, AJ):

    * scope verified dead — apply the handoff; the row lands in its
      requested lane with the payload intact and the generic reclaim
      never sees it. On a DEFINITE pid-semantics host (cgroup
      ``unsupported``), a confirmed-dead worker pid is the same proof;
    * a scope that is or may be alive — NEVER apply, whatever the
      marker's age: ``active`` is authoritative alive (a worker with a
      stale heartbeat is still alive), and applying beside it cleared
      the claim/scope and let the same tick spawn a duplicate from the
      newly spawnable row. Under the drain ceiling the claim is
      re-extended like any live-worker hold; AT the ceiling the
      teardown is ESCALATED instead — the queued verified stop keeps
      its SIGKILL escalation, one ``handoff_stop_escalated`` event per
      run records it, and the handoff applies on a later tick once the
      cgroup confirms empty. A drain that survives
      ``_OWN_WORKER_HANDOFF_DRAIN_BREAKER_TICKS`` consecutive ceiling
      ticks trips the breaker (pass 11, AO): the task blocks
      (``needs_input``) with the run ended and the scope retained on
      the row — nothing spawnable beside the wedged unit, the orphan
      audit keeps reaping it, and the operator unblocks after a
      verified death.

    Returns True when the row was handled (apply or extension); False
    when there is no pending marker for this run or the handoff could
    not be applied (row moved on, payload unusable) — the caller then
    falls through to the ordinary stale-claim handling.
    """
    run_id = row["current_run_id"]
    if run_id is None:
        return False
    marker = conn.execute(
        "SELECT payload, created_at FROM task_events "
        "WHERE task_id = ? AND kind = 'own_worker_handoff' "
        "  AND run_id IS ? ORDER BY id DESC LIMIT 1",
        (row["id"], int(run_id)),
    ).fetchone()
    if marker is None:
        return False
    try:
        payload = json.loads(marker["payload"])
    except (TypeError, ValueError):
        return True
    if not isinstance(payload, dict) or payload.get("scope") != row["worker_scope"]:
        return True
    # Reserve the exact attempt before a stop can affect its processes. A
    # concurrent heartbeat or successor invalidates this stale decision.
    reserved = _kanban_claims.reserve_reclaim(conn, row["id"], row, now=now)
    if reserved is None:
        return True
    row = reserved
    scope = row["worker_scope"]
    scope_dead = (not scope) or _kanban_worker_stop.request_worker_scope_stop(
        scope,
        task_id=row["id"],
    )
    drain_age = now - int(marker["created_at"])
    if not scope_dead:
        provably_empty = False  # Root PID death cannot prove a scope empty.
        if not provably_empty:
            # The scope is alive, or its emptiness cannot be proven
            # (``active`` / ``deactivating`` / ``unknown``, or an
            # ``unsupported`` host with a live or unattributable pid).
            # A handoff is NEVER applied in that state — the freed row
            # is spawnable and the same tick would duplicate the run
            # beside the live cgroup (pass 10, AJ).
            if drain_age >= _OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS:
                # Pass 11 (AO): escalation alone never terminates the
                # hold — a scope whose SIGKILL cannot drain it left the
                # row running forever with a per-tick warning. After N
                # consecutive ceiling ticks still not provably empty,
                # stop extending: the task blocks (needs_input) so
                # nothing can spawn beside the wedged scope, the run
                # ends, and the scope stays on the row for the operator
                # — the orphan audit keeps requesting the verified stop,
                # and a verified death later lets the operator unblock.
                ceiling_ticks = _own_worker_handoff_ceiling_ticks(
                    conn,
                    row["id"],
                    int(run_id),
                )
                if ceiling_ticks + 1 >= _OWN_WORKER_HANDOFF_DRAIN_BREAKER_TICKS:
                    worker_pid = (
                        int(row["worker_pid"])
                        if row["worker_pid"] is not None
                        else None
                    )
                    reason = (
                        "scope will not drain; manual cleanup needed "
                        f"(unit {scope}"
                        + (
                            f", worker pid {worker_pid}"
                            if worker_pid is not None
                            else ""
                        )
                        + ")"
                    )
                    with _kanban_db_connect.write_txn(conn):
                        cur = conn.execute(
                            "UPDATE tasks "
                            "   SET status = 'blocked', "
                            "       block_kind = 'needs_input', "
                            "       claim_lock = NULL, "
                            "       claim_expires = NULL, "
                            "       worker_pid = NULL, "
                            "       worker_pid_started_at = NULL, "
                            "       worker_registered_at = NULL, reclaim_reserved_at = NULL "
                            " WHERE id = ? AND status = 'running' "
                            "   AND claim_lock IS ? "
                            "   AND claim_expires IS ? AND current_run_id IS ? "
                            "   AND worker_scope IS ?",
                            (
                                row["id"],
                                row["claim_lock"],
                                row["claim_expires"],
                                run_id,
                                scope,
                            ),
                        )
                        if cur.rowcount == 1:
                            stuck_run = _kanban_db._end_run(
                                conn,
                                row["id"],
                                outcome="blocked",
                                status="blocked",
                                error=reason,
                                metadata={
                                    "scope": scope,
                                    "worker_pid": worker_pid,
                                    "drain_age": drain_age,
                                    "drain_ceiling": (
                                        _OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS
                                    ),
                                    "ceiling_ticks": ceiling_ticks + 1,
                                },
                            )
                            _kanban_db._append_event(
                                conn,
                                row["id"],
                                "handoff_scope_stuck",
                                {
                                    "reason": reason,
                                    "scope": scope,
                                    "worker_pid": worker_pid,
                                    "drain_age": drain_age,
                                    "drain_ceiling": (
                                        _OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS
                                    ),
                                    "ceiling_ticks": ceiling_ticks + 1,
                                },
                                run_id=stuck_run,
                            )
                            _log.warning(
                                "kanban: own-worker handoff for %s "
                                "(run %s) blocked after %d drain-"
                                "ceiling ticks — %s",
                                row["id"],
                                run_id,
                                ceiling_ticks + 1,
                                reason,
                            )
                    # Handled either way: the breaker fired, or its CAS
                    # missed because the row moved under us. This tick
                    # must not fall through to the extension or the
                    # generic reclaim.
                    return True
                # Ceiling reached with the scope still not provably
                # empty: the worker asked to hand off and has not
                # exited — it is wedged. Escalate the teardown rather
                # than the transition: the verified stop queued above
                # already escalates to SIGKILL on the service, the
                # claim and marker survive, and a later tick applies
                # the handoff once the cgroup confirms empty. One event
                # per run keeps the escalation auditable — and gates the
                # warning to once per run, not once per tick.
                with _kanban_db_connect.write_txn(conn):
                    already = conn.execute(
                        "SELECT 1 FROM task_events "
                        "WHERE task_id = ? "
                        "  AND kind = 'handoff_stop_escalated' "
                        "  AND run_id IS ? LIMIT 1",
                        (row["id"], int(run_id)),
                    ).fetchone()
                    if already is None:
                        _kanban_db._append_event(
                            conn,
                            row["id"],
                            "handoff_stop_escalated",
                            {
                                "scope": scope,
                                "drain_age": drain_age,
                                "drain_ceiling": (
                                    _OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS
                                ),
                            },
                            run_id=run_id,
                        )
                        _log.warning(
                            "kanban: deferred own-worker handoff for %s "
                            "(run %s) hit the %ds drain ceiling with the "
                            "scope not provably empty — escalating the "
                            "verified stop (SIGKILL) and holding the "
                            "claim; the handoff applies once the scope "
                            "confirms dead",
                            row["id"],
                            run_id,
                            _OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS,
                        )
            # Hold the claim (the marker must outlive its single defer
            # grace), exactly like a live worker would, and let a later
            # tick re-check.
            grace = now + _kanban_db.RECLAIM_DEFER_GRACE_SECONDS
            with _kanban_db_connect.write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET claim_expires = ? "
                    "WHERE id = ? AND status = 'running' "
                    "  AND claim_lock IS ? AND claim_expires IS ? "
                    "  AND current_run_id IS ? AND worker_scope IS ?",
                    (
                        grace,
                        row["id"],
                        row["claim_lock"],
                        row["claim_expires"],
                        run_id,
                        scope,
                    ),
                )
                if cur.rowcount != 1:
                    # CAS miss: the row moved between the stale scan and
                    # this write — the usual winner is a concurrent
                    # ``heartbeat_claim`` refreshing ``claim_expires``,
                    # i.e. the worker is ALIVE. Act on the fresh state
                    # read inside this transaction, never on the stale
                    # snapshot: falling through would hand the caller a
                    # termination keyed to *row*'s pre-refresh pid
                    # (pass 10, AK). Whatever the fresh state is (a
                    # landed heartbeat, or the row moved on entirely)
                    # the safe action this tick is to stand down — the
                    # next tick re-scans and sees the truth.
                    fresh = conn.execute(
                        "SELECT status, claim_lock, claim_expires "
                        "FROM tasks WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                    _log.info(
                        "kanban: own-worker handoff extension CAS missed "
                        "for %s (run %s) — fresh state status=%s "
                        "claim_lock=%s claim_expires=%s; skipping the "
                        "row this tick",
                        row["id"],
                        run_id,
                        fresh["status"] if fresh is not None else None,
                        fresh["claim_lock"] if fresh is not None else None,
                        (fresh["claim_expires"] if fresh is not None else None),
                    )
                    return True
                current_run = _kanban_db._current_run_id(conn, row["id"])
                if current_run is not None:
                    conn.execute(
                        "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                        (grace, current_run),
                    )
                _kanban_db._append_event(
                    conn,
                    row["id"],
                    "claim_extended",
                    {
                        "reason": "own_worker_handoff_draining",
                        "scope": scope,
                        "claim_expires_was": int(row["claim_expires"]),
                        "claim_expires_now": grace,
                        "drain_age": drain_age,
                    },
                    run_id=current_run,
                )
            return True
    return _apply_pending_own_worker_handoff(conn, row["id"], run_id)


def _handoff_caller_is_worker(
    row: Any,
    *,
    expected_run_id: Optional[int] = None,
) -> bool:
    """True when THIS process owns the run being handed off.

    A terminal handoff issued by the run's own worker must skip the
    pre-write teardown: stopping its scope or signalling its pid here
    would kill the caller before its own write commits (the detached
    post-commit stop in :func:`_stop_scope_after_worker_exit` exists for
    exactly that case). Ownership signals, strongest first:

    * scoped run: the worker's pinned ``HERMES_KANBAN_SCOPE`` env
      (inherited by every CLI child it shells out to) matches the row's
      scope — the caller is the worker or a descendant;
    * unscoped run: the row's registered pid IS this process with a
      matching start-time fingerprint;
    * a matching ``expected_run_id`` — the run-id ownership rule
      ``request_review`` has always honored: a caller that knows the
      live run id is the worker or its delegate.
    """
    keys = row.keys() if hasattr(row, "keys") else ()
    scope = row["worker_scope"] if "worker_scope" in keys else None
    if scope and os.environ.get("HERMES_KANBAN_SCOPE", "").strip() == scope:
        return True
    pid = row["worker_pid"] if "worker_pid" in keys else None
    if pid is not None and int(pid) == os.getpid():
        started = (
            row["worker_pid_started_at"] if "worker_pid_started_at" in keys else None
        )
        mine = _kanban_worker_identity._worker_pid_start_time(os.getpid())
        if started is None or (mine is not None and int(started) == int(mine)):
            return True
    if expected_run_id is not None:
        current = row["current_run_id"] if "current_run_id" in keys else None
        if current is not None and int(current) == int(expected_run_id):
            return True
    return False


def _stop_scope_after_worker_exit(unit_name: Optional[str]) -> None:
    """Best-effort DETACHED scope stop for worker-side terminal paths.

    ``complete_task`` / ``block_task`` run INSIDE the worker process — a
    synchronous ``systemctl stop`` here would SIGTERM the caller before
    it returns from its own ``kanban_complete`` call. Everything durable
    (run outcome, events, hooks) is already written when this fires, so
    the stop is launched detached and never waited on: the worker gets
    its terminal signal moments after finishing, and any descendant the
    worker leaves behind dies with the scope. The dispatcher's
    ``reap_orphaned_worker_scopes`` audit sweep guarantees the verified
    kill even if this fire-and-forget attempt is lost.
    """
    if not unit_name:
        return
    try:
        import shutil

        binary = shutil.which("systemctl")
        if binary is None:
            return
        subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            [binary, "--user", "stop", unit_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        _log.debug(
            "kanban: detached scope stop for %s failed: %s",
            unit_name,
            exc,
        )


from hermes_cli import kanban_db as _kanban_db
from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_stop as _kanban_worker_stop

from hermes_cli import kanban_claims as _kanban_claims

_OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS = _kanban_db.RECLAIM_DEFER_GRACE_SECONDS * 5
_OWN_WORKER_HANDOFF_DRAIN_BREAKER_TICKS = 3
