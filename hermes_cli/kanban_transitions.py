"""Kanban transitions."""
from __future__ import annotations

import json
import sqlite3
import logging
import time
from typing import Any, Optional


_log = logging.getLogger(__name__)

def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the newest ``blocked``/``unblocked`` event is ``blocked`` — an
    explicit ``kanban_block`` that must wait for an operator. A breaker trip
    emits ``gave_up`` (not ``blocked``) and so auto-recovers, as does a task
    with no such event at all (direct DB edit).

    See #28712.
    Returns ``False`` when there is no such event at all (e.g. the task was set to ``status='blocked'`` by
    the circuit breaker or by direct DB manipulation) — preserves the pre-#28712 auto-recover semantics for
    that path.
    """
    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return bool(row) and row["kind"] == "blocked"


def _resume_status_from_events(conn: sqlite3.Connection, task_id: str) -> str:
    """``review`` when the newest lifecycle event carries a review
    ``resume_status``/``retry_status``/``source_status``, else ``ready`` (legacy)."""
    from hermes_cli.kanban_db import _json_dict, _row_get

    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind IN ("
        "'blocked', 'block_loop_detected', 'dependency_wait', 'gave_up', "
        "'unblocked', 'changes_requested', 'review_reopened', 'status', 'reclaimed', "
        "'stale', 'timed_out', 'crashed', 'spawn_failed', 'rate_limited'"
        ") ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    payload = _json_dict(_row_get(row, "payload"))
    for key in ("resume_status", "retry_status", "source_status"):
        if payload.get(key) == "review":
            return "review"
    return "ready"


def recompute_ready(conn: sqlite3.Connection, failure_limit: int = None) -> int:
    """Promote ``todo``/``blocked`` tasks whose parents are all done/archived;
    returns the count. Opens its own IMMEDIATE txn — call OUTSIDE any write txn.

    ``blocked`` is skipped when sticky (explicit ``kanban_block``) or when
    ``consecutive_failures`` reached the limit (else the breaker could never
    trip). Limit order matches ``_record_task_failure``: ``max_retries`` >
    ``failure_limit`` > ``DEFAULT_FAILURE_LIMIT``.

    ``blocked`` tasks are also considered for promotion (so a task
    blocked purely by a parent dependency unblocks itself when the
    parent completes), *except* in two cases:

    1. The most recent block event was a worker-initiated
       ``kanban_block`` — those stay blocked until an explicit
       ``kanban_unblock`` (#28712).

    2. The task's ``consecutive_failures`` has reached the effective
       failure limit.  This prevents infinite retry loops when a task
       repeatedly exhausts its iteration budget: without this guard the
       counter would reset on every recovery cycle and the circuit
       breaker could never trip (#35072).

    3. The task still carries a ``worker_scope`` (pass 12, AQ — e.g. a
       drain-ceiling breaker row).  Promoting it would make it spawnable
       beside a cgroup that may still be live, duplicating the run.  It
       stays where it is until the scope is verified dead and cleared by
       the pre-spawn scope sweep / verified-stop service.

    The effective failure limit resolves in the same order as the
    circuit breaker in ``_record_task_failure`` so the two never
    disagree about when a task is permanently blocked:

      1. per-task ``max_retries`` if set
      2. caller-supplied ``failure_limit`` (the dispatcher passes the
         ``kanban.failure_limit`` config value through ``dispatch_once``)
      3. ``DEFAULT_FAILURE_LIMIT``
    """
    from hermes_cli.kanban_db import _append_event
    from hermes_cli.kanban_db_connect import write_txn
    from hermes_cli.kanban_db_dispatch import DEFAULT_FAILURE_LIMIT

    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    promoted = 0
    with write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries, "
            "worker_scope "
            "FROM tasks WHERE status IN ('todo', 'blocked')"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            cur_status = row["status"]
            if row["worker_scope"]:
                # Pass 12 (AQ): a non-running row that still carries a
                # worker scope is NEVER auto-promoted — doing so makes
                # it spawnable beside a cgroup that may still be live
                # (the drain-ceiling breaker's blocked rows keep their
                # scope for the operator). It stays put until the scope
                # is verified dead and cleared by the pre-spawn scope
                # sweep / verified-stop service.
                continue
            if cur_status == "blocked" and _has_sticky_block(conn, task_id):
                # Explicit human-intervention block; only ``unblock_task`` may exit it.
                continue
            parents = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?", (task_id,),
            ).fetchall()
            if all(p["status"] in ("done", "archived") for p in parents):
                resume_status = _resume_status_from_events(conn, task_id)
                if cur_status == "blocked":
                    # At the breaker limit, no auto-recovery (else block ->
                    # recover -> respawn -> exhaust -> block forever). The
                    # counter is preserved so it accumulates across cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = ? "
                        "WHERE id = ? AND status = 'blocked'", (resume_status, task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'",
                        (resume_status, task_id),
                    )
                _append_event(
                    conn, task_id, "promoted",
                    {"status": resume_status} if resume_status != "ready" else None,
                )
                promoted += 1
    return promoted


def _parents_satisfied(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return whether every direct parent is terminal for dependency gating."""
    return conn.execute(
        # Check if this task has children that still need the workspace. If any child is not yet
        # done/archived, defer cleanup so the child can read handoff artifacts from the workspace (#33774).
        "SELECT 1 FROM task_links l "
        "JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? "
        "AND p.status NOT IN ('done', 'archived') LIMIT 1", (task_id,),
    ).fetchone() is None





def _route_block(
    kind: Optional[str], reason: Optional[str], source_status: str, *,
    prev_kind: Optional[str], prev_recurrences: int,
) -> tuple[str, str, str, tuple, dict]:
    """``(new_status, event_kind, set_sql, params, payload)`` for :func:`block_task`.

    ``dependency`` never enters the human ``blocked`` bucket: it waits in
    ``todo`` for ``recompute_ready``, so a cron never sees a dependency-wait
    as something to "unblock". Every other kind counts unblock-loop
    recurrences: block_task only fires from running/ready (AFTER an unblock
    returned the task to the pool), so a stored ``block_kind`` equal to the
    incoming one means blocked -> unblocked -> re-block for the same cause
    (un-typed None compares equal to a prior un-typed block). At
    ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
    """
    from hermes_cli.kanban_db import BLOCK_RECURRENCE_LIMIT

    payload = {"reason": reason, "kind": kind, "source_status": source_status}
    if kind == "dependency":
        return "todo", "dependency_wait", "block_kind    = ?", (kind,), payload
    recurrences = prev_recurrences + 1 if prev_kind == kind else 1
    set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
    payload = {"reason": reason, "kind": kind, "recurrences": recurrences, "source_status": source_status}
    if recurrences >= BLOCK_RECURRENCE_LIMIT:
        payload["limit"] = BLOCK_RECURRENCE_LIMIT
        return "triage", "block_loop_detected", set_sql, (kind, recurrences), payload
    return "blocked", "blocked", set_sql, (kind, recurrences), payload


def redact_review_value(value: Any) -> Any:
    """Redact secrets at the domain boundary for durable review handoffs."""
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {key: redact_review_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_review_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_review_value(item) for item in value)
    return value


def request_review(
    conn: sqlite3.Connection, task_id: str, *, summary: Optional[str] = None,
    metadata: Optional[dict] = None, reviewer: Optional[str] = None,
    expected_run_id: Optional[int] = None, force: bool = False, with_reason: bool = False,
):
    """``running``/``ready`` -> ``review``; never touches block recurrence accounting.

    Implementer and reviewer are recorded on the event so requested changes
    route back to the right profile; ``reviewer`` reassigns the task, and on
    re-review defaults to the latest ``changes_requested`` provenance. A live
    claim is only cleared with proof of ownership (``expected_run_id``) or
    ``force=True``. Returns ``bool``, or ``(ok, reason)`` with ``with_reason``.
    """
    from hermes_cli.kanban_db import _append_event, _end_or_synthesize_run, _first_line
    from hermes_cli.kanban_db_connect import write_txn
    from hermes_cli.kanban_worker_handoff import _defer_own_worker_handoff, _handoff_caller_is_worker, _mark_run_scope_stopping, _stop_scope_after_worker_exit
    from hermes_cli.kanban_worker_identity import _defer_reclaim_for_live_worker, _terminate_reclaimed_worker, _worker_survived_termination, _worker_termination_tuple
    from hermes_cli.kanban_worker_scope import _kanban_scope_state
    from hermes_cli.kanban_worker_stop import request_worker_scope_stop


    def _ret(ok: bool, reason: Optional[str] = None):
        return (ok, reason) if with_reason else ok

    summary = redact_review_value(summary)
    metadata = redact_review_value(metadata)
    # Phase 0 — verified teardown BEFORE the spawnable write (dashboard
    # contract, 7c8e884f72): a third party yanking a live running row to
    # the spawnable ``review`` lane must confirm the worker is dead
    # first, or defer with the stopping marker. The run's OWN worker
    # (scope env / pid / run-id ownership) skips this — stopping it here
    # would kill the caller before its write commits; its own exit plus
    # the detached post-commit stop retire the scope.
    snapshot = conn.execute(
        "SELECT status, claim_lock, current_run_id, worker_pid, "
        "worker_pid_started_at, worker_scope, assignee "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    caller_is_worker = (
        _handoff_caller_is_worker(snapshot, expected_run_id=expected_run_id)
        if snapshot is not None
        else False
    )
    if (
        snapshot is not None
        and snapshot["status"] == "running"
        and snapshot["claim_lock"] is not None
        and (force or expected_run_id is not None)
        and (
            expected_run_id is None
            or int(snapshot["current_run_id"] or 0) == int(expected_run_id)
        )
        and not caller_is_worker
        and _parents_satisfied(conn, task_id)
    ):
        _, reviewer_error = _resolve_request_review_reviewer(conn, task_id, reviewer)
        if reviewer_error is not None:
            return _ret(False, reviewer_error)
        pid, claim_lock, pid_started, scope = _worker_termination_tuple(
            snapshot,
        )
        termination = _terminate_reclaimed_worker(
            pid, claim_lock,
            scope_unit=scope or None,
            pid_started_at=pid_started,
            task_id=task_id, run_id=snapshot["current_run_id"],
        )
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, task_id, snapshot["claim_lock"], int(time.time()),
                termination,
                reason="request_review_stop_unconfirmed",
            )
            if scope:
                _mark_run_scope_stopping(
                    conn, task_id, scope,
                    reason="request_review_stop_unconfirmed",
                    expected_run_id=snapshot["current_run_id"],
                )
            return _ret(
                False,
                "task stays running: its worker could not be verified "
                "stopped (scope still draining); retry once the teardown "
                "completes",
            )
    if (
        snapshot is not None
        and caller_is_worker
        and snapshot["status"] == "running"
        and snapshot["worker_scope"]
        and snapshot["current_run_id"] is not None
        and (
            force
            or (
                expected_run_id is not None
                and int(snapshot["current_run_id"]) == int(expected_run_id)
            )
        )
        and _parents_satisfied(conn, task_id)
        and _kanban_scope_state(snapshot["worker_scope"]) != "dead"
    ):
        # Pass 8 (V): the own worker's scope necessarily still holds the
        # caller, so a pre-write verified stop is impossible by
        # construction — and writing the spawnable ``review`` row before
        # the scope drains is exactly the shape the third-party contract
        # forbids. Defer the whole transition: the marker carries the
        # handoff, the verified-stop sweep applies it once the cgroup is
        # empty, and the worker may simply exit after a True return.
        reviewer, reviewer_error = _resolve_request_review_reviewer(
            conn, task_id, reviewer,
        )
        if reviewer_error is not None:
            return _ret(False, reviewer_error)
        deferred = _defer_own_worker_handoff(
            conn, task_id, snapshot["worker_scope"],
            {
                "handoff": "review_requested",
                "implementer": snapshot["assignee"],
                "reviewer": reviewer,
                "summary": summary,
                "metadata": metadata,
            },
            claim_lock=snapshot["claim_lock"],
            expected_run_id=snapshot["current_run_id"],
        )
        if not deferred:
            return _ret(False, "task changed during review handoff")
        request_worker_scope_stop(
            snapshot["worker_scope"], task_id=task_id, conn=conn,
        )
        return _ret(
            True,
            "review handoff deferred: the row flips to review once this "
            "worker's scope drains (the verified-stop sweep applies it)",
        )
    with write_txn(conn):
        if not _parents_satisfied(conn, task_id):
            return _ret(False, "parent dependencies are not satisfied")
        trow = conn.execute(
            "SELECT assignee, status, claim_lock, current_run_id, "
            "worker_scope, worker_pid, worker_pid_started_at "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if trow is None:
            return _ret(False, "task not found")
        if (
            snapshot is not None
            and snapshot["status"] == "running"
            and snapshot["claim_lock"] is not None
            and (
                int(snapshot["current_run_id"] or 0)
                != int(trow["current_run_id"] or 0)
                or snapshot["worker_pid"] != trow["worker_pid"]
                or snapshot["worker_pid_started_at"]
                != trow["worker_pid_started_at"]
            )
        ):
            # Pass 8 (V): a force handoff verified/stopped the SNAPSHOT's
            # worker; if a newer run took the row before this write, the
            # spawnable flip must not clear ITS claim. No-op — the caller
            # retries against the new run.
            return _ret(
                False,
                "task changed hands during the handoff (a newer run "
                "started); retry",
            )
        # Refuse to clear a live worker's claim without proof of ownership
        # (expected_run_id) or an explicit human override (force=True).
        if (
            expected_run_id is None
            and not force
            and trow["status"] == "running"
            and trow["claim_lock"] is not None
        ):
            return _ret(
                False, "task is running under a live claim; pass expected_run_id "
                "(worker ownership) or force=True (explicit operator "
                "override) instead of clearing the live run's claim",
            )
        implementer = trow["assignee"]
        reviewer, reviewer_error = _resolve_request_review_reviewer(
            conn, task_id, reviewer,
        )
        if reviewer_error is not None:
            return _ret(False, reviewer_error)
        assignee_sql = ", assignee = ?" if reviewer is not None else ""
        run_guard = "" if expected_run_id is None else " AND current_run_id = ?"
        params: tuple[Any, ...] = (
            *(() if reviewer is None else (reviewer,)), task_id,
            *(() if expected_run_id is None else (int(expected_run_id),)),
        )
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'review',
                   claim_lock    = NULL,
                   claim_expires = NULL,
                   worker_pid    = NULL,
                   worker_pid_started_at = NULL,
                   worker_registered_at = NULL,
                   worker_scope = NULL
            """ + assignee_sql + """
             WHERE id = ?
               AND status IN ('running', 'ready')
            """ + run_guard,
            params,
        )
        if cur.rowcount != 1:
            return _ret(
                False, "task is not in running/ready (or expected_run_id did not match the current run)",
            )
        run_id = _end_or_synthesize_run(
            conn, task_id, outcome="review_requested", status="review",
            summary=summary, metadata=metadata, synthesize=bool(summary or metadata),
        )
        _append_event(
            conn,
            task_id,
            "review_requested",
            {
                "summary": _first_line(summary, 400) or None,
                "implementer": implementer,
                "reviewer": reviewer,
            },
            run_id=run_id,
        )
    # The implementer's attempt is over (review lane takes over). A
    # third-party handoff verified the teardown in Phase 0 before the
    # write; only the run's OWN worker still needs the fire-and-forget
    # scope stop — see _stop_scope_after_worker_exit.
    if caller_is_worker:
        _stop_scope_after_worker_exit(
            trow["worker_scope"] if trow is not None else None
        )
    return _ret(True)


def _prior_reviewer(conn: sqlite3.Connection, task_id: str):
    """Reviewer recorded by the latest ``changes_requested`` run's event.
    ``None`` = first review (no such run); ``False`` = a run exists but its
    provenance is missing/malformed."""
    from hermes_cli.kanban_db import _json_dict, _latest_event, _row_get

    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if changes_run is None:
        return None
    changes_event = _latest_event(conn, task_id, "changes_requested", changes_run["id"])
    reviewer = _json_dict(_row_get(changes_event, "payload")).get("reviewer")
    return reviewer if isinstance(reviewer, str) and reviewer.strip() else False


def _nonblank_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def request_changes(
    conn: sqlite3.Connection, task_id: str, *, reason: str, expected_run_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Close an active reviewer run (claimed from ``review``) and hand the task
    back to the implementer from the latest ``review_requested`` event, parent
    gating reapplied. Returns ``(ok, implementer | reason)``."""
    from hermes_cli.kanban_db import _append_event, _canonical_assignee, _end_run
    from hermes_cli.kanban_db_connect import write_txn
    from hermes_cli.kanban_worker_handoff import _defer_own_worker_handoff, _handoff_caller_is_worker, _mark_run_scope_stopping, _stop_scope_after_worker_exit
    from hermes_cli.kanban_worker_identity import _defer_reclaim_for_live_worker, _terminate_reclaimed_worker, _worker_survived_termination, _worker_termination_tuple
    from hermes_cli.kanban_worker_scope import _kanban_scope_state
    from hermes_cli.kanban_worker_stop import request_worker_scope_stop

    reason = str(redact_review_value(reason or "")).strip()
    if not reason:
        return False, "reason is required"

    # Phase 0 — verified teardown BEFORE the spawnable write (dashboard
    # contract, 7c8e884f72): routing the task back to its implementer is
    # a spawnable write, so a third party closing a live reviewer run
    # must confirm that worker dead first, or defer with the stopping
    # marker. The reviewer's OWN handoff (scope env / pid / run-id
    # ownership) skips this — stopping it here would kill the caller
    # before its write commits; its own exit plus the detached
    # post-commit stop retire the scope. The claimed-from-review
    # provenance is pre-checked so a request the transaction would
    # refuse never tears a live worker down for nothing.
    snapshot = conn.execute(
        "SELECT status, claim_lock, current_run_id, worker_pid, "
        "worker_pid_started_at, worker_scope, assignee "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    caller_is_worker = (
        _handoff_caller_is_worker(snapshot, expected_run_id=expected_run_id)
        if snapshot is not None
        else False
    )
    _reviewer_provenance = None
    if (
        snapshot is not None
        and snapshot["status"] == "running"
        and snapshot["claim_lock"] is not None
        and (
            expected_run_id is None
            or int(snapshot["current_run_id"] or 0) == int(expected_run_id)
        )
        and not caller_is_worker
        and snapshot["current_run_id"] is not None
    ):
        _, provenance_error = _review_handoff_provenance(
            conn, task_id, snapshot["current_run_id"],
        )
        if provenance_error is not None:
            return False, provenance_error
        _reviewer_provenance = "review"
    if _reviewer_provenance == "review":
        pid, claim_lock, pid_started, scope = _worker_termination_tuple(
            snapshot,
        )
        termination = _terminate_reclaimed_worker(
            pid, claim_lock,
            scope_unit=scope or None,
            pid_started_at=pid_started,
            task_id=task_id, run_id=snapshot["current_run_id"],
        )
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, task_id, snapshot["claim_lock"], int(time.time()),
                termination,
                reason="request_changes_stop_unconfirmed",
            )
            if scope:
                _mark_run_scope_stopping(
                    conn, task_id, scope,
                    reason="request_changes_stop_unconfirmed",
                    expected_run_id=snapshot["current_run_id"],
                )
            return (
                False,
                "task stays running: its reviewer worker could not be "
                "verified stopped (scope still draining); retry once the "
                "teardown completes",
            )
    if (
        snapshot is not None
        and caller_is_worker
        and snapshot["status"] == "running"
        and snapshot["worker_scope"]
        and snapshot["current_run_id"] is not None
        and expected_run_id is not None
        and int(snapshot["current_run_id"]) == int(expected_run_id)
        and _kanban_scope_state(snapshot["worker_scope"]) != "dead"
    ):
        # Pass 8 (V): the reviewer's own scope still holds the caller,
        # so the spawnable rework write must wait for the verified stop.
        # Defer the whole transition (the marker carries it; the
        # verified-stop sweep applies it once the cgroup is empty) and
        # fail fast on any precondition the transaction would raise — a
        # deferral must never accept a request that would be refused.
        implementer, provenance_error = _review_handoff_provenance(
            conn, task_id, snapshot["current_run_id"],
        )
        if provenance_error is not None:
            return False, provenance_error
        reviewer = snapshot["assignee"]
        if isinstance(reviewer, str) and reviewer.strip():
            reviewer = _canonical_assignee(reviewer)
        else:
            reviewer = None
        deferred = _defer_own_worker_handoff(
            conn, task_id, snapshot["worker_scope"],
            {
                "handoff": "changes_requested",
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
            },
            claim_lock=snapshot["claim_lock"],
            expected_run_id=snapshot["current_run_id"],
        )
        if not deferred:
            return False, "task changed during review handoff"
        request_worker_scope_stop(
            snapshot["worker_scope"], task_id=task_id, conn=conn,
        )
        return True, implementer

    with write_txn(conn):
        task_row = conn.execute(
            "SELECT status, assignee, current_run_id, worker_scope, "
            "worker_pid, worker_pid_started_at "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task_row is None:
            return False, "task not found"
        if (
            snapshot is not None
            and snapshot["status"] == "running"
            and snapshot["claim_lock"] is not None
            and (
                int(snapshot["current_run_id"] or 0)
                != int(task_row["current_run_id"] or 0)
                or snapshot["worker_pid"] != task_row["worker_pid"]
                or snapshot["worker_pid_started_at"]
                != task_row["worker_pid_started_at"]
            )
        ):
            # Pass 8 (V): the Phase 0 teardown verified the SNAPSHOT's
            # reviewer; a newer run that took the row before this write
            # must not have its claim cleared by the rework flip.
            return False, (
                "task changed hands during the handoff (a newer run "
                "started); retry"
            )
        current_run_id = task_row["current_run_id"]
        if task_row["status"] != "running" or current_run_id is None:
            return False, "task is not in an active review run"
        if expected_run_id is not None and int(current_run_id) != int(expected_run_id):
            return False, "run_id mismatch"

        implementer, provenance_error = _review_handoff_provenance(
            conn, task_id, current_run_id,
        )
        if provenance_error is not None:
            return False, provenance_error
        reviewer = task_row["assignee"]
        if isinstance(reviewer, str) and reviewer.strip():
            reviewer = _canonical_assignee(reviewer)
        else:
            reviewer = None

        new_status = _landing_status_after_parents(conn, task_id)
        # consecutive_failures deliberately PRESERVED: a review transition is
        # not evidence the pathology cleared; only complete_task resets it.
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL,
                   worker_pid_started_at = NULL,
                   worker_registered_at = NULL,
                   worker_scope = NULL
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, task_id, int(current_run_id)),
        )
        if cur.rowcount != 1:
            return False, "task changed during review handoff"
        run_id = _end_run(
            conn, task_id, outcome="changes_requested", status=new_status, summary=reason,
        )
        _append_event(
            conn,
            task_id,
            "changes_requested",
            {
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
                "status": new_status,
            },
            run_id=run_id,
        )
    # The reviewer's attempt is over (work returns to the implementer).
    # A third-party handoff verified the teardown in Phase 0 before the
    # write; only the reviewer's OWN run still needs the fire-and-forget
    # scope stop — see _stop_scope_after_worker_exit.
    if caller_is_worker:
        _stop_scope_after_worker_exit(
            task_row["worker_scope"] if task_row is not None else None
        )
    return True, implementer


def _parked_worker_snapshot(conn: sqlite3.Connection, task_id: str):
    return conn.execute(
        "SELECT status, current_run_id, claim_lock, worker_pid, "
        "worker_pid_started_at, worker_scope FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()


def _parked_scope_is_dead(snapshot) -> bool:
    from hermes_cli.kanban_worker_scope import _kanban_scope_state

    return snapshot is not None and (
        not snapshot["worker_scope"]
        or _kanban_scope_state(snapshot["worker_scope"]) == "dead"
    )


def _clear_verified_parked_scope(conn, task_id, snapshot) -> bool:
    """Consume a scope verdict only for its unchanged row under the write lock."""
    current = _parked_worker_snapshot(conn, task_id)
    if current is None or snapshot is None or tuple(current) != tuple(snapshot):
        return False
    if current["worker_scope"]:
        conn.execute("UPDATE tasks SET worker_scope = NULL WHERE id = ?", (task_id,))
    return True


def promote_task(
    conn: sqlite3.Connection, task_id: str, *, actor: str, reason: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Operator promotion ``todo``/``blocked`` -> ``ready`` with an audit event.
    Refused while a parent is unfinished; ``dry_run`` only validates.
    Returns ``(ok, reason)``."""
    from hermes_cli.kanban_db import _append_event, _task_status
    from hermes_cli.kanban_db_connect import write_txn

    cur_status = _task_status(conn, task_id)
    if cur_status is None:
        return False, f"task {task_id} not found"

    if cur_status not in ("todo", "blocked"):
        return False, (
            f"task {task_id} is {cur_status!r}; promote only applies to "
            f"'todo' or 'blocked'"
        )

    # No override: claim_task demotes ready -> todo on an undone parent whichever
    # writer set 'ready', so a forced promotion would only report a success the
    # first claim silently reverts (#106195). The dependency itself is the knob.
    parents = conn.execute(
        "SELECT t.id, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ?", (task_id,),
    ).fetchall()
    unsatisfied = [p["id"] for p in parents if p["status"] not in ("done", "archived")]
    if unsatisfied:
        return False, (
            f"unsatisfied parent dependencies: {', '.join(unsatisfied)} "
            f"(the ready -> running claim re-checks parents, so promotion cannot "
            f"bypass them; complete the parents or drop the link with "
            f"`hermes kanban unlink <parent_id> {task_id}`)"
        )

    snapshot = _parked_worker_snapshot(conn, task_id)
    if not _parked_scope_is_dead(snapshot):
        return False, "worker scope is still alive or unverified"
    if dry_run:
        return True, None

    with write_txn(conn):
        if not _parents_satisfied(conn, task_id):
            return False, "parent dependencies changed during promotion"
        if not _clear_verified_parked_scope(conn, task_id, snapshot):
            return False, "task changed during the worker scope probe"
        upd = conn.execute(
            "UPDATE tasks SET status = 'ready' "
            "WHERE id = ? AND status IN ('todo', 'blocked')", (task_id,),
        )
        if upd.rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _append_event(conn, task_id, "promoted_manual", {"actor": actor, "reason": reason})

    return True, None


def _reclaim_dangling_run(
    conn: sqlite3.Connection, task_id: str, *, statuses, now: int, note: str,
) -> None:
    """Close a leaked open run before a status flip so the invariant
    ``current_run_id IS NULL <=> run row terminal`` holds; no-op normally."""
    placeholders = ", ".join("?" for _ in statuses)
    stale = conn.execute(
        f"SELECT current_run_id FROM tasks WHERE id = ? AND status IN ({placeholders})",
        (task_id, *statuses),
    ).fetchone()
    if stale and stale["current_run_id"]:
        conn.execute(
            """
            UPDATE task_runs
               SET status = 'reclaimed', outcome = 'reclaimed',
                   summary = COALESCE(summary, ?),
                   ended_at = ?,
                   claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
             WHERE id = ? AND ended_at IS NULL
            """,
            (note, now, int(stale["current_run_id"])),
        )


def _landing_status_after_parents(conn: sqlite3.Connection, task_id: str) -> str:
    """``ready`` if every parent is terminal else ``todo`` — the re-gate shared by
    unblock/reopen so neither can spawn a child whose upstream is unfinished."""
    return "ready" if _parents_satisfied(conn, task_id) else "todo"


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``blocked``/``scheduled`` to its safe resumable phase.

    Defensively closes any stale ``current_run_id`` pointer before flipping
    status. In the common path (``block_task`` closed the run already) this
    is a no-op. If a future or external write left the pointer dangling,
    the leaked run is closed as ``reclaimed`` inside the same txn so the
    runs invariant (``current_run_id IS NULL`` ⇔ run row in terminal
    state) holds for the rest of this function's lifetime.

    Pass 12 (AQ): a blocked breaker row can still carry its
    ``worker_scope`` (the drain-ceiling breaker keeps it for the
    operator). Unblocking such a row would make it spawnable beside a
    cgroup that may still be live, so the unblock is refused unless the
    scope is verified dead at that moment — in which case the stale
    pointer is cleared and the unblock proceeds.
    """
    from hermes_cli.kanban_db import _append_event, _task_status
    from hermes_cli.kanban_db_connect import write_txn

    now = int(time.time())
    snapshot = _parked_worker_snapshot(conn, task_id)
    if snapshot is None or snapshot["status"] not in ("blocked", "scheduled"):
        return False
    if not _parked_scope_is_dead(snapshot):
        return False
    with write_txn(conn):
        if not _clear_verified_parked_scope(conn, task_id, snapshot):
            return False
        resume_status = (
            _resume_status_from_events(conn, task_id)
            if _task_status(conn, task_id) == "blocked"
            else "ready"
        )
        _reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        # Re-gate on parent completion before restoring the source phase.
        landing_status = _landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        # ``block_kind``/``block_recurrences`` deliberately survive the unblock:
        # resetting them is the amnesia that let cron-unblock <-> re-block loop
        # unbounded; only complete_task clears them. ``consecutive_failures``
        # (the dispatcher's spawn/crash counter) IS reset — a deliberate unblock
        # is a fresh start for the retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "unblocked",
            (
                {"status": new_status, "resume_status": resume_status}
                if new_status != "ready" or resume_status != "ready"
                else None
            ),
        )
        return True


def reopen_review_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """``review`` -> ``ready``/``todo`` so the implementer re-runs on the new
    comments; restores the implementer from the ``review_requested`` event.
    Preserves ``consecutive_failures`` and the block loop counter (review is
    not a block; only :func:`complete_task` clears them)."""
    from hermes_cli.kanban_db import _append_event, _json_dict, _latest_event, _row_get
    from hermes_cli.kanban_db_connect import write_txn

    now = int(time.time())
    with write_txn(conn):
        _reclaim_dangling_run(
            conn, task_id, statuses=("review",), now=now,
            note="invariant recovery on review reopen",
        )
        new_status = _landing_status_after_parents(conn, task_id)
        review_event = _latest_event(conn, task_id, "review_requested")
        handoff = _json_dict(_row_get(review_event, "payload"))
        implementer = _nonblank_str(handoff.get("implementer"))
        params: tuple[Any, ...] = (new_status, *((implementer,) if implementer else ()), task_id)
        cur = conn.execute(
            # consecutive_failures deliberately PRESERVED: review reopen is not
            # a success signal; only complete_task resets the breaker (#35072).
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            + (", assignee = ?" if implementer else "")
            + " WHERE id = ? AND status = 'review'",
            params,
        )
        if cur.rowcount != 1:
            return False
        payload: dict[str, Any] = {"status": new_status}
        if implementer:
            payload["implementer"] = implementer
        _append_event(
            conn, task_id, "review_reopened", payload if payload != {"status": "ready"} else None,
        )
        return True


def invalidate_descendants_for_parent_reopen(
    conn: sqlite3.Connection, task_id: str, *, author: str,
) -> dict[str, Any]:
    """THE done-reopen invalidation: every ``ready``/``review``/``running``/``done``
    descendant of a reopened ancestor is demoted to ``todo`` and re-gated.
    Every surface that reopens a done task (dashboard PATCH/drag) routes here.

    Composes under the caller's txn (``allow_nested=True``) so the flip and the
    retractions commit atomically. Each descendant gets a
    ``descendant_invalidated`` event, the legacy ``status`` event the live feed
    renders, and a comment naming the ancestor. Running descendants are closed
    ``reclaimed`` and their workers killed strictly post-commit (audit trail
    before death) — when composed, the CALLER must drain ``terminations``
    after its own commit. ``consecutive_failures`` resets (deliberate operator
    action), the opposite of :func:`reopen_review_task`.

    Transactionality: composes under the caller's already-open transaction
    via ``write_txn(conn, allow_nested=True)`` — the dashboard's status
    writer must commit the ancestor's status flip and the descendant
    retractions atomically (a crash between the two would leave stale done
    descendants claiming a premise that no longer holds). Called standalone
    it opens its own transaction. All SQL is inline per this file's txn
    conventions (no calls into other txn-opening helpers).

    Non-silent contract: every invalidated descendant gets
    * a ``descendant_invalidated`` event with ``{ancestor, prior_status,
      new_status}`` (plus ``resume_status``) for board/notifier surfaces,
    * the legacy ``status`` event (``reason=ancestor_reopened``) the live
      feed already renders, and
    * a ``task_comments`` row naming the reopened ancestor, so operators see
      WHY a card moved instead of watching it silently teleport.

    Live ``running`` descendants keep the termination behavior (a running
    child building on a retracted premise is wasted spend), split by
    isolation so no descendant is ever demoted beside a live worker
    (Gate B review, finding E):

    * Scoped descendants are verified stopped BEFORE the demotion: a
      confirmed-empty cgroup demotes immediately; an unconfirmed stop
      DEFERS that descendant — it stays ``running`` with its claim held
      and a ``scope_stopping`` marker, and the crash-cleanup path
      requeues it once the verified-stop service finishes the kill.
    * Unscoped descendants keep the audit-first contract: events and
      comments are written inside the transaction and the kill happens
      strictly post-commit via :func:`_terminate_reclaimed_worker`.
      When this function opened its own transaction it performs the
      terminations itself after commit; when composing under a caller's
      transaction the caller MUST drain the returned ``terminations``
      list with ``_terminate_reclaimed_worker`` after its own commit.

    ``consecutive_failures`` is reset to 0 on every invalidated descendant:
    ancestor reopen is a deliberate operator action, so demoted work gets a
    fresh start with the breaker (a previously auto-blocked-then-completed
    descendant should not re-enter the queue one failure from the breaker).
    This is deliberately the OPPOSITE of the review-transition rule
    (:func:`reopen_review_task` / #35072 preserves the counter) because the
    autonomous review loop must not be able to launder its own failure
    streak, while an operator invalidating a subtree is an explicit reset
    signal.

    Returns ``{"invalidated": [...], "terminations": [...]}`` where each
    invalidated entry is ``{id, prior_status, new_status, resume_status}``
    and each termination is a ``(worker_pid, claim_lock,
    worker_pid_started_at, worker_scope)`` tuple — pass all four to
    ``_terminate_reclaimed_worker`` so unscoped descendants die with the
    retraction. (Scoped running descendants never appear here: they are
    either verified dead before the demotion or deferred whole.)
    """
    from hermes_cli.kanban_claims import _retry_status_for_run
    from hermes_cli.kanban_db import _append_event, _end_run, _insert_comment
    from hermes_cli.kanban_db_connect import write_txn
    from hermes_cli.kanban_worker_handoff import _mark_run_scope_stopping
    from hermes_cli.kanban_worker_identity import _defer_reclaim_for_live_worker, _terminate_reclaimed_worker, _worker_termination_tuple
    from hermes_cli.kanban_worker_stop import request_worker_scope_stop

    caller_owns_txn = bool(conn.in_transaction)
    now = int(time.time())
    invalidated: list[dict[str, Any]] = []
    terminations: list[
        tuple[Optional[int], Optional[str], Optional[int], Optional[str]]
    ] = []

    # Phase 0 — scoped running descendants: verified stop BEFORE any
    # demotion. Demoting flips the row to spawnable 'todo'; doing that
    # beside a live or still-draining scope is the duplication loop, so
    # an unconfirmed stop defers the whole descendant (claim held,
    # ``scope_stopping`` marked) and the crash-cleanup path requeues it
    # once the verified-stop service lands the kill. Runs before the
    # write transaction so the stop probe never sits inside it.
    # The probed run identity (current_run_id + worker fingerprint) is
    # captured here for EVERY running descendant — scoped or not (pass 8,
    # W) — and re-compared inside the transaction: a concurrent retry can
    # replace the descendant's run between the two phases (including
    # scoped -> unscoped, the spawn-fallback shape), and a stop verdict
    # about the OLD run must never demote the NEW one. The guard used to
    # key on the row still having a worker_scope, so a confirmed-dead
    # scoped run replaced by a fresh UNSCOPED run slipped past it.
    deferred: set[str] = set()
    probed_identity: dict[str, tuple[Any, Any, Any, Any]] = {}
    running_rows = conn.execute(
        """
        WITH RECURSIVE descendants(id) AS (
            SELECT child_id FROM task_links WHERE parent_id = ?
            UNION
            SELECT l.child_id
            FROM task_links l
            JOIN descendants d ON d.id = l.parent_id
        )
        SELECT t.id, t.claim_lock, t.worker_pid, t.worker_scope,
               t.current_run_id, t.worker_pid_started_at
        FROM descendants d
        JOIN tasks t ON t.id = d.id
        WHERE t.status = 'running'
        """,
        (task_id,),
    ).fetchall()
    for row in running_rows:
        probed_identity[row["id"]] = (
            row["current_run_id"],
            row["worker_pid"],
            row["worker_pid_started_at"],
            row["worker_scope"],
        )
        if not row["worker_scope"]:
            # Unscoped: nothing to stop here — the pid is terminated
            # post-commit; only its captured identity matters below.
            continue
        if request_worker_scope_stop(
            row["worker_scope"], task_id=row["id"], conn=conn,
        ):
            continue  # cgroup confirmed empty — safe to demote in-txn
        termination = {
            "prev_pid": row["worker_pid"],
            "host_local": True,
            "termination_attempted": True,
            "terminated": False,
            "sigkill": False,
            "scope_unit": row["worker_scope"],
            "scope_stopped": False,
        }
        _defer_reclaim_for_live_worker(
            conn, row["id"], row["claim_lock"], now, termination,
            reason="ancestor_reopen_scope_still_stopping",
        )
        _mark_run_scope_stopping(
            conn, row["id"], row["worker_scope"],
            reason="ancestor_reopen_stop_unconfirmed",
            expected_run_id=row["current_run_id"],
        )
        deferred.add(row["id"])
    with write_txn(conn, allow_nested=True):
        rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id
                FROM task_links l
                JOIN descendants d ON d.id = l.parent_id
            )
            SELECT t.id, t.status, t.current_run_id, t.worker_pid,
                   t.worker_pid_started_at, t.claim_lock, t.worker_scope
            FROM descendants d
            JOIN tasks t ON t.id = d.id
            ORDER BY t.id
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            previous_status = row["status"]
            if previous_status not in {"ready", "review", "running", "done"}:
                continue
            if row["id"] in deferred:
                # Scoped descendant whose stop is unconfirmed — kept
                # running with its claim by Phase 0; nothing to demote.
                continue
            if (
                previous_status == "running"
                and probed_identity.get(row["id"]) != (
                    row["current_run_id"],
                    row["worker_pid"],
                    row["worker_pid_started_at"],
                    row["worker_scope"],
                )
            ):
                # The run was replaced (or first appeared) after the
                # Phase 0 probe — scoped OR unscoped (pass 8, W): the
                # confirmed-empty verdict belonged to the previous run's
                # cgroup, not this worker's. Skip the demotion — the row
                # stays running and the next reconcile/invalidate pass
                # re-probes the new scope.
                _log.info(
                    "kanban: descendant %s of reopened %s changed runs "
                    "between the stop probe and the demotion (run %s -> "
                    "%s) — left running for the next pass",
                    row["id"], task_id,
                    probed_identity.get(row["id"], (None,))[0],
                    row["current_run_id"],
                )
                continue
            resume_status = "ready"
            run_id = None
            if previous_status == "review":
                resume_status = "review"
            elif previous_status == "running":
                resume_status = _retry_status_for_run(
                    conn, row["id"], row["current_run_id"]
                )
                if not row["worker_scope"]:
                    # Unscoped pid: kill post-commit (audit-first). Scoped
                    # rows here were verified dead in Phase 0 — no kill.
                    terminations.append(_worker_termination_tuple(row))
                run_id = _end_run(
                    conn, row["id"], outcome="reclaimed", status="todo",
                    summary=f"ancestor {task_id} reopened",
                )
            # consecutive_failures = 0: deliberate operator reset — see
            # docstring for why this diverges from reopen_review_task.
            conn.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                "worker_pid_started_at = NULL, worker_registered_at = NULL, "
                "worker_scope = NULL, "
                "current_run_id = NULL, consecutive_failures = 0 WHERE id = ?",
                (row["id"],),
            )
            entry = {
                "id": row["id"], "prior_status": previous_status,
                "new_status": "todo", "resume_status": resume_status,
            }
            _append_event(
                conn, row["id"], "descendant_invalidated",
                {"ancestor": task_id, **{k: v for k, v in entry.items() if k != "id"}},
                run_id=run_id,
            )
            # Legacy 'status' event so existing live-feed consumers still see
            # the move without learning the new event kind.
            _append_event(
                conn, row["id"], "status",
                {
                    "status": "todo", "reason": "ancestor_reopened", "parent": task_id,
                    "previous_status": previous_status, "resume_status": resume_status,
                },
                run_id=run_id,
            )
            _insert_comment(
                conn, row["id"], author, f"Invalidated: ancestor {task_id} was reopened; "
                f"retracted from '{previous_status}' to 'todo' "
                f"(will resume via '{resume_status}').", now,
            )
            invalidated.append(entry)
    if not caller_owns_txn:
        # Standalone call: we committed above, so the audit trail is durable
        # — safe to kill workers now. Composed calls leave this to the
        # caller (post-commit), preserving events-before-termination.
        # Scope-aware: a running descendant's whole unit is stopped, so
        # descendants it spawned die with the retraction.
        for pid, claim_lock, pid_started, scope in terminations:
            _terminate_reclaimed_worker(
                pid, claim_lock,
                scope_unit=scope or None,
                pid_started_at=pid_started,
            )
    return {"invalidated": invalidated, "terminations": terminations}


def specify_triage_task(
    conn: sqlite3.Connection, task_id: str, *, title: Optional[str] = None,
    body: Optional[str] = None, assignee: Optional[str] = None, author: Optional[str] = None,
) -> bool:
    """Update title/body/assignee (when given) and move ``triage -> todo`` in one
    txn; False when not in triage. Lands in ``todo`` (not ``ready``) so parent
    gating still applies; the audit comment is written only when a field changed.
    """
    from hermes_cli.kanban_db import _append_event, _canonical_assignee, _insert_comment
    from hermes_cli.kanban_db_connect import write_txn

    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = _canonical_assignee(assignee)
    with write_txn(conn):
        existing = conn.execute(
            "SELECT title, body, assignee FROM tasks WHERE id = ? AND status = 'triage'",
            (task_id,),
        ).fetchone()
        if existing is None:
            return False
        sets: list[str] = ["status = 'todo'"]
        params: list[Any] = []
        changed_fields: list[str] = []
        if title is not None and title.strip() != (existing["title"] or ""):
            sets.append("title = ?")
            params.append(title.strip())
            changed_fields.append("title")
        if body is not None and (body or "") != (existing["body"] or ""):
            sets.append("body = ?")
            params.append(body)
            changed_fields.append("body")
        if assignee is not None and assignee != (existing["assignee"] or None):
            sets.append("assignee = ?")
            params.append(assignee)
            changed_fields.append("assignee")
        params.append(task_id)
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} "
            f"WHERE id = ? AND status = 'triage'", tuple(params),
        )
        if cur.rowcount != 1:
            return False
        if changed_fields and author and author.strip():
            # Not add_comment (own txn + 'commented' event); 'specified' below records it.
            _insert_comment(
                conn, task_id, author.strip(),
                "Specified — updated " + ", ".join(changed_fields) + " and promoted to todo.",
                int(time.time()),
            )
        _append_event(
            conn, task_id, "specified",
            {"changed_fields": changed_fields} if changed_fields else None,
        )
    # Own IMMEDIATE txn (outside the one above): a parent-free specified task
    # flips to 'ready' now instead of idling until the next tick.
    recompute_ready(conn)
    return True


def archive_task(conn: sqlite3.Connection, task_id: str) -> bool:
    from hermes_cli.kanban_db import _append_event, _end_run
    from hermes_cli.kanban_db_connect import write_txn
    from hermes_cli.kanban_db_workspace import _cleanup_workspace
    from hermes_cli.kanban_worker_scope import _stop_kanban_worker_scope

    with write_txn(conn):
        prior = conn.execute(
            "SELECT worker_scope FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        prior_scope = prior["worker_scope"] if prior else None
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', "
            "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
            "    worker_pid_started_at = NULL, worker_registered_at = NULL, "
            "    worker_scope = NULL "
            "WHERE id = ? AND status != 'archived'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        # Archived mid-run (dashboard): close the run so history isn't orphaned.
        run_id = _end_run(
            conn, task_id, outcome="reclaimed", status="reclaimed",
            summary="task archived with run still active",
        )
        _append_event(conn, task_id, "archived", None, run_id=run_id)
    # Archiving a still-running task must not leave its scoped worker
    # (and whatever it spawned) running untracked — verified stop, with
    # the dispatcher's audit sweep as the backstop if this one fails.
    if prior_scope:
        _stop_kanban_worker_scope(prior_scope)
    # ``archived`` parents no longer block children, same as ``done``.
    # Promote newly-unblocked dependents immediately instead of waiting
    # for a later dispatcher tick.
    recompute_ready(conn)
    # Reap the workspace on archive too (never-completed tasks kept it forever).
    _cleanup_workspace(conn, task_id)
    return True


def _delete_task_relations(conn: sqlite3.Connection, task_id: str) -> None:
    """Delete every row referencing ``task_id`` (schema has no ON DELETE CASCADE)."""
    conn.execute("DELETE FROM task_links WHERE parent_id = ? OR child_id = ?", (task_id, task_id))
    for table in ("task_comments", "task_events", "task_runs", "kanban_notify_subs"):
        conn.execute(f"DELETE FROM {table} WHERE task_id = ?", (task_id,))


def delete_archived_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete an ARCHIVED task (+ related rows); anything else must be
    archived first so data loss takes two deliberate actions."""
    from hermes_cli.kanban_db import _task_status
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        if _task_status(conn, task_id) != "archived":
            return False
        _delete_task_relations(conn, task_id)
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount == 1


def delete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete a task and its related rows in one txn; False when not found."""
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if cur.rowcount != 1:
            return False
        _delete_task_relations(conn, task_id)
    recompute_ready(conn)
    return True


def schedule_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Park in ``scheduled`` (waiting on time, not a human; not dispatchable)
    until ``unblock_task`` re-gates it."""
    from hermes_cli.kanban_db import _append_event, _end_or_synthesize_run
    from hermes_cli.kanban_db_connect import write_txn
    from hermes_cli.kanban_worker_scope import _stop_kanban_worker_scope

    with write_txn(conn):
        prior = conn.execute(
            "SELECT worker_scope FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        prior_scope = prior["worker_scope"] if prior else None
        params: list[Any] = [task_id]
        sql = """
            UPDATE tasks
               SET status       = 'scheduled',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL,
                   worker_pid_started_at = NULL,
                   worker_registered_at = NULL,
                   worker_scope = NULL
             WHERE id = ?
               AND status IN ('todo', 'ready', 'running', 'blocked')
        """
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params.append(int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        run_id = _end_or_synthesize_run(
            conn, task_id, outcome="scheduled", status="scheduled", summary=reason, synthesize=bool(reason),
        )
        _append_event(conn, task_id, "scheduled", {"reason": reason}, run_id=run_id)
    # Scheduling away a still-running task must not leave its scoped
    # worker running untracked — verified stop (audit sweep backstops).
    if prior_scope:
        _stop_kanban_worker_scope(prior_scope)
    return True


def _resolve_request_review_reviewer(
    conn: sqlite3.Connection,
    task_id: str,
    reviewer: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve the reviewer for :func:`request_review` — shared by the
    synchronous transition and the deferred own-worker handoff (pass 8,
    V) so both fail identically on missing re-review provenance.

    Returns ``(canonical_reviewer_or_None, error_reason_or_None)``: an
    explicit reviewer wins, otherwise re-review reuses the reviewer
    provenance persisted by the latest ``changes_requested`` event.
    """
    from hermes_cli.kanban_db import _canonical_assignee

    if reviewer is not None:
        return _canonical_assignee(reviewer), None
    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    changes_event = None
    if changes_run is not None:
        changes_event = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND run_id = ? "
            "AND kind = 'changes_requested' "
            "ORDER BY id DESC LIMIT 1",
            (task_id, int(changes_run["id"])),
        ).fetchone()
    try:
        changes_payload = (
            json.loads(changes_event["payload"])
            if changes_event and changes_event["payload"]
            else {}
        )
    except (json.JSONDecodeError, TypeError):
        changes_payload = {}
    prior_reviewer = (
        changes_payload.get("reviewer")
        if isinstance(changes_payload, dict)
        else None
    )
    if changes_run is not None:
        if not isinstance(prior_reviewer, str) or not prior_reviewer.strip():
            return None, (
                "re-review has no durable reviewer provenance (the "
                "latest changes_requested event is missing or "
                "malformed); pass reviewer= explicitly"
            )
        return _canonical_assignee(prior_reviewer), None
    return None, None


def _review_handoff_provenance(
    conn: sqlite3.Connection,
    task_id: str,
    current_run_id: int,
) -> tuple[Optional[str], Optional[str]]:
    """The claimed-from-review + implementer checks for
    :func:`request_changes` — shared by the synchronous transition and
    the deferred own-worker handoff (pass 8, V).

    Returns ``(implementer, error_reason_or_None)`` with exactly the
    errors the transaction would raise, so a deferral never accepts a
    request the synchronous path would refuse.
    """
    claimed_event = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND run_id = ? AND kind = 'claimed' "
        "ORDER BY id DESC LIMIT 1",
        (task_id, int(current_run_id)),
    ).fetchone()
    try:
        claimed_payload = (
            json.loads(claimed_event["payload"])
            if claimed_event and claimed_event["payload"]
            else {}
        )
    except (json.JSONDecodeError, TypeError):
        claimed_payload = {}
    if not isinstance(claimed_payload, dict):
        claimed_payload = {}
    if claimed_payload.get("source_status") != "review":
        return None, "active run was not claimed from review"
    requested_event = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if requested_event is None:
        return None, "no prior review_requested event"
    try:
        requested_payload = (
            json.loads(requested_event["payload"])
            if requested_event["payload"]
            else {}
        )
    except (json.JSONDecodeError, TypeError):
        requested_payload = {}
    if not isinstance(requested_payload, dict):
        requested_payload = {}
    implementer = requested_payload.get("implementer")
    if not isinstance(implementer, str) or not implementer.strip():
        return None, "review handoff has no valid implementer provenance"
    return implementer, None
