"""Kanban worker lifecycle; integrated from PR #101911 by danashburn.

Attempt ownership and persistence use the current main Kanban modules.
"""

from __future__ import annotations

import hermes_cli.kanban_db as _owner_kanban_db

import hermes_cli.kanban_db_boards as _owner_kanban_boards
import hermes_cli.kanban_claims as _owner_kanban_claims
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


_SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK = 3


_scope_audit_first_seen: dict[str, float] = {}


_scope_audit_cursor: int = 0


def enforce_max_runtime(conn: sqlite3.Connection, *, signal_fn=None) -> list[str]:
    """Settle an expired attempt only after its owned worker has stopped."""
    now = int(time.time())
    timed_out = []
    rows = conn.execute(
        "SELECT t.*, COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t LEFT JOIN task_runs r ON r.id=t.current_run_id "
        "WHERE t.status='running' AND t.max_runtime_seconds IS NOT NULL "
        "AND t.worker_pid IS NOT NULL"
    ).fetchall()
    for row in rows:
        if not (row["claim_lock"] or "").startswith(_kanban_db._host_prefix()):
            continue
        if row["active_started_at"] is None:
            continue
        elapsed = now - int(row["active_started_at"])
        if elapsed < int(row["max_runtime_seconds"]):
            continue
        reserved = _kanban_claims.reserve_reclaim(conn, row["id"], row, now=now)
        if reserved is None:
            continue
        termination = _stop_reserved_worker(conn, reserved, signal_fn=signal_fn)
        if _kanban_worker_identity._worker_survived_termination(termination):
            _kanban_worker_identity._defer_reclaim_for_live_worker(
                conn,
                row["id"],
                row["claim_lock"],
                now,
                termination,
                reason="runtime_expired_worker_alive",
                expected_run_id=row["current_run_id"],
            )
            continue
        error = f"elapsed {elapsed}s >= limit {row['max_runtime_seconds']}s"
        payload = dict(
            termination,
            pid=row["worker_pid"],
            elapsed_seconds=elapsed,
            limit_seconds=int(row["max_runtime_seconds"]),
        )
        if _settle_recovered_run(
            conn, reserved, "timed_out", error, payload, count_failure=True
        ):
            timed_out.append(row["id"])
    return timed_out


_STALE_HEARTBEAT_GAP_SECONDS = 3600


def detect_stale_running(
    conn: sqlite3.Connection, *, stale_timeout_seconds: int = 0, signal_fn=None
) -> list[str]:
    """Reserve an unchanged stale attempt before terminating its worker."""
    if stale_timeout_seconds <= 0:
        return []
    now = int(time.time())
    stale = []
    rows = conn.execute(
        "SELECT t.*, COALESCE(r.started_at,t.started_at) AS active_started_at "
        "FROM tasks t LEFT JOIN task_runs r ON r.id=t.current_run_id "
        "WHERE t.status='running'"
    ).fetchall()
    for row in rows:
        if row["active_started_at"] is None:
            continue
        elapsed = now - int(row["active_started_at"])
        hb = row["last_heartbeat_at"]
        age = None if hb is None else now - int(hb)
        if elapsed < stale_timeout_seconds or (
            age is not None and age < _STALE_HEARTBEAT_GAP_SECONDS
        ):
            continue
        reserved = _kanban_claims.reserve_reclaim(conn, row["id"], row, now=now)
        if reserved is None:
            continue
        termination = _stop_reserved_worker(conn, reserved, signal_fn=signal_fn)
        if _kanban_worker_identity._worker_survived_termination(termination):
            _kanban_worker_identity._defer_reclaim_for_live_worker(
                conn,
                row["id"],
                row["claim_lock"],
                now,
                termination,
                reason="heartbeat_stale_worker_alive",
                expected_run_id=row["current_run_id"],
            )
            continue
        error = (
            f"no heartbeat for {age}s" if age is not None else "no heartbeat ever"
        ) + f" after {elapsed}s running"
        payload = dict(
            termination,
            elapsed_seconds=elapsed,
            last_heartbeat_at=hb,
            heartbeat_age_seconds=age,
            timeout_seconds=stale_timeout_seconds,
            pid=row["worker_pid"],
        )
        if _settle_recovered_run(conn, reserved, "stale", error, payload):
            stale.append(row["id"])
    return stale


def reconcile_orphaned_running(
    conn: sqlite3.Connection,
) -> list[str]:
    """Reconcile ``running`` cards whose claim bookkeeping is broken.

    Tracked-state vs. reality divergence: a task can sit in
    ``status='running'`` with ``claim_lock IS NULL`` or ``claim_expires IS
    NULL`` (crash mid-claim, manual SQL, DB restore). None of the other
    recovery paths ever touch such a card — ``release_stale_claims``
    requires a non-NULL ``claim_expires``, ``detect_crashed_workers``
    requires a host-local claim_lock + worker_pid, and
    ``detect_stale_running`` is disabled by default — so the card shows
    Running forever (a zombie).

    This pass finds those orphans, requeues them to ``ready`` with an
    explanatory comment, closes any leaked run, and appends a
    ``reconciled`` event. If the orphan row still records a live PID on
    this host, requeueing is deferred to a later tick so we never spawn a
    duplicate beside a possibly-alive worker.

    Returns the list of reconciled task ids. Safe to call every tick.

    Idea from openai/symphony's tracker reconciliation (Apache-2.0).
    """
    now = int(time.time())
    reconciled: list[str] = []
    rows = conn.execute(
        "SELECT * "
        "FROM tasks "
        "WHERE status = 'running' "
        "  AND (claim_lock IS NULL OR claim_expires IS NULL)"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        pid = row["worker_pid"]
        alive, alive_reason = _kanban_worker_identity._run_worker_alive(row)
        if pid and alive:
            # The recorded worker may still be doing real work — never
            # requeue beside a live process. Retry next tick.
            _log.debug(
                "kanban reconcile: task %s has broken claim bookkeeping but "
                "worker pid %s is alive (%s) on this host — deferring",
                tid,
                pid,
                alive_reason,
            )
            continue
        # Worker is gone. If it ran in a scope, reap the unit so nothing
        # it double-forked survives the requeue. An unconfirmed stop
        # defers the requeue for the same reason as a live pid.
        if row["worker_scope"]:
            if not _kanban_worker_stop.request_worker_scope_stop(
                row["worker_scope"],
                task_id=tid,
                conn=conn,
            ):
                _log.debug(
                    "kanban reconcile: task %s scope %s still stopping — "
                    "deferring requeue to next tick",
                    tid,
                    row["worker_scope"],
                )
                continue
        with _kanban_db_connect.write_txn(conn):
            retry_status = _kanban_claims._retry_status_for_run(
                conn, tid, row["current_run_id"]
            )
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "worker_pid_started_at = NULL, "
                "worker_registered_at = NULL, worker_scope = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ? AND claim_expires IS ? AND current_run_id IS ? "
                "AND worker_pid IS ? AND worker_pid_started_at IS ? AND worker_scope IS ? "
                "AND last_heartbeat_at IS ?",
                (
                    retry_status,
                    tid,
                    row["claim_lock"],
                    row["claim_expires"],
                    row["current_run_id"],
                    row["worker_pid"],
                    row["worker_pid_started_at"],
                    row["worker_scope"],
                    row["last_heartbeat_at"],
                ),
            )
            if cur.rowcount != 1:
                continue
            payload = {
                "reason": "orphaned_running",
                "retry_status": retry_status,
                "claim_lock": row["claim_lock"],
                "claim_expires": (
                    int(row["claim_expires"])
                    if row["claim_expires"] is not None
                    else None
                ),
                "worker_pid": int(pid) if pid else None,
                "now": now,
            }
            run_id = _kanban_db._end_run(
                conn,
                tid,
                outcome="reclaimed",
                status="reclaimed",
                error="orphaned running card (broken claim bookkeeping)",
                metadata=payload,
            )
            # Inline comment INSERT — add_comment opens its own write_txn
            # and would raise on nesting (see write_txn pitfalls).
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    tid,
                    "dispatcher",
                    "reconciliation: card was 'running' with no valid claim "
                    f"(dead/gone worker) — requeued to {retry_status}",
                    now,
                ),
            )
            _kanban_db._append_event(conn, tid, "reconciled", payload, run_id=run_id)
            reconciled.append(tid)
        _log.info(
            "kanban reconcile: requeued orphaned running task %s "
            "(claim_lock=%r, worker_pid=%r)",
            tid,
            row["claim_lock"],
            pid,
        )
    return reconciled


def fail_unregistered_workers(conn: sqlite3.Connection) -> list[str]:
    """Fail ``running`` tasks whose scoped worker never self-registered.

    For scoped runs the dispatcher records the LAUNCHER's pid (the
    ``systemd-run`` client), not the worker's; the worker proves life by
    calling ``register_worker_pid`` from its heartbeat bridge on first
    activity. A row that stays unregistered past
    ``WORKER_REGISTRATION_GRACE_SECONDS`` breaks the launch contract —
    whether the exec died before the first tool call or the worker wedged
    pre-heartbeat, nothing downstream (adoption, crash detection) can
    vouch for it — so the scope is stopped (verified) and the run is
    recorded as ``spawn_failed``.

    Also lazily normalizes unscoped running rows with a pid: for a plain
    spawn the recorded pid IS the worker, so backfill
    ``worker_registered_at`` (this also covers true legacy rows from
    before the isolation feature: they are unscoped by definition, since
    the ``worker_scope`` column shipped with it).

    Returns the list of failed task ids. Safe to call every tick.
    """
    now = int(time.time())
    failed: list[str] = []
    rows = conn.execute(
        "SELECT t.*, r.started_at AS run_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.worker_pid IS NOT NULL "
        "  AND t.worker_registered_at IS NULL"
    ).fetchall()
    for row in rows:
        scope = row["worker_scope"]
        if not scope:
            # Lazy normalization for unscoped rows (plain spawns: the
            # recorded pid is the worker itself).
            with _kanban_db_connect.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET worker_registered_at = ? "
                    "WHERE id = ? AND worker_scope IS NULL "
                    "  AND worker_pid IS NOT NULL "
                    "  AND worker_registered_at IS NULL",
                    (now, row["id"]),
                )
            continue
        started = row["run_started_at"] or row["started_at"]
        if (
            started is None
            or (now - int(started))
            < _kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS
        ):
            continue  # still inside the launch grace window
        # Registration-race guard (review finding a), first half: the
        # snapshot above is a moment in time. Re-read under the write
        # lock so a worker that registered since the snapshot survives
        # this sweep — its scope must not be stopped on stale evidence.
        # Third half (finding J): the stop below may be QUEUED, not
        # inline — the worker can register while it sits in the queue, so
        # the request carries skip_if_registered and the service re-checks
        # the row immediately before acting.
        with _kanban_db_connect.write_txn(conn):
            fresh = conn.execute(
                "SELECT status, worker_registered_at FROM tasks WHERE id = ?",
                (row["id"],),
            ).fetchone()
            if fresh is None or fresh["status"] != "running":
                continue  # a terminal path already moved the row on
            if fresh["worker_registered_at"] is not None:
                continue  # registered since the snapshot — it is alive
        if not _kanban_worker_stop.request_worker_scope_stop(
            scope,
            task_id=row["id"],
            skip_if_registered=True,
            conn=conn,
        ):
            continue  # stop unconfirmed — retry next tick
        _kanban_db_dispatch._record_task_failure(
            conn,
            row["id"],
            error=(
                f"worker never registered its pid within "
                f"{_kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS}s (scope {scope}); "
                "launch failed or the worker wedged before its first "
                "activity"
            ),
            require_worker_unregistered=True,
            expected_run_id=row["current_run_id"],
            expected_worker_scope=scope,
            outcome="spawn_failed",
            release_claim=True,
            end_run=True,
        )
        # Second half of the race guard: a worker can register WHILE the
        # verified stop runs. The CAS inside the failure record aborted
        # in that case (no failure counted, bookkeeping untouched) —
        # detect it and leave the row alone; adoption and the crash
        # sweep own it from here.
        fresh = conn.execute(
            "SELECT worker_registered_at FROM tasks WHERE id = ?",
            (row["id"],),
        ).fetchone()
        if fresh is not None and fresh["worker_registered_at"] is not None:
            _log.info(
                "kanban: task %s worker registered while its "
                "unregistered-worker stop ran — leaving it running",
                row["id"],
            )
            continue
        failed.append(row["id"])
        _log.warning(
            "kanban: task %s scoped worker never registered within %ss "
            "(scope %s) — recorded as spawn failure",
            row["id"],
            _kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS,
            scope,
        )
    return failed


def _claimed_worker_scopes_globally() -> Optional[set[str]]:
    """Scope units claimed by ``running`` rows across ALL boards.

    The audit sweep lists scopes host-globally (the systemd user bus has
    no board concept), so its notion of "claimed" must be global too — a
    per-board view would let board A's sweep reap a live worker that
    board B is running. Membership is by ``tasks.worker_scope`` equality,
    NOT by parsing the task id back out of the unit name: unit names are
    sanitised (``_SCOPE_UNIT_UNSAFE`` chars replaced), so a parse-back
    can silently disagree with the row id it came from.

    Returns ``None`` when any board's database cannot be read — the
    sweep then reaps NOTHING, because "cannot read a board" must never
    look like "that board has no live scopes".
    """
    claimed: set[str] = set()
    try:
        boards = _owner_kanban_boards.list_boards(include_archived=False)
    except Exception as exc:
        _log.debug("kanban: scope sweep cannot list boards: %s", exc)
        return None
    slugs = [b.get("slug") for b in boards if b.get("slug")] or [
        _owner_kanban_boards.DEFAULT_BOARD
    ]
    for slug in slugs:
        try:
            path = _owner_kanban_db.kanban_db_path(board=slug)
            if not path.exists():
                continue  # a board with no DB cannot hold running workers
            ro = sqlite3.connect(
                f"file:{path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            try:
                claimed.update(
                    r[0]
                    for r in ro.execute(
                        "SELECT DISTINCT worker_scope FROM tasks "
                        "WHERE status = 'running' "
                        "  AND worker_scope IS NOT NULL"
                    )
                )
            finally:
                ro.close()
        except Exception as exc:
            _log.debug(
                "kanban: scope sweep cannot read board %s: %s",
                slug,
                exc,
            )
            return None
    return claimed


def _live_untracked_worker_runs() -> Optional[dict[str, tuple[str, str, int]]]:
    """Live runs whose worker scope is NOT recorded, keyed by task id.

    ``{sanitised_task_id: (board_slug, task_id, run_id)}`` for every
    ``running`` row across all boards whose ``worker_scope`` is NULL and
    whose attempt is still open. These are the runs the audit sweep
    cannot recognise through :func:`_claimed_worker_scopes_globally`
    (which matches by recorded unit name), and on this branch's own
    deploy they are REAL: a worker spawned by the pre-fix build on a
    managed gateway with ``worker_isolation: none`` runs inside
    ``hermes-worker-kanban-<task>-run-<n>.scope`` while its row records
    no scope at all. Without this map the first tick after the upgrade
    would list that unit, find no claim, and reap a LIVE worker.

    The key is the SANITISED id (what a unit name can encode), so a
    sweep can look a parsed unit name up here; an id whose sanitised
    form collides with another task's is dropped rather than guessed,
    because an ambiguous owner must never adopt a unit.

    Returns ``None`` when any board is unreadable — same fail-closed
    rule as :func:`_claimed_worker_scopes_globally`: "cannot read a
    board" must never look like "that board has no live workers".
    """
    live: dict[str, tuple[str, str, int]] = {}
    ambiguous: set[str] = set()
    try:
        boards = _owner_kanban_boards.list_boards(include_archived=False)
    except Exception as exc:
        _log.debug("kanban: scope sweep cannot list boards: %s", exc)
        return None
    slugs = [b.get("slug") for b in boards if b.get("slug")] or [
        _owner_kanban_boards.DEFAULT_BOARD
    ]
    for slug in slugs:
        try:
            path = _owner_kanban_db.kanban_db_path(board=slug)
            if not path.exists():
                continue
            ro = sqlite3.connect(
                f"file:{path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            try:
                rows = ro.execute(
                    "SELECT t.id, r.id FROM tasks t "
                    "JOIN task_runs r ON r.id = t.current_run_id "
                    "WHERE t.status = 'running' "
                    "  AND t.worker_scope IS NULL "
                    "  AND r.status = 'running'"
                ).fetchall()
            finally:
                ro.close()
        except Exception as exc:
            _log.debug(
                "kanban: scope sweep cannot read board %s: %s",
                slug,
                exc,
            )
            return None
        for task_id, run_id in rows:
            key = _kanban_worker_scope._SCOPE_UNIT_UNSAFE.sub("-", str(task_id)).strip(
                "-."
            )
            if not key:
                # An id that sanitises away entirely is minted as the
                # shared literal "task" by _kanban_worker_scope_unit, so
                # its unit name identifies nothing — never adoptable.
                continue
            owner = (slug, str(task_id), int(run_id))
            if key in live and live[key] != owner:
                ambiguous.add(key)
            live[key] = owner
            qualified = _kanban_worker_scope._scope_task_key(task_id, db_path=str(path))
            live[qualified] = owner
    for key in ambiguous:
        live.pop(key, None)
    return live


def _untracked_owner_of_scope_unit(
    unit: str, live_runs: dict[str, tuple[str, str, int]]
) -> Optional[tuple[str, str, int]]:
    """The live run *unit* belongs to, or ``None`` when nothing owns it.

    Ownership is proven, not guessed: the task id is parsed back out of
    the unit name (:func:`_task_id_from_kanban_scope_unit`), looked up
    among the runs that record no scope
    (:func:`_live_untracked_worker_runs`), and then the unit must be
    EXACTLY one of the two names that run's own attempt would produce —
    kanban's isolation unit or the restart-safe wrap's. Requiring the
    attempt's own name is what keeps a lingering unit from an EARLIER
    attempt of the same task reapable: only the current run's unit is
    adopted.
    """
    task_id = _kanban_worker_scope._scope_identity_key_from_unit(unit)
    if task_id is None:
        return None
    owner = live_runs.get(task_id)
    if owner is None:
        return None
    slug, real_task_id, run_id = owner
    candidates = {
        _kanban_worker_scope._kanban_worker_scope_unit(
            real_task_id, run_id, board=slug
        ),
        f"hermes-worker-kanban-{_kanban_worker_scope._scope_task_key(real_task_id, board=slug)}-run-{run_id}.scope",
        f"hermes-worker-kanban-{real_task_id}-run-{run_id}.scope",
        f"hermes-kanban-{real_task_id}-r{run_id}.scope",
    }
    return owner if unit in candidates else None


def _record_adopted_worker_scope(
    board: str, task_id: str, run_id: int, unit: str
) -> None:
    """Backfill ``worker_scope`` on an adopted run (task row + run row).

    Best-effort and idempotent: written only while the row still holds
    the NULL the adoption was decided on, so a worker that registered
    its own scope in the meantime wins. Once recorded, the unit is
    ordinary tracked state — the normal teardown stops it and the next
    sweep recognises it through the claimed-scope path instead of
    re-deriving the adoption.

    Recording a scope also moves the row under the never-registered
    launch grace, so the same write carries the normalization
    :func:`fail_unregistered_workers` already applies to every UNSCOPED
    running row with a pid: mark it registered. The row was written by a
    build that recorded no scope, and under that build its pid was
    treated as the worker's — adoption must not turn a healthy long-lived
    worker into a past-grace ``spawn_failed`` just because the sweep
    learned which unit it lives in. Doing it inside the adoption write
    (rather than leaning on the tick's call order) makes the two writes
    inseparable.
    """
    conn = None
    try:
        path = _owner_kanban_db.kanban_db_path(board=board)
        if not path.exists():
            return
        conn = _kanban_db_connect.connect(db_path=path)
        with _kanban_db_connect.write_txn(conn):
            adopted = conn.execute(
                "UPDATE tasks SET worker_scope = ? "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_scope IS NULL",
                (unit, task_id),
            ).rowcount
            if not adopted:
                return  # the row moved on — leave every field alone
            conn.execute(
                "UPDATE tasks SET worker_registered_at = ? "
                "WHERE id = ? AND worker_pid IS NOT NULL "
                "  AND worker_registered_at IS NULL",
                (int(time.time()), task_id),
            )
            conn.execute(
                "UPDATE task_runs SET worker_scope = ? "
                "WHERE id = ? AND task_id = ? AND worker_scope IS NULL",
                (unit, run_id, task_id),
            )
    except Exception as exc:
        _log.debug(
            "kanban: cannot backfill worker_scope %s on task %s: %s",
            unit,
            task_id,
            exc,
        )
    finally:
        if conn is not None:
            conn.close()


def _task_has_registered_worker(
    task_id: str, expected_scope: Optional[str] = None
) -> bool:
    """True when a board's ``running`` row for *task_id* has a registered
    worker — fresh evidence that an "unregistered launch" diagnosis is
    stale (finding J). Read across ALL boards with the same read-only
    pattern as :func:`_claimed_worker_scopes_globally`, because the
    scope-stop service is host-global while the row lives on one board.

    Individual board read failures are ignored (an unreadable board can
    neither confirm nor refute); False overall means "act on the
    enqueue-time evidence", which next ticks simply re-verify.
    """
    try:
        boards = _owner_kanban_boards.list_boards(include_archived=False)
    except Exception:
        return False
    slugs = [b.get("slug") for b in boards if b.get("slug")] or [
        _owner_kanban_boards.DEFAULT_BOARD
    ]
    for slug in slugs:
        try:
            path = _owner_kanban_db.kanban_db_path(board=slug)
            if not path.exists():
                continue
            ro = sqlite3.connect(
                f"file:{path}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            try:
                row = ro.execute(
                    "SELECT 1 FROM tasks WHERE id = ? "
                    "  AND status = 'running' "
                    "  AND worker_registered_at IS NOT NULL "
                    "AND (? IS NULL OR worker_scope = ?) LIMIT 1",
                    (task_id, expected_scope, expected_scope),
                ).fetchone()
            finally:
                ro.close()
            if row is not None:
                return True
        except Exception as exc:
            _log.debug(
                "kanban: registration re-check cannot read board %s: %s",
                slug,
                exc,
            )
    return False


def _mark_run_stop_pending(task_id: str, expected_scope: Optional[str] = None) -> bool:
    """CAS the current run of *task_id* as ``stop_pending`` (pass 4, R).

    Runs on the scope-stop service immediately before it signals a
    registration-sensitive stop. The UPDATE matches only while the task
    is still ``running`` AND unregistered: a registration that already
    committed makes it a no-op (rowcount 0 → False; the caller stands
    the stop down). Once the marker is committed,
    :func:`register_worker_pid` refuses the run — both writes take the
    database write lock, so no interleaving survives between the
    service's re-check and the signal. Re-marking an already-marked run
    still matches (SQLite counts no-op updates), so re-enqueued retry
    stops are not stood down by their own earlier marker.

    Writes across ALL boards with the same traversal as
    :func:`_task_has_registered_worker`; per-board failures are ignored
    and the walk continues (a CAS elsewhere still closes the race). No
    match anywhere → False, which callers treat as "do not signal".
    """
    try:
        boards = _owner_kanban_boards.list_boards(include_archived=False)
    except Exception:
        return False
    slugs = [b.get("slug") for b in boards if b.get("slug")] or [
        _owner_kanban_boards.DEFAULT_BOARD
    ]
    for slug in slugs:
        conn = None
        try:
            path = _owner_kanban_db.kanban_db_path(board=slug)
            if not path.exists():
                continue
            conn = _kanban_db_connect.connect(db_path=path)
            with _kanban_db_connect.write_txn(conn):
                cur = conn.execute(
                    "UPDATE task_runs SET stop_pending = 1 "
                    "WHERE id = (SELECT current_run_id FROM tasks "
                    "            WHERE id = ? AND status = 'running' "
                    "              AND worker_registered_at IS NULL "
                    "AND (? IS NULL OR worker_scope = ?))",
                    (task_id, expected_scope, expected_scope),
                )
                marked = cur.rowcount == 1
            if marked:
                return True
        except Exception as exc:
            _log.debug(
                "kanban: stop-pending CAS cannot write board %s: %s",
                slug,
                exc,
            )
        finally:
            if conn is not None:
                conn.close()
    return False


def _clear_run_stop_pending(task_id: str, expected_scope: Optional[str] = None) -> None:
    """Clear the current run's ``stop_pending`` marker (pass 8, AC).

    Called by the scope-stop service once a registration-sensitive stop
    CONFIRMS: the marker made ``register_worker_pid`` self-abort while
    the service was about to signal the scope, but a verified-empty
    cgroup means there is nothing left to signal. Without this clearing
    path the marker was write-only — a run marked and later re-adopted
    (gateway restart before the run row closed) would refuse its
    worker's registration forever. ``_end_run`` clears the marker with
    the same UPDATE for every run it closes.

    Best-effort across all boards (same traversal as
    :func:`_mark_run_stop_pending`): a failed clear retries on the next
    confirmed stop or dies with the run at ``_end_run``.
    """
    try:
        boards = _owner_kanban_boards.list_boards(include_archived=False)
    except Exception:
        return
    slugs = [b.get("slug") for b in boards if b.get("slug")] or [
        _owner_kanban_boards.DEFAULT_BOARD
    ]
    for slug in slugs:
        conn = None
        try:
            path = _owner_kanban_db.kanban_db_path(board=slug)
            if not path.exists():
                continue
            conn = _kanban_db_connect.connect(db_path=path)
            with _kanban_db_connect.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET stop_pending = 0 "
                    "WHERE id = (SELECT current_run_id FROM tasks "
                    "            WHERE id = ? AND (? IS NULL OR worker_scope = ?)) AND stop_pending = 1",
                    (task_id, expected_scope, expected_scope),
                )
        except Exception as exc:
            _log.debug(
                "kanban: stop-pending clear cannot write board %s: %s",
                slug,
                exc,
            )
        finally:
            if conn is not None:
                conn.close()


def reap_orphaned_worker_scopes(conn: sqlite3.Connection) -> list[str]:
    """Stop systemd scopes that no longer belong to any running task.

    The audit backstop for every fire-and-forget stop the domain layer
    fires from the worker's own side (completion, block, review paths):
    the worker cannot wait on its own scope teardown, so those stops are
    detached and verified here instead. This sweep lists every active
    ``hermes-kanban-*`` and ``hermes-worker-kanban-*`` scope (kanban's own
    isolation units and the shared restart-safe wrap's) on the user bus
    and stops the ones no
    running task claims — leaked descendants, scopes from tasks that
    completed without a stop, or a stale unit name left over from a
    retry that already moved on to a new unique unit.

    "No running task claims it" is decided on the recorded unit name
    FIRST and, for what is left, on the unit's own name second: a live
    unit that names a running task's CURRENT attempt while that run
    records no scope is ADOPTED (logged, and the unit written back to
    the row) rather than reaped. That case is not hypothetical — a
    worker spawned by a build that predates this branch, on a managed
    gateway with ``worker_isolation: none``, is running inside
    ``hermes-worker-kanban-<task>-run-<n>.scope`` with a NULL
    ``worker_scope``, and the first sweep after the upgrade would
    otherwise kill it.

    The sweep runs inside the dispatch tick, under the dispatcher lock,
    and every unit it touches costs a synchronous bus interaction (a
    liveness probe for orphans, a collect for terminal-but-loaded
    units) — with many orphans that held the lock for N probe timeouts
    (pass 8, AD). The synchronous work is therefore bounded per tick
    (``_SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK``) and ordered OLDEST
    orphan first, so an old leak never starves behind fresh ones; the
    remainder is deferred to the next tick, when the listing rediscovers
    it with its original first-seen stamp. The per-tick window also
    ROTATES across ticks (pass 9, AI): a persistent cursor round-robins
    over the age-sorted list, so permanently wedged units at its head
    cannot occupy the bound forever and starve every newer orphan —
    every orphan is probed within a bounded number of ticks.

    Returns the list of reaped unit names. Safe to call every tick.
    """
    if (
        not _kanban_worker_scope._kanban_worker_scope_enabled()
        and not _kanban_worker_scope._managed_gateway_dispatch()
    ):
        return []
    claimed = _claimed_worker_scopes_globally()
    if claimed is None:
        return []  # a board was unreadable — reap nothing this tick
    live_untracked = _live_untracked_worker_runs()
    if live_untracked is None:
        return []  # a board was unreadable — reap nothing this tick
    reaped: list[str] = []
    active = _kanban_worker_scope._kanban_list_scope_units("hermes-kanban-*")
    # The restart-safe wrap names its unit ``hermes-worker-kanban-<task>-
    # run-<n>.scope``, and on a systemd-managed gateway that is the unit a
    # worker runs in whenever kanban's own isolation did not wrap the argv
    # (``worker_isolation: none``). Both prefixes are swept, or those units
    # are invisible to the audit and leak.
    active.update(
        _kanban_worker_scope._kanban_list_scope_units("hermes-worker-kanban-*")
    )
    now = time.monotonic()
    # Units needing synchronous work this sweep: terminal-but-loaded
    # units (collect) plus live/deactivating orphans (probe + stop).
    work: list[tuple[str, str]] = []
    for unit, state in active.items():
        if _kanban_worker_scope._kanban_scope_is_live(state) or state == "deactivating":
            if unit in claimed:
                continue  # a running task owns it
            owner = _untracked_owner_of_scope_unit(unit, live_untracked)
            if owner is not None:
                # A live worker whose row records no scope — the shape a
                # worker spawned by the pre-fix build on a managed
                # gateway has after this branch is deployed under it.
                # Reaping it here would kill a running worker on the
                # first tick after the upgrade, so adopt it instead and
                # record the unit the run is actually in.
                board, task_id, run_id = owner
                _log.info(
                    "kanban: adopted unregistered restart-safe unit %s "
                    "for live task %s (run %s) — recording it as the "
                    "run's worker scope instead of reaping it",
                    unit,
                    task_id,
                    run_id,
                )
                _record_adopted_worker_scope(board, task_id, run_id, unit)
                continue
        work.append((unit, state))
    # Age bookkeeping: forget vanished units, stamp first-seen on new
    # ones, then order oldest first (unit name as the deterministic
    # tiebreak among same-tick discoveries).
    for unit in list(_scope_audit_first_seen):
        if unit not in active:
            _scope_audit_first_seen.pop(unit, None)
    for unit, _state in work:
        _scope_audit_first_seen.setdefault(unit, now)
    work.sort(
        key=lambda unit_state: (
            _scope_audit_first_seen.get(unit_state[0], now),
            unit_state[0],
        )
    )
    global _scope_audit_cursor
    if len(work) > _SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK:
        # Pass 9 (AI): the window must ROTATE. Oldest-first alone let
        # _SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK permanently wedged
        # units occupy the window every tick, so any newer orphan behind
        # them was never probed. A persistent cursor round-robins over
        # the sorted list instead — the head of the age order keeps its
        # priority only until the cursor passes it, and every orphan is
        # probed within ceil(total / bound) ticks regardless of wedged
        # ones.
        total = len(work)
        start = _scope_audit_cursor % total
        window = [
            work[(start + i) % total]
            for i in range(_SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK)
        ]
        _scope_audit_cursor = (start + _SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK) % total
        _log.info(
            "kanban: scope audit deferring %d unit(s) to the next tick "
            "(per-tick synchronous bound %d under the dispatch lock)",
            total - _SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK,
            _SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK,
        )
    else:
        window = work
        _scope_audit_cursor = 0
    for unit, state in window:
        if (
            not _kanban_worker_scope._kanban_scope_is_live(state)
            and state != "deactivating"
        ):
            # Terminal-but-still-loaded unit (a failed scope stays
            # inspectable without --collect by design): now that nothing
            # runs in it, collect it.  Once per unit — collection makes
            # it vanish from this listing.
            _kanban_worker_scope._collect_kanban_scope(unit)
            continue
        # "deactivating" is NOT terminal: a stop job is draining the
        # cgroup, and one wedged on a stubborn process would sit here
        # forever if the sweep only collected.  It falls through to a
        # (re)requested verified stop, whose SIGKILL escalation is what
        # eventually clears a wedged drain.
        if not _kanban_worker_stop.request_worker_scope_stop(unit, conn=conn):
            continue  # still stopping — retry next tick
        reaped.append(unit)
        _log.info(
            "kanban: reaped orphaned worker scope %s (no running task claims it)",
            unit,
        )
    return reaped


def clear_dead_worker_scopes_on_nonrunning_tasks(
    conn: sqlite3.Connection,
) -> list[str]:
    """Clear ``worker_scope`` on non-running rows whose unit is verified dead.

    The row-side counterpart of :func:`reap_orphaned_worker_scopes`
    (which stops the cgroup): the drain-ceiling breaker (pass 11, AO)
    blocks a task with its scope deliberately retained for the operator,
    and rows like that are never spawnable while the pointer remains —
    ``recompute_ready`` skips them and ``unblock_task`` refuses them
    (pass 12, AQ). Once the audit's verified stop lands and the unit
    confirms dead, this sweep drops the stale pointer so the row becomes
    spawnable again.

    It runs BEFORE promotion in the dispatch tick, so a scope that died
    since the last tick is cleared and its row promoted in the same
    tick. Like the audit, each row costs a synchronous bus probe under
    the dispatch lock, so the sweep is bounded per tick
    (``_SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK``); anything behind the
    bound — or any unit not yet provably dead — waits for the next tick.

    Returns the task ids whose scope was cleared. Safe to call every
    tick; read-only rows (live/unknown/unsupported units) are untouched.
    """
    rows = conn.execute(
        "SELECT id, worker_scope FROM tasks "
        "WHERE status != 'running' AND worker_scope IS NOT NULL "
        "ORDER BY created_at ASC, id ASC LIMIT ?",
        (_SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK,),
    ).fetchall()
    cleared: list[str] = []
    for row in rows:
        unit = row["worker_scope"]
        if _kanban_worker_scope._kanban_scope_state(unit) != "dead":
            continue  # live, draining, or unverifiable — the audit keeps
            # requesting the verified stop; clear it once it confirms
        with _kanban_db_connect.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET worker_scope = NULL "
                "WHERE id = ? AND status != 'running' AND worker_scope IS ?",
                (row["id"], unit),
            )
            if cur.rowcount != 1:
                continue
            _kanban_db._append_event(
                conn,
                row["id"],
                "worker_scope_cleared",
                {"scope": unit, "reason": "scope verified dead"},
            )
        cleared.append(row["id"])
        _log.info(
            "kanban: cleared dead worker scope %s from task %s",
            unit,
            row["id"],
        )
    return cleared


def stop_all_scoped_workers(
    conn: sqlite3.Connection,
    *,
    should_abort: "Optional[Callable[[], bool]]" = None,
    cancel_event: Optional[threading.Event] = None,
    deadline: Optional[float] = None,
) -> list[str]:
    """Stop every scoped worker claimed by this host.

    The ``kanban.worker_isolation_stop_on_shutdown`` shutdown policy:
    when true, a graceful gateway shutdown verifies teardown of each
    still-running isolated worker instead of leaving it to re-adoption
    (the default). Scoped workers are in their own user systemd scopes
    so they survive a gateway crash by design; this is the explicit
    "bring them down with us" switch.

    ``should_abort`` is checked between units: when it returns True
    (shutdown budget expired, caller cancelled) the loop stands down
    immediately — the remaining units stay for the next gateway's
    re-adoption sweep instead of being stopped after the caller has
    already moved on (Gate B pass 4, finding Q). ``cancel_event`` /
    ``deadline`` (pass 8, Y) extend the same cancellation INTO an
    in-flight verified stop: the escalation steps themselves check them,
    so a wedged TERM wait or SIGKILL drain is abandoned mid-unit, not
    merely after the unit finishes.

    Returns the list of scope units confirmed stopped. Units that could
    not be confirmed remain for the next gateway's re-adoption sweep.
    """
    stopped: list[str] = []
    host_prefix = f"{_kanban_db._claimer_id().split(':', 1)[0]}:"
    rows = conn.execute(
        "SELECT id, claim_lock, worker_scope FROM tasks "
        "WHERE status = 'running' AND worker_scope IS NOT NULL"
    ).fetchall()
    for row in rows:
        if should_abort is not None and should_abort():
            _log.info(
                "kanban shutdown: scoped-worker stop stood down before "
                "%s — budget expired or cancelled",
                row["worker_scope"],
            )
            break
        if not (row["claim_lock"] or "").startswith(host_prefix):
            continue  # another host owns this worker
        if not _kanban_worker_scope._stop_kanban_worker_scope(
            row["worker_scope"],
            cancel_event=cancel_event,
            deadline=deadline,
        ):
            continue  # unconfirmed stop — re-adoption will catch it
        stopped.append(row["worker_scope"])
        _log.info(
            "kanban shutdown: stopped scoped worker %s (task %s)",
            row["worker_scope"],
            row["id"],
        )
    return stopped


def adopt_surviving_running_workers(conn: sqlite3.Connection) -> list[str]:
    """Re-adopt running workers that outlived their dispatcher.

    The scenario (production, 2026-09-01): the gateway restarts — watchdog
    trip, operator restart, OOM — and every kanban worker it spawned keeps
    running in its own scope. The new gateway sees claim_locks owned by
    the OLD gateway pid (``host:<old_pid>``) and previously had no move
    but to let ``detect_crashed_workers`` reclaim the rows, killing the
    runs' progress. With worker isolation those workers are genuine
    survivors: pid verified alive (PID-reuse guard), heartbeat fresh.

    Adoption rewrites claim_lock to THIS dispatcher and re-arms the claim
    TTL, so the run continues seamlessly and the worker's eventual
    ``kanban_complete`` / ``kanban_block`` lands normally — those prove
    ownership via ``expected_run_id``, not claim_lock. The worker's own
    ``heartbeat_claim`` extension stops matching after adoption; that
    failure is swallowed at debug level in the worker (it never kills the
    run) and the dispatcher-side pid-alive branch of
    ``release_stale_claims`` keeps the adopted claim extended (cost: one
    ``claim_extended`` event per claim TTL per adopted run).

    Rows whose pid is dead / pid-reused, or whose heartbeat is staler
    than ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS``, are NOT adopted —
    the existing crash / stale paths still classify and count them. Safe
    to call every tick: once adopted, claim_lock equals this claimer and
    the row stops matching. The singleton dispatcher lock guarantees only
    one dispatcher per host ever runs this pass.
    """
    now = int(time.time())
    current = _kanban_db._claimer_id()
    host_prefix = f"{current.split(':', 1)[0]}:"
    rows = conn.execute(
        "SELECT * "
        "FROM tasks "
        "WHERE status = 'running' AND claim_lock IS NOT NULL "
        "  AND worker_pid IS NOT NULL"
    ).fetchall()
    adopted: list[str] = []
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            # Another host's worker — its own dispatcher owns recovery.
            continue
        if lock == current:
            continue  # Already ours (claimed or adopted on a prior tick).
        if row["reclaim_reserved_at"] is not None:
            # Never cancel another controller's live stop reservation. After
            # restart, recover only an expired orphan whose controller is gone.
            if row["claim_expires"] is None or int(row["claim_expires"]) >= now:
                continue
            try:
                import psutil

                owner_pid = int(lock.rsplit(":", 1)[1])
                if (
                    owner_pid <= 0
                    or psutil.Process(owner_pid).status() != psutil.STATUS_ZOMBIE
                ):
                    continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (ValueError, IndexError, psutil.Error):
                continue
        pid = int(row["worker_pid"])
        # Authoritative liveness: scope cgroup state for isolated runs,
        # pid+fingerprint otherwise — never bare PID liveness (a
        # recycled PID would adopt a stranger's process; the dead
        # launcher of a scoped run would hide its live worker).
        alive, alive_reason = _kanban_worker_identity._run_worker_alive(row)
        if not alive:
            # Dead / pid-reused / scope gone — detect_crashed_workers
            # and the registration-grace sweep handle it.
            continue
        hb = row["last_heartbeat_at"]
        if (
            hb is None
            or (now - int(hb)) > _kanban_db.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        ):
            # Not demonstrably progressing — release_stale_claims /
            # detect_stale_running handle it.
            continue
        new_expires = now + _kanban_db._resolve_claim_ttl_seconds()
        with _kanban_db_connect.write_txn(conn):
            # A deferred worker handoff or verified-stop service owns this
            # attempt until its terminal intent is settled, even after restart.
            pending = conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND run_id IS ? "
                "AND kind IN ('own_worker_handoff', 'scope_stopping') "
                "UNION ALL SELECT 1 FROM task_runs WHERE id IS ? AND stop_pending = 1 LIMIT 1",
                (row["id"], row["current_run_id"], row["current_run_id"]),
            ).fetchone()
            if pending is not None:
                continue
            predicate, values = _kanban_claims._snapshot_predicate(row)
            cur = conn.execute(
                "UPDATE tasks SET claim_lock = ?, claim_expires = ?, reclaim_reserved_at = NULL "
                "WHERE id = ? AND " + predicate,
                (current, new_expires, row["id"], *values),
            )
            if cur.rowcount != 1:
                continue
            run_id = row["current_run_id"]
            if run_id is not None:
                conn.execute(
                    "UPDATE task_runs SET claim_lock = ?, claim_expires = ? "
                    "WHERE id = ? AND claim_lock IS ?",
                    (current, new_expires, run_id, lock),
                )
            _kanban_db._append_event(
                conn,
                row["id"],
                "adopted",
                {
                    "previous_claimer": lock,
                    "claimer": current,
                    "worker_pid": pid,
                    "scope": row["worker_scope"],
                    "verified_by": alive_reason,
                    "heartbeat_age_seconds": now - int(hb),
                },
                run_id=run_id,
            )
            adopted.append(row["id"])
        _log.info(
            "kanban: adopted surviving worker pid %s for task %s "
            "(previous claimer %s, heartbeat %ss old, verified by %s) "
            "— run continues",
            pid,
            row["id"],
            lock,
            now - int(hb),
            alive_reason,
        )
    return adopted


def _classify_crashed_workers(conn, now):
    """Snapshot candidates; all external stop work happens after this read."""
    pending_rows = []
    with _kanban_db_connect.write_txn(conn):
        rows = conn.execute(
            "SELECT t.*,COALESCE(r.started_at,t.started_at) AS active_started_at "
            "FROM tasks t LEFT JOIN task_runs r ON r.id=t.current_run_id "
            "WHERE t.status = 'running' AND t.worker_pid IS NOT NULL"
        ).fetchall()
        host_prefix = f"{_kanban_db._claimer_id().split(':', 1)[0]}:"
        for row in rows:
            # Only check liveness for claims owned by this host.
            lock = row["claim_lock"] or ""
            if not lock.startswith(host_prefix):
                continue
            # Skip liveness check inside the launch-window grace period
            # so a freshly-spawned worker isn't reclaimed before its PID
            # is visible on /proc.
            started_at = row["active_started_at"]
            if started_at is not None:
                grace = _kanban_db._resolve_crash_grace_seconds()
                if time.time() - started_at < grace:
                    continue
            # Scoped runs that have not yet self-registered are still
            # "starting": the recorded pid is the systemd-run launcher
            # (which dies with the gateway), so PID reasoning about it is
            # meaningless. fail_unregistered_workers owns the past-grace
            # classification; inside the grace the row is skipped.
            if (
                row["worker_scope"]
                and row["worker_registered_at"] is None
                and started_at is not None
                and (now - int(started_at))
                < _kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS
            ):
                continue
            # Authoritative liveness — scope cgroup state for isolated
            # runs (a recycled pid must not be mistaken for the worker,
            # and a dead launcher pid must not hide a live scoped
            # worker), pid+fingerprint for registered/unscoped runs,
            # bare liveness only for legacy rows (see
            # ``_run_worker_alive``).
            alive, alive_reason = _kanban_worker_identity._run_worker_alive(row)
            if alive:
                continue

            pid = int(row["worker_pid"])
            pid_reused = _kanban_db_dispatch._pid_alive(pid)
            kind, code = _kanban_db_dispatch._classify_worker_exit(pid)
            rate_limited_exit = False
            if kind == "clean_exit":
                # Worker subprocess returned 0 but its task is still
                # ``running`` in the DB — it exited without calling
                # ``kanban_complete`` / ``kanban_block``. Overwhelmingly the
                # work itself succeeded and only the paperwork was skipped, so
                # a retry usually completes; the corrective sentence below is
                # surfaced to the retry worker via the prior-attempt error in
                # ``build_worker_context`` (guidance approach from #61817).
                protocol_violation = True
                error_text = (
                    "worker exited cleanly (rc=0) without calling "
                    "kanban_complete or kanban_block — protocol violation. "
                    "If the prior run already did the work, verify it and "
                    "report the result via kanban_complete; a run that ends "
                    "without a terminal kanban call counts as failed no "
                    "matter what it did."
                )
                event_kind = "protocol_violation"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                    # Durable marker for _protocol_violation_streak: _end_run
                    # copies this payload into the run metadata, which is how
                    # the violation-only retry budget is derived later.
                    "protocol_violation": True,
                }
            elif kind == "rate_limited":
                # Worker bailed because the provider rate-limited / exhausted
                # quota (EX_TEMPFAIL sentinel). This is NOT a task failure —
                # the task is fine, the account just hit a wall. Release it
                # back to its source phase so the respawn guard defers it until the
                # quota window clears, and crucially do NOT count a failure
                # (skip ``_record_task_failure``) so a long quota window can't
                # trip the circuit breaker and permanently block the card.
                protocol_violation = False
                rate_limited_exit = True
                error_text = (
                    f"pid {pid} exited rate-limited (quota wall) — "
                    f"requeued without counting a failure"
                )
                event_kind = "rate_limited"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                }
            else:
                protocol_violation = False
                if kind == "nonzero_exit":
                    error_text = f"pid {pid} exited with code {code}"
                elif kind == "signaled":
                    error_text = f"pid {pid} killed by signal {code}"
                else:
                    error_text = f"pid {pid} not alive"
                    if pid_reused:
                        error_text += " (pid reused by another process)"
                    if alive_reason == "scope_dead":
                        error_text += " (scope cgroup empty)"
                event_kind = "crashed"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "liveness": alive_reason,
                }
                if pid_reused:
                    event_payload["pid_reused"] = True
                if code is not None and kind != "unknown":
                    event_payload["exit_kind"] = kind
                    event_payload["exit_code"] = code

            retry_status = _owner_kanban_claims._retry_status_for_run(conn, row["id"])
            event_payload["retry_status"] = retry_status
            pending_rows.append({
                **dict(row),
                "id": row["id"],
                "pid": pid,
                "claim_lock": row["claim_lock"],
                "assignee": row["assignee"],
                "worker_scope": row["worker_scope"],
                "current_run_id": row["current_run_id"],
                "kind": kind,
                "code": code,
                "error_text": error_text,
                "event_kind": event_kind,
                "event_payload": event_payload,
                "protocol_violation": protocol_violation,
                "rate_limited_exit": rate_limited_exit,
                "retry_status": retry_status,
            })

    return pending_rows


def detect_crashed_workers(conn: sqlite3.Connection) -> list[str]:
    """Confirm attempt-scoped death, then atomically settle its retry budget."""
    now = int(time.time())
    candidates = _classify_crashed_workers(conn, now)
    closed = []
    for entry in candidates:
        # Reserve exact generation before an external stop. A newly registered
        # worker or heartbeat invalidates the snapshot and prevents the signal.
        reserved = _kanban_claims.reserve_reclaim(conn, entry["id"], entry, now=now)
        if reserved is None:
            continue
        scope = entry["worker_scope"]
        if scope and not _kanban_worker_stop.request_worker_scope_stop(
            scope, task_id=entry["id"], conn=conn
        ):
            _kanban_worker_handoff._mark_run_scope_stopping(
                conn, entry["id"], scope, expected_run_id=entry["current_run_id"]
            )
            continue
        if scope and _kanban_worker_handoff._apply_pending_own_worker_handoff(
            conn, entry["id"], entry["current_run_id"]
        ):
            continue
        closed.append(entry)
    fingerprints = {}
    for entry in closed:
        fp = _kanban_db_dispatch._error_fingerprint(entry["error_text"])
        fingerprints[fp] = fingerprints.get(fp, 0) + 1
    crashed = []
    rate_limited = []
    auto_blocked = []
    hooks = []
    for entry in closed:
        with _kanban_db_connect.write_txn(conn, allow_nested=True):
            if not _settle_crash(conn, entry, fingerprints):
                continue
            if entry["rate_limited_exit"]:
                rate_limited.append(entry["id"])
            else:
                crashed.append(entry["id"])
            status = conn.execute(
                "SELECT status FROM tasks WHERE id=?", (entry["id"],)
            ).fetchone()
            if status and status["status"] == "blocked":
                auto_blocked.append(entry["id"])
            hooks.append(entry)
    detect_crashed_workers._last_auto_blocked = auto_blocked
    detect_crashed_workers._last_rate_limited = rate_limited
    if hooks and _kanban_db._kanban_observer_consumed("on_kanban_worker_exited"):
        for entry in hooks:
            _kanban_db._fire_kanban_lifecycle_hook(
                "on_kanban_worker_exited",
                entry["id"],
                board=_owner_kanban_boards.get_current_board(),
                assignee=entry["assignee"],
                run_id=entry["current_run_id"],
                worker_pid=entry["pid"],
                exit_kind=entry["kind"],
                exit_code=entry["code"],
                outcome="rate_limited" if entry["rate_limited_exit"] else "crashed",
                retry_status=entry["retry_status"],
            )
    return crashed


def _settle_crash(conn, entry, fingerprints):
    """Caller owns the transaction including phase publication and breaker."""
    retry = entry["retry_status"]
    cur = conn.execute(
        "UPDATE tasks SET status=?,claim_lock=NULL,claim_expires=NULL,worker_pid=NULL,"
        "worker_pid_started_at=NULL,worker_registered_at=NULL,worker_scope=NULL,"
        "last_heartbeat_at=NULL,reclaim_reserved_at=NULL "
        "WHERE id=? AND status='running' AND current_run_id IS ? AND claim_lock IS ? "
        "AND worker_pid IS ? AND worker_pid_started_at IS ? AND worker_scope IS ?",
        (
            retry,
            entry["id"],
            entry["current_run_id"],
            entry["claim_lock"],
            entry["pid"],
            entry["worker_pid_started_at"],
            entry["worker_scope"],
        ),
    )
    if cur.rowcount != 1:
        return False
    outcome = "rate_limited" if entry["rate_limited_exit"] else "crashed"
    run_id = _kanban_db._end_run(
        conn,
        entry["id"],
        outcome=outcome,
        status=outcome,
        error=entry["error_text"],
        metadata=dict(entry["event_payload"]),
    )
    _kanban_db._append_event(
        conn, entry["id"], entry["event_kind"], entry["event_payload"], run_id=run_id
    )
    conn.execute(
        "UPDATE tasks SET last_failure_error=? WHERE id=?",
        (entry["error_text"][:500], entry["id"]),
    )
    if entry["rate_limited_exit"]:
        return True
    kwargs = {}
    if entry["protocol_violation"]:
        streak = _kanban_db_dispatch._protocol_violation_streak(conn, entry["id"])
        row = conn.execute(
            "SELECT max_retries FROM tasks WHERE id=?", (entry["id"],)
        ).fetchone()
        limit = (
            int(row["max_retries"])
            if row["max_retries"] is not None
            else _kanban_db_dispatch._PROTOCOL_VIOLATION_FAILURE_LIMIT
        )
        if streak < limit:
            return True
        kwargs = dict(
            failure_limit=limit,
            force_trip=True,
            event_payload_extra={
                "protocol_violations": streak,
                "protocol_violation_limit": limit,
            },
        )
    else:
        fp = _kanban_db_dispatch._error_fingerprint(entry["error_text"])
        if fingerprints.get(fp, 0) >= 3:
            kwargs["failure_limit"] = 1
    _kanban_db_dispatch._record_task_failure(
        conn, entry["id"], entry["error_text"], outcome="crashed", **kwargs
    )
    return True


from hermes_cli import kanban_db as _kanban_db
from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_worker_handoff as _kanban_worker_handoff
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_stop as _kanban_worker_stop


def _stop_reserved_worker(conn, row, *, signal_fn=None):
    return _kanban_worker_identity._terminate_reclaimed_worker(
        row["worker_pid"],
        row["claim_lock"],
        signal_fn=signal_fn,
        scope_unit=row["worker_scope"],
        pid_started_at=row["worker_pid_started_at"],
        task_id=row["id"],
        run_id=row["current_run_id"],
        expected_db=_kanban_claims._reclaim_db_path(conn),
    )


def _settle_recovered_run(conn, row, outcome, error, payload, *, count_failure=False):
    """Publish the retry, run outcome and failure budget in one transaction."""
    with _kanban_db_connect.write_txn(conn, allow_nested=True):
        retry = _kanban_claims._retry_status_for_run(
            conn, row["id"], row["current_run_id"]
        )
        cur = conn.execute(
            "UPDATE tasks SET status=?,claim_lock=NULL,claim_expires=NULL,worker_pid=NULL,"
            "worker_pid_started_at=NULL,worker_registered_at=NULL,worker_scope=NULL,"
            "last_heartbeat_at=NULL,reclaim_reserved_at=NULL "
            "WHERE id=? AND status='running' AND current_run_id IS ? "
            "AND claim_lock IS ? AND worker_pid IS ? AND worker_pid_started_at IS ? "
            "AND worker_scope IS ?",
            (
                retry,
                row["id"],
                row["current_run_id"],
                row["claim_lock"],
                row["worker_pid"],
                row["worker_pid_started_at"],
                row["worker_scope"],
            ),
        )
        if cur.rowcount != 1:
            return False
        payload = dict(payload, retry_status=retry)
        run_id = _kanban_db._end_run(
            conn,
            row["id"],
            outcome=outcome,
            status=outcome,
            error=error,
            metadata=payload,
        )
        _kanban_db._append_event(conn, row["id"], outcome, payload, run_id=run_id)
        if count_failure:
            _kanban_db_dispatch._record_task_failure(
                conn, row["id"], error, outcome=outcome, event_payload_extra=payload
            )
        return True


from hermes_cli import kanban_claims as _kanban_claims
