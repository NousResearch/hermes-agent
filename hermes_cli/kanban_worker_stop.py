"""Kanban worker lifecycle; integrated from PR #101911 by danashburn.

Attempt ownership and persistence use the current main Kanban modules.
"""

from __future__ import annotations
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


SCOPE_STOP_SERVICE_THREAD_NAME = "kanban-scope-stop"


_scope_stop_inline = False


_SCOPE_STOP_CANCEL_DRAIN_WAIT_SECONDS = 5.0


@dataclass
class _ScopeStopRequest:
    unit: str
    task_id: Optional[str] = None
    attempts: int = 0
    # Set by the unregistered-launch sweep (finding J): a stop queued on
    # "this worker never registered" must re-check that diagnosis
    # immediately before acting, because the worker can register while
    # the request sits in this queue. Terminal-path stops leave it False:
    # a completed task's scope MUST be reaped regardless of registration.
    skip_if_registered: bool = False


_scope_stop_lock = threading.Lock()


_scope_stop_pending: dict[str, _ScopeStopRequest] = {}


_scope_stop_inflight: Optional[str] = None


_scope_stop_confirmed: set[str] = set()


_scope_stop_attempts: dict[str, int] = {}


_scope_stop_warned: set[str] = set()


_scope_stop_wake = threading.Event()


_scope_stop_thread: Optional[threading.Thread] = None


_scope_stop_service_cancel = threading.Event()


_scope_stop_service_stopping = threading.Event()


def _scope_stop_service_loop() -> None:
    while not _scope_stop_service_stopping.is_set():
        _scope_stop_wake.wait()
        _scope_stop_wake.clear()
        if _scope_stop_service_stopping.is_set():
            return
        _drain_scope_stop_requests()


def _drain_scope_stop_requests() -> None:
    """Run every queued verified stop (service thread; inline in tests).

    A cancelled drain (pass 9, AH — a join whose shutdown budget expired
    set the cancel token) stops before the next unit and requeues the
    one it was working on, so nothing is signalled after the caller has
    moved on and the queue survives for re-adoption. The drain
    snapshots the CURRENT cancel token at entry (pass 10, AL): a join
    that expires mid-drain sets the token this drain holds and swaps a
    fresh one into the global, so THIS drain stands down while the NEXT
    drain (a request enqueued after the join) runs clean — cancellation
    is per join/run, never a latch the service never recovers from.
    """
    global _scope_stop_inflight
    with _scope_stop_lock:
        cancel_token = _scope_stop_service_cancel
    while True:
        if cancel_token.is_set():
            return
        with _scope_stop_lock:
            if not _scope_stop_pending:
                return
            unit = next(iter(_scope_stop_pending))
            request = _scope_stop_pending.pop(unit)
            _scope_stop_inflight = unit
        try:
            if (
                request.skip_if_registered
                and request.task_id
                and _kanban_worker_recovery._task_has_registered_worker(
                    request.task_id, expected_scope=unit
                )
            ):
                # Finding J: the worker registered while this stop sat in
                # the queue — the "never launched" diagnosis is stale.
                # Killing it now would execute a death sentence the row
                # already repudiated; drop the request and leave the run
                # to adoption / crash detection, which see a live worker.
                _log.info(
                    "kanban: skipping queued stop of scope %s — task %s "
                    "registered a worker while the stop was queued",
                    unit,
                    request.task_id,
                )
                continue
            if (
                request.skip_if_registered
                and request.task_id
                and not _kanban_worker_recovery._mark_run_stop_pending(
                    request.task_id, expected_scope=unit
                )
            ):
                # Gate B pass 4 (R): the re-check above is a plain read —
                # registration can still commit after it and before the
                # signal below. The stop-pending CAS closes that window
                # atomically: it marks the run only while the task is
                # still running and unregistered, and a registration for
                # a marked run self-aborts inside register_worker_pid.
                # A CAS miss therefore means the registration won this
                # instant — stand the stop down exactly like the J skip.
                # It also misses when no board is writable; standing
                # down then is the safe side (next ticks re-verify).
                _log.info(
                    "kanban: standing down queued stop of scope %s — task "
                    "%s won the stop-pending CAS (or no board was "
                    "writable)",
                    unit,
                    request.task_id,
                )
                continue
            verified = _kanban_worker_scope._stop_kanban_worker_scope(
                unit,
                cancel_event=cancel_token,
            )
        finally:
            with _scope_stop_lock:
                _scope_stop_inflight = None
        with _scope_stop_lock:
            if verified:
                _scope_stop_confirmed.add(unit)
                _scope_stop_attempts.pop(unit, None)
            else:
                _scope_stop_attempts[unit] = request.attempts + 1
                if unit not in _scope_stop_warned:
                    _scope_stop_warned.add(unit)
                    _log.warning(
                        "kanban: verified stop of worker scope %s did not "
                        "confirm (task %s, attempt %d) — keeping claims "
                        "and retrying on later ticks",
                        unit,
                        request.task_id,
                        request.attempts + 1,
                    )
                # Left unconfirmed: the next request re-enqueues it.
        if not verified and cancel_token.is_set():
            # Pass 9 (AH): the stop was cancelled mid-flight — put the
            # unit back so the queue reflects what is still stopping
            # (the join that cancelled reports it), and stand the drain
            # down rather than starting another unit the caller cannot
            # wait for.
            with _scope_stop_lock:
                if unit not in _scope_stop_pending:
                    _scope_stop_pending[unit] = _ScopeStopRequest(
                        unit=unit,
                        task_id=request.task_id,
                        attempts=_scope_stop_attempts.get(
                            unit,
                            request.attempts + 1,
                        ),
                        skip_if_registered=request.skip_if_registered,
                    )
            return
        if verified and request.skip_if_registered and request.task_id:
            # Pass 8 (AC): this drain CAS-marked the run stop-pending right
            # before signalling; the cgroup is now verified empty, so the
            # marker has done its job and must not outlive the stop — a
            # re-adopted run with a stale marker would refuse its worker's
            # registration forever.
            _kanban_worker_recovery._clear_run_stop_pending(
                request.task_id, expected_scope=unit
            )


def _ensure_scope_stop_thread() -> None:
    global _scope_stop_thread
    if _scope_stop_thread is not None and _scope_stop_thread.is_alive():
        return
    _scope_stop_service_stopping.clear()
    _scope_stop_thread = threading.Thread(
        target=_scope_stop_service_loop,
        name=SCOPE_STOP_SERVICE_THREAD_NAME,
        daemon=True,
    )
    _scope_stop_thread.start()


def request_worker_scope_stop(
    unit_name: Optional[str],
    *,
    task_id: Optional[str] = None,
    skip_if_registered: bool = False,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Hand a worker scope to the verified-stop service. True = confirmed
    dead, the caller may release its bookkeeping THIS tick.

    Fast path: a unit whose cgroup is already empty (the common case —
    the worker died and nothing double-forked) confirms after one cheap
    liveness probe. Otherwise (live pids, a stop job still draining, or
    a probe failure) the unit is queued on the background service and
    this returns False = "stopping": keep the claim, requeue/clear
    nothing, retry next tick — releasing beside an unverified cgroup is
    what let the dispatcher spawn duplicates. Never blocks longer than
    one liveness probe; the stop+escalate+verify sequence runs on the
    service thread.

    ``skip_if_registered`` (finding J) marks the request as
    evidence-based, not terminal: immediately before the service acts it
    re-checks the task row and stands down when the worker has
    registered in the meantime, and atomically CAS-marks the run
    ``stop_pending`` so a registration landing inside the remaining
    window self-aborts instead of dying under the signal (pass 4, R).
    Only the unregistered-launch sweep sets it — a completed task's
    scope must be reaped regardless.

    Called with the ``conn`` of an open :func:`write_txn`, the request
    becomes commit-conditional (finding O): it is collected as an intent
    and queued only after the OUTERMOST transaction on that connection
    commits, discarded on rollback — the queue entry is process state,
    not a DB row, so without this a rolled-back demotion would still
    kill its worker. The stack is keyed by CONNECTION, not thread
    (pass 8, finding AA): a caller on a shared
    ``check_same_thread=False`` connection must pass ``conn`` so a
    request from ANY thread inside that transaction is collected, and
    threads using other connections never see it. Without ``conn`` the
    request always queues immediately (legacy paths that by construction
    run outside any transaction).
    """
    if not unit_name:
        return False
    with _scope_stop_lock:
        if unit_name in _scope_stop_confirmed:
            return True
    state = _kanban_worker_scope._kanban_scope_state(unit_name)
    if state == "dead":
        with _scope_stop_lock:
            _scope_stop_confirmed.add(unit_name)
        # Confirmed dead = terminal: collect the (possibly still loaded,
        # failed) unit so it does not linger on the bus.  The verified
        # stop path collects on its own; this fast path bypasses it.
        _kanban_worker_scope._collect_kanban_scope(unit_name)
        return True
    if state != "active":
        # Probe failed ("unknown") or the host has no readable cgroup
        # hierarchy ("unsupported"): death can be neither confirmed nor
        # ruled out here. Queue a stop attempt anyway — the stop/kill
        # still fires over the bus (the only kill path there is), and on
        # unsupported hosts the service confirms via the unit's bus
        # state instead of cgroup.procs. An unreachable bus makes the
        # background pass fail cheaply and the tick retries.
        _log.debug(
            "kanban: scope %s state %s (task %s) — queueing a verified stop attempt",
            unit_name,
            state,
            task_id,
        )
    if conn is not None and _kanban_db_connect._collect_scope_stop_intent(
        conn,
        unit_name,
        task_id,
        skip_if_registered,
    ):
        # Commit-conditional (Gate B pass 4, finding O): *conn* has an
        # open write transaction — the intent is collected on the
        # innermost savepoint level instead of queueing now. It only
        # reaches the queue if the OUTERMOST transaction commits; any
        # rollback discards it, so a demotion that was rolled back can
        # never leave its worker killed. Same return contract as
        # queueing: not confirmed, the caller keeps its hold and retries
        # next tick.
        return False
    with _scope_stop_lock:
        existing = _scope_stop_pending.get(unit_name)
        if existing is not None:
            # Pass 8, finding AB: coalescing must not let whichever
            # request arrived LAST decide whether the stop can be
            # skipped. A terminal request (skip_if_registered=False — the
            # completed task's scope MUST be reaped) and an
            # unregistered-launch request (True) for the same unit
            # compose with AND: once a terminal stop is queued, no later
            # True can re-enable skipping past it, in either arrival
            # order.
            existing.skip_if_registered = (
                existing.skip_if_registered and skip_if_registered
            )
            if task_id and not existing.task_id:
                existing.task_id = task_id
            existing.attempts = _scope_stop_attempts.get(unit_name, 0)
        else:
            _scope_stop_pending[unit_name] = _ScopeStopRequest(
                unit=unit_name,
                task_id=task_id,
                attempts=_scope_stop_attempts.get(unit_name, 0),
                skip_if_registered=skip_if_registered,
            )
    if _scope_stop_inline:
        _drain_scope_stop_requests()
        with _scope_stop_lock:
            return unit_name in _scope_stop_confirmed
    _ensure_scope_stop_thread()
    _scope_stop_wake.set()
    return False


def join_scope_stop_service(
    timeout: float,
    *,
    cancel_event: Optional[threading.Event] = None,
) -> list[str]:
    """Wait for the background service to DRAIN (shutdown path). Returns
    the units still pending or in flight once the budget expires, so the
    caller can log exactly what it leaves to the next gateway's adoption
    sweep.

    The service thread runs for the life of the process, so "joined"
    means DRAINED, not thread-exit: a plain ``Thread.join`` would always
    burn the full timeout even with nothing left to stop. The wait ends
    the moment the queue is empty AND no stop is mid-flight.

    On budget expiry the join CANCELS the in-flight stop (pass 9, AH):
    the current cancel token fires, the service's verified stop
    abandons — its systemctl helper subprocesses are killed, the unit is
    requeued — so the old gateway stops signalling the moment its budget
    is gone instead of a full stop/SIGKILL/verify sequence past the
    dispatcher lock release. The cancellation is per join/run (pass 10,
    AL): setting the token is immediately followed by swapping a fresh,
    unset token into the global — under the same lock the drain
    snapshots with — so the service thread (which keeps running) serves
    any request enqueued after the join instead of latching cancelled
    forever. ``cancel_event`` is the caller's own shutdown event (the
    same one the direct cleanup propagates); it is set alongside so
    both paths share one cancel signal.
    """
    global _scope_stop_service_cancel
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _scope_stop_lock:
            pending = list(_scope_stop_pending)
            inflight = _scope_stop_inflight
        if not pending and inflight is None:
            return []
        if time.monotonic() >= deadline:
            with _scope_stop_lock:
                # Set the token the in-flight drain snapshotted, then
                # swap in a fresh one — atomically w.r.t. the drain's
                # snapshot, so no drain can start on a token that is
                # already set (a latch) nor miss one that should abort
                # it.
                _scope_stop_service_cancel.set()
                _scope_stop_service_cancel = threading.Event()
            if cancel_event is not None:
                cancel_event.set()
            # Pass 10 (AM): do not return while the cancelled stop is
            # still mid-flight. The helper observes cancellation within
            # its <=0.5 s poll and is killed+reaped unconditionally, so
            # this bounded wait sees the drain stand down and the report
            # below lists the queue exactly as the service requeued it.
            inflight_deadline = time.monotonic() + (
                _SCOPE_STOP_CANCEL_DRAIN_WAIT_SECONDS
            )
            while time.monotonic() < inflight_deadline:
                with _scope_stop_lock:
                    if _scope_stop_inflight is None:
                        break
                time.sleep(0.05)
            # Re-snapshot AFTER cancelling so the report lists the unit
            # the service was inside of (it may have requeued itself by
            # now — either way it appears exactly once).
            with _scope_stop_lock:
                pending = list(_scope_stop_pending)
                inflight = _scope_stop_inflight
            if inflight and inflight not in pending:
                pending.append(inflight)
            return pending
        time.sleep(0.05)


def reset_scope_stop_service_for_tests() -> None:
    """Drop all service state AND stop a running service thread.

    Production unit names are unique per attempt so state never needs
    resetting there; tests reuse ids. The thread stop matters as much
    as the state clear: a daemon left alive by an earlier test file
    waits on the same wake event, so any later file that flushes a
    queue without its own thread races that daemon's drain — the
    assertions see an empty queue the moment it fills (pass 8b, found
    in the AE single-process run)."""
    global _scope_stop_inflight, _scope_stop_thread, _scope_stop_service_cancel
    with _scope_stop_lock:
        _scope_stop_pending.clear()
        _scope_stop_confirmed.clear()
        _scope_stop_attempts.clear()
        _scope_stop_warned.clear()
        _scope_stop_inflight = None
        # Swap in a fresh token (pass 10, AL): clearing the old one would
        # un-abort a drain that is mid-stand-down; replacing it starts a
        # clean epoch for the next test's drains.
        _scope_stop_service_cancel = threading.Event()
    _kanban_worker_recovery._scope_audit_first_seen.clear()
    pass
    _kanban_worker_recovery._scope_audit_cursor = 0
    thread = _scope_stop_thread
    _scope_stop_thread = None
    if (
        thread is not None
        and thread.is_alive()
        and thread is not threading.current_thread()
    ):
        _scope_stop_service_stopping.set()
        _scope_stop_wake.set()
        thread.join(timeout=2.0)
        _scope_stop_service_stopping.clear()
    _scope_stop_wake.clear()


from hermes_cli import kanban_db as _kanban_db
from hermes_cli import kanban_worker_recovery as _kanban_worker_recovery
from hermes_cli import kanban_worker_scope as _kanban_worker_scope

from hermes_cli import kanban_db_connect as _kanban_db_connect
