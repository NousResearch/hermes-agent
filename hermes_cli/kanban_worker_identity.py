"""Kanban worker lifecycle; integrated from PR #101911 by danashburn.

Attempt ownership and persistence use the current main Kanban modules.
"""

from __future__ import annotations

import hermes_cli.kanban_db as _owner_kanban_db

import hermes_cli.kanban_db_boards as _owner_kanban_boards
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


def _run_worker_alive(row: Any) -> "tuple[bool, str]":
    """Authoritative liveness of a running row's worker → (alive, reason).

    Decision order (never bare PID liveness when a better signal exists):

    1. Scope truth — when the run has a ``worker_scope`` and the unit's
       state is queryable, that IS the answer: ``active`` means processes
       of this run exist (the worker or descendants it spawned); ``dead``
       means the cgroup is empty, so nothing of this run survives even if
       a recycled PID looks alive.
    2. PID + start-time fingerprint — for registered workers (and for
       unscoped runs), the recorded pid must be alive AND present the
       same creation time; a recycled PID fails the fingerprint.
    3. Launcher pid — scoped runs that have not registered yet ("starting")
       are represented by the systemd-run launcher pid, which is all the
       dispatcher has until the worker self-registers.
    4. Legacy rows (pre-isolation schema: no scope, no fingerprint) fall
       back to bare PID liveness — there is no better signal for them.

    ``reason`` names the branch that decided, for events and logs.
    """
    scope = row["worker_scope"] if "worker_scope" in row.keys() else None
    pid = row["worker_pid"] if "worker_pid" in row.keys() else None
    started_at = (
        row["worker_pid_started_at"] if "worker_pid_started_at" in row.keys() else None
    )
    registered_at = (
        row["worker_registered_at"] if "worker_registered_at" in row.keys() else None
    )
    if scope:
        state = _kanban_worker_scope._kanban_scope_state(scope)
        if state == "active":
            return True, "scope_active"
        if state == "dead":
            return False, "scope_dead"
        # unknown/unsupported: no cgroup truth here (bus unreachable, or
        # a host with no readable cgroup hierarchy at all) — fall through
        # to pid checks, which for unsupported hosts IS the contract.
    if registered_at is not None:
        # Liveness keeps unknown-alive (conservative: hold the claim) —
        # see ``_worker_pid_identity_state`` for why signal gates differ.
        if _worker_pid_identity_state(pid, started_at) != "dead":
            return True, "pid_identity"
        return False, "pid_gone_or_reused"
    if scope:
        # Unregistered scoped run: the recorded pid is the launcher.
        if _worker_pid_identity_state(pid, started_at) != "dead":
            return True, "launcher_alive"
        return False, "launcher_gone"
    if started_at is not None:
        # Unscoped row with a fingerprint but no registered_at (written
        # before that column existed): the fingerprint is still the
        # authority — a recycled pid must not pass.
        if _worker_pid_identity_state(pid, started_at) != "dead":
            return True, "pid_identity"
        return False, "pid_gone_or_reused"
    return _kanban_db_dispatch._pid_alive(pid), "legacy_bare_pid"


def register_worker_pid(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_run_id: Optional[int] = None,
    pid: Optional[int] = None,
) -> bool:
    """Worker-side self-registration: the WORKER records its own pid.

    ``systemd-run --scope`` launches the command as a child of the
    systemd-run client, and that client stays in the gateway's cgroup and
    dies with the gateway — while the scoped worker survives. The pid the
    dispatcher records at spawn is therefore the launcher's, useless for
    adoption or crash detection across a gateway restart. This function
    is called from the worker process itself (the kanban auto-heartbeat
    bridge and the explicit ``kanban_heartbeat`` tool) on first activity,
    and overwrites the launcher pid with the worker's own pid + start-time
    fingerprint, flipping ``worker_registered_at``.

    Deliberately NOT gated on ``claim_lock``: after a gateway restart the
    dispatcher re-adopts the row and rewrites the claim, so the worker's
    pinned env claim lock no longer matches — registration must survive
    adoption. ``expected_run_id`` still pins the attempt, so a stale
    worker from a superseded attempt can never hijack the row.
    """
    real_pid = int(pid) if pid is not None else os.getpid()
    started = _worker_pid_start_time(real_pid)
    now = int(time.time())
    with _kanban_db_connect.write_txn(conn):
        row = conn.execute(
            "SELECT status, current_run_id, worker_pid, "
            "       worker_pid_started_at, worker_registered_at, reclaim_reserved_at "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if (
            row is None
            or row["status"] != "running"
            or row["reclaim_reserved_at"] is not None
        ):
            return False
        if expected_run_id is not None:
            if row["current_run_id"] is None:
                return False
            if int(row["current_run_id"]) != int(expected_run_id):
                return False
        if row["current_run_id"] is not None:
            marked = conn.execute(
                "SELECT 1 FROM task_runs WHERE id = ? AND stop_pending = 1",
                (int(row["current_run_id"]),),
            ).fetchone()
            if marked is not None:
                # Pass 4 (R): the scope-stop service CAS-marked this run
                # stop-pending — a queued registration-sensitive stop is
                # signalling the scope right now. Self-abort instead of
                # registering into a death sentence: the stop's
                # "never launched" evidence stays valid, and the next
                # attempt (a fresh run row) registers cleanly. This read
                # is authoritative because write_txn already holds the
                # database write lock — the marker cannot be set behind
                # our back mid-registration.
                _log.info(
                    "kanban: task %s registration for run %s self-aborted "
                    "— a queued scope stop is signalling this run",
                    task_id,
                    row["current_run_id"],
                )
                return False
        first_registration = row["worker_registered_at"] is None
        if not first_registration and row["worker_pid"] is not None:
            if int(row["worker_pid"]) == real_pid:
                # Same numeric pid: the start-time fingerprint MUST match.
                # A mismatch means the worker died and the kernel handed
                # its pid to an unrelated process (or a stale worker from
                # a previous boot) — reject rather than let it inherit
                # the registration (review finding a).
                if (
                    row["worker_pid_started_at"] is not None
                    and started is not None
                    and int(row["worker_pid_started_at"]) != int(started)
                ):
                    _log.warning(
                        "kanban: task %s re-registration for pid %s with a "
                        "different start fingerprint (recorded %s, new %s) "
                        "— rejected as a reused pid",
                        task_id,
                        real_pid,
                        row["worker_pid_started_at"],
                        started,
                    )
                    return False
            elif (
                _worker_pid_identity_state(
                    row["worker_pid"],
                    row["worker_pid_started_at"],
                )
                != "dead"
            ):
                # A different, still-alive pid already owns this attempt —
                # do not let a stray process overwrite it. Unknown
                # identity also refuses the overwrite (conservative).
                return False
            # A different, DEAD pid: superseded launcher/worker (adoption
            # rewrote the claim, a reclaim killed it) — the new worker
            # may take over the registration.
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_pid_started_at = ?, "
            "worker_registered_at = ? WHERE id = ? AND status = 'running'",
            (real_pid, started, now, task_id),
        )
        run_id = _kanban_db._current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
                (real_pid, run_id),
            )
        if first_registration:
            _kanban_db._append_event(
                conn,
                task_id,
                "worker_registered",
                {"pid": real_pid, "pid_started_at": started},
                run_id=run_id,
            )
    return True


def _worker_pid_start_time(pid: int) -> Optional[int]:
    """Spawn-time PID-reuse fingerprint for a worker (see gateway.status)."""
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(int(pid))
    except Exception:
        return None


_LEGACY_PID_START_TOLERANCE_SECONDS = 15


def _worker_pid_epoch_start(pid: int) -> Optional[float]:
    """Epoch start time of *pid*, or None.

    Unlike :func:`_worker_pid_start_time` — the opaque, per-platform
    fingerprint recorded at spawn — this is a comparable wall-clock
    timestamp (psutil's ``create_time``), so it can be checked against
    run ``started_at`` values.  None when the process is gone or its
    start time cannot be read.
    """
    try:
        import psutil

        return psutil.Process(int(pid)).create_time()
    except Exception:
        return None


def _legacy_pid_belongs_to_run(
    pid: int, run_started_at: Optional[int]
) -> Optional[bool]:
    """Membership gate for legacy rows without a start-time fingerprint.

    A worker is spawned at or after its run row's ``started_at``; a live
    process that was already running before that (outside the tolerance
    above) therefore cannot be the run's worker — it is an unrelated
    process that merely carries a recycled pid number.

    Tri-state (Gate B pass 4, finding P): ``True`` when the pid
    plausibly belongs to the run (the bare-pid kill may proceed);
    ``False`` when it predates the run (never signal it); ``None`` when
    membership could not be determined (no run start, psutil/process
    data unreadable). ``None`` fails SAFE at the call site — no
    bare-pid signal, reclaim without signalling — because a signal we
    cannot attribute might hit an unrelated process.
    """
    if run_started_at is None:
        return None
    started = _worker_pid_epoch_start(pid)
    if started is None:
        return None
    return started >= int(run_started_at) - _LEGACY_PID_START_TOLERANCE_SECONDS


_bare_pid_gate_warned: set[tuple[str, Optional[int], int, str]] = set()


def _warn_bare_pid_gate_stood_down(
    task_id: str,
    run_id: Optional[int],
    pid: int,
    verdict: str,
) -> None:
    key = (task_id, run_id, pid, verdict)
    if key in _bare_pid_gate_warned:
        return
    _bare_pid_gate_warned.add(key)
    reason = {
        "pid_reused": "no longer matches the recorded start fingerprint (recycled pid)",
        "pid_predates_run": "was running before the run began (unrelated recycled pid)",
        "pid_identity_unknown": "could not be verified against the "
        "recorded start fingerprint (process data "
        "unreadable)",
        "pid_membership_unknown": "could not be verified as belonging to "
        "this run (process data unreadable)",
    }.get(verdict, verdict)
    _log.warning(
        "kanban: task %s run %s pid %s %s — not signalling; retaining unresolved ownership",
        task_id,
        run_id,
        pid,
        reason,
    )


def _pid_identity_state(pid: int, started_at: Optional[int]) -> str:
    """Compare a live pid against its recorded start fingerprint.

    Returns ``"gone"`` (not alive), ``"match"`` (same process we
    spawned), ``"different"`` (alive but the fingerprint differs — a
    recycled pid), or ``"unknown"`` (alive, but no fingerprint was
    recorded or the live one could not be read). ``"unknown"`` is
    distinct so signal gates can fail safe on it (finding P) while
    liveness checks keep treating it as alive (no regression).
    """
    if not _kanban_db_dispatch._pid_alive(pid):
        return "gone"
    if started_at is None:
        return "unknown"
    try:
        from gateway.status import get_process_start_time
    except Exception:
        return "unknown"
    live = get_process_start_time(int(pid))
    if live is None:
        return "unknown"
    return "match" if live == int(started_at) else "different"


def _worker_pid_identity_state(pid: int, started_at: Optional[int]) -> str:
    """Tri-state identity verdict for a recorded worker pid (pass 8, X).

    ``"dead"`` — the pid is gone, or alive with a DIFFERENT start
    fingerprint (a recycled pid): nothing of ours to signal or extend.
    ``"alive"`` — the live process presents the recorded fingerprint.
    ``"unknown"`` — alive, but the identity cannot be verified (no
    fingerprint recorded — legacy rows — or the live one is unreadable).

    The tri-state exists so SIGNAL gates can fail safe: a caller about to
    ``kill(pid, ...)`` treats ``"unknown"`` as never-signal (the pid may
    belong to an unrelated process), while LIVENESS callers keep
    ``state != "dead"`` so an unreadable identity stays conservative
    (keep the claim, refuse the overwrite) exactly as before. The old
    boolean helper folded ``"unknown"`` into alive, which made every
    generic reclaim path happily signal an unattributable pid
    (``enforce_max_runtime`` had already fixed its own gate — finding P).
    """
    state = _pid_identity_state(pid, started_at)
    if state in ("gone", "different"):
        return "dead"
    if state == "match":
        return "alive"
    return "unknown"


def _worker_termination_tuple(
    row: Any,
) -> tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    """Uniform termination record for a row being moved off ``running``:
    ``(worker_pid, claim_lock, worker_pid_started_at, worker_scope)``.

    Every status-transition path that collects pending worker
    terminations for a post-commit drain appends exactly this shape, so
    the drain can unpack four fields uniformly. The dashboard's direct
    moves recorded two-tuples and crashed the drain with a ValueError
    (Gate B review, finding 5) — one helper, one shape."""
    keys = row.keys() if hasattr(row, "keys") else ()
    return (
        row["worker_pid"] if "worker_pid" in keys else None,
        row["claim_lock"] if "claim_lock" in keys else None,
        row["worker_pid_started_at"] if "worker_pid_started_at" in keys else None,
        row["worker_scope"] if "worker_scope" in keys else None,
    )


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
    scope_unit: Optional[str] = None,
    pid_started_at: Optional[int] = None,
    task_id: Optional[str] = None,
    run_id: Optional[int] = None,
    expected_db: Optional[str] = None,
) -> dict[str, Any]:
    """Stop the owned scope, or the identified unscoped worker, and verify death.

    An unreadable identity is an inconclusive stop; it never becomes a death
    receipt. Legacy workers can supply their persisted task/run/board identity
    through their process environment. Native process fingerprints work on all
    supported desktop platforms through gateway.status.
    """
    info = dict(
        prev_pid=int(pid) if pid else None,
        host_local=False,
        termination_attempted=False,
        terminated=False,
        sigkill=False,
        execution_scope="systemd_scope" if scope_unit else "worker_process",
    )
    if not claim_lock or not str(claim_lock).startswith(_kanban_db._host_prefix()):
        return info
    info["host_local"] = True
    if scope_unit:
        info.update(scope_unit=scope_unit, termination_attempted=True)
        info["scope_stopped"] = _kanban_worker_stop.request_worker_scope_stop(
            scope_unit, task_id=task_id
        )
        info["terminated"] = info["scope_stopped"]
        # The scope is the ownership boundary even when the launcher PID is
        # absent/recycled. Never substitute PID death for cgroup death.
        return info
    if not pid or pid <= 0:
        return info
    info["termination_attempted"] = True
    if pid_started_at is None:
        pid_started_at = _legacy_worker_fingerprint(pid, task_id, run_id, expected_db)
    state = _worker_pid_identity_state(pid, pid_started_at)
    if state == "dead":
        info["terminated"] = True
        if _kanban_db_dispatch._pid_alive(pid):
            info["pid_reused"] = True
        return info
    if state != "alive":
        info["signal_skipped"] = "pid_identity_unknown"
        _warn_bare_pid_gate_stood_down(task_id, run_id, pid, "pid_identity_unknown")
        return info
    pidfd = None
    try:
        # Pin Linux process identity across TERM/KILL; a recycled number must
        # never receive escalation intended for this worker.
        if (
            signal_fn is None
            and hasattr(os, "pidfd_open")
            and hasattr(signal, "pidfd_send_signal")
        ):
            pidfd = os.pidfd_open(int(pid))
            if _worker_pid_identity_state(pid, pid_started_at) != "alive":
                info["terminated"] = (
                    _worker_pid_identity_state(pid, pid_started_at) == "dead"
                )
                return info

            def kill(_pid, sig):
                signal.pidfd_send_signal(pidfd, sig)

        elif signal_fn is not None:
            kill = signal_fn
        else:
            import psutil

            process = psutil.Process(int(pid))
            if _worker_pid_identity_state(pid, pid_started_at) != "alive":
                info["terminated"] = (
                    _worker_pid_identity_state(pid, pid_started_at) == "dead"
                )
                return info

            def kill(_pid, sig):
                process.send_signal(sig)

        kill(int(pid), signal.SIGTERM)
        for _ in range(10):
            state = _worker_pid_identity_state(pid, pid_started_at)
            if state == "dead":
                info["terminated"] = True
                return info
            if state != "alive":
                info["signal_skipped"] = "pid_identity_unknown"
                return info
            time.sleep(0.5)
        if _worker_pid_identity_state(pid, pid_started_at) == "alive":
            kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
            info["sigkill"] = True
        for _ in range(10):
            if _worker_pid_identity_state(pid, pid_started_at) == "dead":
                info["terminated"] = True
                break
            time.sleep(0.05)
    except ProcessLookupError:
        info["terminated"] = True
    except (OSError, PermissionError):
        pass
    except Exception as exc:
        # psutil uses its own exception hierarchy on Windows/macOS. An
        # access failure is inconclusive; disappearing after identity pinning
        # is a successful stop. Do not let either abandon reclaim bookkeeping.
        import psutil

        if isinstance(exc, psutil.NoSuchProcess):
            info["terminated"] = True
        elif isinstance(exc, psutil.AccessDenied):
            info["signal_skipped"] = "access_denied"
        else:
            raise
    finally:
        if pidfd is not None:
            os.close(pidfd)
    return info


def _worker_survived_termination(termination: dict) -> bool:
    """True when we tried to kill our own host-local worker and it is still alive.

    Reclaiming in this state would release the claim and let the dispatcher
    spawn a second worker while the first is still running — the duplication
    loop. Survival now includes the scope dimension: a scoped worker whose
    unit stop could not be VERIFIED counts as survived even when the
    recorded pid is gone, because the cgroup may still be draining
    descendants (dev servers, browsers) that would overlap the retry.
    Only host-local workers we actually signalled count: a non-local
    claim lock or a no-op attempt (no ``os.kill`` available) must fall
    through to the normal release path, since we cannot manage that
    worker anyway.
    """
    if not (termination.get("termination_attempted") and termination.get("host_local")):
        return False
    if termination.get("scope_unit") and not termination.get("scope_stopped"):
        return True
    return not termination.get("terminated")


def _defer_reclaim_for_live_worker(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    now: int,
    termination: dict,
    *,
    reason: str,
    expected_run_id: Optional[int] = None,
) -> None:
    """Hold a claim whose worker survived termination instead of releasing it.

    Extends ``claim_expires`` by ``RECLAIM_DEFER_GRACE_SECONDS`` so the task
    stays ``running`` (no duplicate spawn) and records a ``reclaim_deferred``
    event so the hold is visible in ``hermes kanban tail``. The next dispatch
    tick retries the kill; this is self-correcting because not spawning a
    duplicate is what lets the throttled worker finally die. Any reclaim
    reservation marker is dropped too (pass 12, AP): the reservation has
    concluded — the signal fired — so a heartbeat landing during the defer
    grace extends the claim exactly as it did before reservations existed.
    """
    grace = now + _kanban_db.RECLAIM_DEFER_GRACE_SECONDS
    # Nested-safe: composed under the dashboard's outer commit by the
    # ancestor-reopen invalidation's Phase 0 (savepoint there, own txn
    # everywhere else).
    with _kanban_db_connect.write_txn(conn, allow_nested=True):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ?, reclaim_reserved_at = NULL "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
            "AND (? IS NULL OR current_run_id = ?)",
            (grace, task_id, claim_lock, expected_run_id, expected_run_id),
        )
        if cur.rowcount != 1:
            return
        run_id = _kanban_db._current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                (grace, run_id),
            )
        payload = {
            "reason": reason,
            "claim_lock": claim_lock,
            "claim_expires_now": grace,
        }
        payload.update(termination)
        _kanban_db._append_event(
            conn, task_id, "reclaim_deferred", payload, run_id=run_id
        )


from hermes_cli import kanban_db as _kanban_db
from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_stop as _kanban_worker_stop


def _legacy_worker_fingerprint(pid, task_id, run_id, expected_db):
    """Recover missing legacy fingerprint from exact worker provenance."""
    if task_id is None or run_id is None:
        return None
    try:
        import psutil

        before = _worker_pid_start_time(int(pid))
        if before is None:
            return None
        process = psutil.Process(int(pid))
        created = process.create_time()
        env = process.environ()
        if env.get("HERMES_KANBAN_TASK") != str(task_id) or env.get(
            "HERMES_KANBAN_RUN_ID"
        ) != str(run_id):
            return None
        owned_db = env.get("HERMES_KANBAN_DB")
        if not owned_db:
            return None
        target = expected_db or str(_owner_kanban_db.kanban_db_path())
        if Path(owned_db).resolve() != Path(target).resolve():
            return None
        after = _worker_pid_start_time(int(pid))
        if (
            before != after
            or not process.is_running()
            or psutil.Process(int(pid)).create_time() != created
        ):
            return None
        return before
    except (OSError, ValueError, psutil.Error):
        return None
