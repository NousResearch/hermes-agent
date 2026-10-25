"""Worker liveness beyond the pid probe (sibling of ``kanban_db_dispatch``): card-level identity
witnesses, drift-safe reclaim termination, and re-attaching a run the sweep closed by mistake.

Why this exists: the 2026-10-25 gmantra incident. The worker fingerprint ("<epoch>|<start time>")
was compared with exact equality while, on macOS, psutil's ``create_time()`` moved with
``kern.boottime`` (``_psosx.adjust_proc_create_time``) between the spawn record and the sweep —
seven live, heartbeating workers were declared ``crashed: pid N not alive``, re-spawned beside, and
one pair raced the same worktree. Every dispatch-level helper is read through ``_kbd()`` /
``_kb()`` at call time so the facades stay the patch seams.
"""

from __future__ import annotations

import os
import re
import signal
import sqlite3
import time
from typing import Any, Optional


def _kb():
    from hermes_cli import kanban_db
    return kanban_db


def _kbd():
    from hermes_cli import kanban_db_dispatch
    return kanban_db_dispatch


# A ``running`` card whose worker heartbeated inside this window is alive whatever the pid probe
# concludes about the process's identity: the heartbeat is the worker's own, DB-side proof of life.
# Same bound as the wedged-worker reclaim (``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS``), so a
# worker that is truly gone is still released once its last heartbeat ages past it.
HEARTBEAT_LIVENESS_WINDOW_SECONDS = 60 * 60


def _heartbeat_fresh(last_heartbeat_at, now: Optional[float] = None) -> bool:
    """True when ``last_heartbeat_at`` lies inside ``HEARTBEAT_LIVENESS_WINDOW_SECONDS``."""
    if last_heartbeat_at is None:
        return False
    try:
        age = (now if now is not None else time.time()) - int(last_heartbeat_at)
    except (TypeError, ValueError):
        return False
    return 0 <= age <= HEARTBEAT_LIVENESS_WINDOW_SECONDS


def _task_token(task_id: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(task_id)}(?![\w-])")


def _process_names_task(pid: int, task_id: str) -> Optional[bool]:
    """Is the live process (or one of its descendants) the worker of ``task_id``?

    The dispatcher pins ``HERMES_KANBAN_TASK`` in the worker's environment and every tool shell
    inherits it, so the environment is the identity record — read it first (psutil: ``/proc`` on
    Linux, ``KERN_PROCARGS2`` on macOS, same-user only). When the environment cannot be read the
    worker's argv (``chat -q "work kanban task <id>"``, see ``_worker_argv``) is the fallback,
    matched as a whole token. ``None`` when neither source can be read."""
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        try:
            env = psutil.Process(int(pid)).environ()
        except (psutil.Error, OSError, ValueError):
            # NoSuchProcess / AccessDenied (another user's process on macOS) / ZombieProcess:
            # no environment to read, argv decides below.
            env = None
        if env is not None and "HERMES_KANBAN_TASK" in env:
            return env.get("HERMES_KANBAN_TASK") == task_id
    from gateway.status import _read_process_cmdline
    cmdline = _read_process_cmdline(int(pid))
    if not cmdline:
        return None
    return _task_token(task_id).search(cmdline) is not None


def _worker_identity_hold(
    pid: Optional[int], started_at, *, task_id: str, last_heartbeat_at,
) -> Optional[str]:
    """Why a live ``pid`` whose fingerprint disagrees with the spawn record is STILL our worker, or
    ``None`` when nothing vouches for it (a stranger recycled the number).

    The fingerprint is one witness; the sweep must not kill a card on it alone. ``"process"``: the
    process carries this very card in its environment / argv — no stranger does. ``"heartbeat"``:
    the card heartbeated inside ``HEARTBEAT_LIVENESS_WINDOW_SECONDS`` — a dead worker cannot. Only
    called for a pid that exists; a pid absent from the process table is dead, kernel-authoritatively.
    """
    if not pid or not task_id:
        return None
    if _process_names_task(int(pid), task_id):
        return "process"
    if _heartbeat_fresh(last_heartbeat_at):
        return "heartbeat"
    return None


def _worker_alive_for_task(
    pid: Optional[int], started_at, *, task_id: str, last_heartbeat_at,
) -> tuple[bool, Optional[str]]:
    """``_worker_alive`` plus the card-level witnesses: ``(alive, hold_reason)``. ``hold_reason`` is
    set only when the fingerprint said "recycled" and a witness overrode it — the caller should
    re-record the fingerprint (``process``) or at least not reclaim (``heartbeat``)."""
    kbd = _kbd()
    if not _kb()._pid_alive(pid):
        return False, None
    if started_at == kbd.UNVERIFIED_WORKER_FINGERPRINT or not kbd._pid_recycled(pid, started_at):
        return True, None
    reason = _worker_identity_hold(pid, started_at, task_id=task_id, last_heartbeat_at=last_heartbeat_at)
    return (reason is not None), reason


def _rebind_worker_fingerprint(conn: sqlite3.Connection, task_id: str, pid: int, reason: str,
                               previous) -> Optional[str]:
    """Re-record the live worker's fingerprint on the task + run rows after a witness proved the
    stored one stale (``liveness_hold`` event carries both). Caller holds the write txn. Returns the
    new fingerprint, or ``None`` when nothing was rewritten (heartbeat-only hold, or unreadable)."""
    kb = _kb()
    fresh = _kbd()._process_fingerprint(int(pid))
    run_id = kb._current_run_id(conn, task_id)
    rebound = fresh is not None and reason == "process"
    if rebound:
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ? AND worker_pid = ?",
                     (fresh, task_id, int(pid)))
        if run_id is not None:
            conn.execute("UPDATE task_runs SET worker_started_at = ? WHERE id = ? AND worker_pid = ?",
                         (fresh, run_id, int(pid)))
    # One event per run per witness kind: the hold is re-evaluated every tick, the record is not.
    already = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'liveness_hold' "
        "AND run_id IS ? AND payload LIKE ? LIMIT 1",
        (task_id, run_id, f'%"reason": "{reason}"%'),
    ).fetchone()
    if already is None:
        kb._append_event(conn, task_id, "liveness_hold", {
            "pid": int(pid), "reason": reason, "recorded_fingerprint": previous,
            "live_fingerprint": fresh, "rebound": rebound,
        }, run_id=run_id)
    return fresh if rebound else None


def _terminate_worker_process_group(pid: int, task_id: str, workspace: Optional[str], *,
                                    kill_fn=None) -> list[int]:
    """SIGTERM what is left of a dead worker's process group before its card is re-spawned.

    Workers start their own session (``start_new_session=True``), so the group id is the worker's
    pid; when the leader is gone its children (tool shells, build/test subprocesses) can outlive
    it in the same workspace and race the replacement worker's checkout. Every member is verified
    before it is signalled — it inherited the card's ``HERMES_KANBAN_TASK`` (``_process_names_task``)
    or its cwd is the card's workspace — so a stranger that inherited the number is left alone.
    Returns the pids signalled."""
    kill = _kbd()._kill_fn(kill_fn)
    if kill is None or not pid or not hasattr(os, "getpgid"):
        return []
    try:
        import psutil  # type: ignore
    except ImportError:
        return []
    signalled: list[int] = []
    ws = os.path.realpath(workspace) if workspace else None
    for proc in psutil.process_iter(["pid", "cwd"]):
        member = proc.info["pid"]
        if member == os.getpid():
            continue
        try:
            if os.getpgid(member) != int(pid):
                continue
        except (ProcessLookupError, PermissionError, OSError):
            continue
        ours = _process_names_task(member, task_id) is True
        cwd = proc.info.get("cwd")
        if not ours and ws and cwd:
            try:
                real = os.path.realpath(cwd)
                ours = real == ws or real.startswith(ws + os.sep)
            except OSError:
                ours = False
        if not ours:
            continue
        try:
            kill(member, signal.SIGTERM)
            signalled.append(member)
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return signalled


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
    started_at=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths. ``started_at`` is the spawn-time
    fingerprint: when the live process no longer matches it, the PID was recycled and nothing is
    signalled — the worker is gone, which is what the reclaim wanted (``terminated`` = True). An
    UNVERIFIED spawn (fingerprint capture failed) that is still live is never signalled either, but
    it is reported as surviving (``signal_refused``) so the reclaim holds the claim instead of
    spawning a duplicate beside it."""
    kb, kbd = _kb(), _kbd()
    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info
    if not str(claim_lock).startswith(kb._host_prefix()):
        return info
    info["host_local"] = True

    kill = kbd._kill_fn(signal_fn)
    if kill is None:
        return info
    if started_at == kbd.UNVERIFIED_WORKER_FINGERPRINT:
        # Never signal by bare number: a dead PID is "gone" (reclaim proceeds), a live one is held.
        info["signal_refused"] = True
        info["terminated"] = not kb._pid_alive(pid)
        return info
    if kb._pid_alive(pid):
        verdict = kbd._fingerprint_matches(pid, started_at)
        if verdict is False:
            info["terminated"] = True
            info["pid_recycled"] = True
            return info
        if verdict is None:
            # Live, identity unreadable right now: an unverified process is never signalled, and
            # the claim is held (not released beside it). The next sweep re-reads.
            info["signal_refused"] = True
            info["fingerprint_unreadable"] = True
            return info

    info["termination_attempted"] = True
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Already gone = successful termination. Leaving terminated=False would
        # make the reclaim guard misread a dead worker as alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    if kbd._poll_worker_exit(pid, started_at):
        info["terminated"] = True
        return info
    if kbd._worker_alive(pid, started_at):
        if not kbd._sigkill(kill, pid):
            return info
        info["sigkill"] = True
    info["terminated"] = not kbd._worker_alive(pid, started_at)
    return info


# Run outcomes a dispatcher sweep writes on the worker's behalf when it believes the process is
# gone. A later board call from that very run is proof the sweep was wrong, so these — and only
# these — may be reopened. ``timed_out``/``gave_up``-by-operator and the worker's own terminal
# transitions are intent, not a liveness verdict, and stay closed.
REATTACHABLE_RUN_OUTCOMES = frozenset({"crashed", "reclaimed"})


def reattach_orphaned_run(
    conn: sqlite3.Connection, task_id: str, run_id: int, *, pid: int, claim_lock: Optional[str] = None,
) -> str:
    """Re-open a run the dispatcher closed as dead when its worker proves it is alive.

    The worker still holds ``HERMES_KANBAN_RUN_ID``; when its next ``kanban_heartbeat`` /
    ``kanban_complete`` / ``kanban_block`` finds the run closed with a liveness outcome
    (``REATTACHABLE_RUN_OUTCOMES``) and no successor run live on the card, the run is reopened as
    the card's current run under the caller's pid + fresh fingerprint, the card returns to
    ``running``, and the crash that was counted against the breaker is uncounted — the handoff then
    lands normally instead of ``unknown id or not running`` (2026-10-25: a finished card stranded in
    ``blocked`` with its result only in a comment). Verdicts:

    ``"reattached"`` — done, proceed. ``"successor_live"`` — another run owns the card now; the caller
    must NOT mutate (an ``orphan_worker_refused`` event surfaces the duplicate to the operator).
    ``"not_orphaned"`` — the run was closed by intent (completed, blocked, timed out) or the card is
    terminal; nothing changed. ``"unknown"`` — no such run on this card.
    """
    kb, kbd = _kb(), _kbd()
    now = int(time.time())
    fresh = kbd._process_fingerprint(int(pid)) or kbd.UNVERIFIED_WORKER_FINGERPRINT
    with kb.write_txn(conn):
        run = conn.execute(
            "SELECT id, task_id, status, outcome, claim_lock FROM task_runs WHERE id = ? AND task_id = ?",
            (int(run_id), task_id),
        ).fetchone()
        if run is None:
            return "unknown"
        task = conn.execute(
            "SELECT status, current_run_id, consecutive_failures FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task is None:
            return "unknown"
        if task["current_run_id"] == int(run_id) and task["status"] == "running":
            return "reattached"  # nothing to do: the run is still live
        if task["status"] == "running" and task["current_run_id"] not in (None, int(run_id)):
            already = conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'orphan_worker_refused' "
                "AND run_id = ? LIMIT 1", (task_id, int(run_id)),
            ).fetchone()
            if already is None:
                kb._append_event(conn, task_id, "orphan_worker_refused", {
                    "pid": int(pid), "orphaned_run_id": int(run_id), "orphaned_outcome": run["outcome"],
                    "successor_run_id": int(task["current_run_id"]),
                    "note": "two workers reached this card: the dispatcher closed the first run as "
                            "dead while its worker was alive, then spawned a successor",
                }, run_id=int(run_id))
            return "successor_live"
        if run["outcome"] not in REATTACHABLE_RUN_OUTCOMES or task["status"] in ("done", "archived"):
            return "not_orphaned"
        lock = claim_lock or run["claim_lock"] or kb._claimer_id()
        expires = now + kb._resolve_claim_ttl_seconds()
        cur = conn.execute(
            "UPDATE task_runs SET status = 'running', outcome = NULL, ended_at = NULL, error = NULL, "
            "claim_lock = ?, claim_expires = ?, worker_pid = ?, worker_started_at = ?, "
            "last_heartbeat_at = ? WHERE id = ? AND task_id = ? AND outcome IN ('crashed', 'reclaimed')",
            (lock, expires, int(pid), fresh, now, int(run_id), task_id),
        )
        if cur.rowcount != 1:
            return "not_orphaned"
        failures = max(0, int(task["consecutive_failures"] or 0) - 1)
        conn.execute(
            "UPDATE tasks SET status = 'running', current_run_id = ?, claim_lock = ?, claim_expires = ?, "
            "worker_pid = ?, worker_started_at = ?, last_heartbeat_at = ?, consecutive_failures = ?, "
            "last_failure_error = NULL, block_kind = NULL WHERE id = ?",
            (int(run_id), lock, expires, int(pid), fresh, now, failures, task_id),
        )
        kb._append_event(conn, task_id, "run_reattached", {
            "pid": int(pid), "started_at": fresh, "previous_outcome": run["outcome"],
            "previous_task_status": task["status"], "consecutive_failures": failures,
        }, run_id=int(run_id))
    return "reattached"
