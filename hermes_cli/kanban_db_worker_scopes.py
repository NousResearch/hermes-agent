"""The transient systemd scope a kanban worker runs in, read by the dispatcher's crash sweep.

A systemd-hosted dispatcher wraps each worker in ``hermes-worker-kanban-<task>-run-<n>.scope``
(``_restart_safe_worker_argv``). The scope runs with the default ``OOMPolicy=stop``, so when the
kernel OOM killer kills a process inside it systemd stops the whole scope and the worker never
reaches its exit epilogue: the death reaches the sweep as a clean rc=0 or as a vanished pid.
``--collect`` removes the scope the moment it exits, so ``systemctl show`` cannot tell that death
from a clean one afterwards; the user manager's journal still can.

Split out of ``hermes_cli.kanban_db_dispatch`` (a facade over its line cap may not grow);
origin-resident helpers are reached late-bound via ``_kb`` (import-cycle breaking) so
monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import time
from typing import Any, Callable

WORKER_SCOPE_UNIT_PREFIX = "hermes-worker-kanban-"

_SCOPE_OOM_MARKERS = (
    "Failed with result 'oom-kill'",
    "The kernel OOM killer killed some processes in this unit",
)
# An exact unit name is an indexed journal match and answers in milliseconds; a glob over every
# run's scope makes journalctl enumerate every unit name the journal ever saw.
_OOM_PROBE_TIMEOUT_S = 10


def worker_scope_unit(task_id: str, run_id: int) -> str:
    """The scope unit ``_restart_safe_worker_argv`` starts for run *run_id* of *task_id*."""
    return f"{WORKER_SCOPE_UNIT_PREFIX}{task_id}-run-{run_id}.scope"


def worker_scope_oom_killed(task_id: str, run_id: int) -> bool:
    """True when the kernel OOM killer stopped the worker scope of this one run.

    Asks the journal about the run's own unit by its exact name: the unit is unique per run, so
    no earlier run of the same card can answer for this one. A missing or unreadable journal is
    ``False``, the booking the death had before this probe existed.
    """
    try:
        proc = subprocess.run(
            ["journalctl", "--user", "-u", worker_scope_unit(task_id, run_id),
             "-o", "cat", "--no-pager"],
            capture_output=True, text=True, timeout=_OOM_PROBE_TIMEOUT_S, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = proc.stdout or ""
    return any(marker in output for marker in _SCOPE_OOM_MARKERS)


def dead_worker_verdicts(
    conn: sqlite3.Connection, alive: Callable[[Any, Any], bool],
) -> dict[tuple[str, int, str], bool]:
    """``(task id, worker pid, claim lock) -> OOM-killed`` for every dead host-local worker.

    Runs BEFORE the crash sweep opens its write transaction, because the OOM verdict is a
    journal read and a journal that is slow to answer must cost the tick time, never hold the
    board's write lock. The key pins the exact worker: the sweep books only a row whose task,
    pid and claim are unchanged, so a worker respawned in between is never reclaimed, and one
    that died after this read waits for the next tick, when its own verdict is read. ``alive``
    is the dispatcher's liveness check (pid plus spawn fingerprint).
    """
    rows = conn.execute(
        "SELECT id, worker_pid, worker_started_at, claim_lock, started_at, current_run_id "
        "FROM tasks WHERE status = 'running' AND worker_pid IS NOT NULL"
    ).fetchall()
    host_prefix = _kb._host_prefix()
    can_probe = shutil.which("journalctl") is not None
    verdicts: dict[tuple[str, int, str], bool] = {}
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        # Launch-window grace so a freshly-spawned worker isn't reclaimed
        # before its PID is visible on /proc.
        started_at = _kb._to_epoch(_kb._row_get(row, "started_at"))
        if started_at is not None and time.time() - started_at < _kb._resolve_crash_grace_seconds():
            continue
        if alive(row["worker_pid"], _kb._row_get(row, "worker_started_at")):
            continue
        run_id = _kb._row_get(row, "current_run_id")
        verdicts[(row["id"], int(row["worker_pid"]), row["claim_lock"])] = (
            can_probe and run_id is not None and worker_scope_oom_killed(row["id"], int(run_id))
        )
    return verdicts


def book_oom_kill(dead: Any, pid: int) -> None:
    """Rebook *dead* (the sweep's ``_DeadWorker``) as the OOM kill it was.

    The rc=0 the scope's stop leaves is the worker's signal path, not a worker that skipped its
    terminal kanban call, so it is a crash carrying the cause, never a protocol violation that
    feeds the violation streak. A bare crash ("pid not alive": the scope was collected before the
    sweep polled the pid) keeps its kind and gains the cause. The remedy rides the error text so
    it reaches the retry worker's context. A rate-limit or terminal-provider exit is the worker's
    own report and stays as it is.
    """
    if dead.rate_limited or dead.terminal_provider:
        return
    if dead.protocol_violation:
        dead.protocol_violation = False
        dead.event_kind = "crashed"
        dead.event_payload.pop("protocol_violation", None)
        # Never the words "protocol violation": the streak counts a run whose error says so.
        dead.error_text = (
            f"pid {pid} exited rc=0 because the kernel OOM killer stopped its worker scope: the "
            "lane needed more memory than the scope limit, and no terminal kanban call was "
            "possible from inside the stopped scope. Bound the lane's memory (test "
            "parallelism) or raise the worker ceiling, then retry."
        )
    else:
        dead.error_text += (
            " - the kernel OOM killer stopped this run's worker scope: the lane needed more "
            "memory than the scope limit"
        )
    dead.event_payload["oom_killed"] = True


# Late-bound origin namespace (see module docstring).
from hermes_cli import kanban_db as _kb
