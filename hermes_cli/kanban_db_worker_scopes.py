"""The transient systemd scope a kanban worker runs in: how it dies, and what it leaves loaded.

A systemd-hosted dispatcher wraps each worker in ``hermes-worker-kanban-<task>-run-<n>.scope``
(``_restart_safe_worker_argv``), and both questions about it are answered by systemd rather than
by the board's run row. The scope runs with the default ``OOMPolicy=stop``, so when the kernel
OOM killer kills a process inside it systemd stops the whole scope and the worker never reaches
its exit epilogue: the death reaches the sweep as a clean rc=0 or as a vanished pid.
``--collect`` removes the scope the moment it exits, so ``systemctl show`` cannot tell that death
from a clean one afterwards; the user manager's journal still can. And ``--collect`` removes a
scope only once its cgroup is empty, so a worker that ends leaving a background child behind (a
dev server, a build daemon, an ``Xvfb``, a wait loop) keeps the scope loaded, holding the
child's port and memory, until something stops it.

Split out of ``hermes_cli.kanban_db_dispatch`` (a facade over its line cap may not grow);
origin-resident helpers are reached late-bound via ``_kb`` (import-cycle breaking) so
monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
import time
from typing import Any, Callable, Optional

# A healthy worker is still alive for a while after kanban_complete /
# kanban_request_review returns (final assistant turn, session persistence), so
# a run's retained worker and its scope are only reaped once ended_at is at
# least this old (two default dispatch ticks).
TERMINAL_WORKER_REAP_GRACE_SECONDS = 120

WORKER_SCOPE_UNIT_PREFIX = "hermes-worker-kanban-"
WORKER_SCOPE_UNIT_RE = re.compile(
    rf"^{re.escape(WORKER_SCOPE_UNIT_PREFIX)}(?P<task_id>.+)-run-(?P<run_id>\d+)\.scope$"
)

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


def _loaded_worker_scope_units() -> list[str]:
    """Unit names of the worker scopes the user manager still has loaded.

    One ``systemctl --user list-units`` call per tick: the loaded set is the authoritative answer
    to "which worker cgroups still hold a process", and it reaches a scope whose retained worker
    evidence a reclaim or an earlier tick already cleared, which a per-run query cannot.
    ``systemctl`` missing, or a user bus that cannot be reached, is a silent no-op: scope reaping
    must never be the reason a tick fails.
    """
    binary = shutil.which("systemctl")
    if binary is None:
        return []
    from tools.process_registry import systemd_user_bus_env

    try:
        proc = subprocess.run(
            [binary, "--user", "list-units", "--type=scope", "--all", "--no-pager",
             "--plain", "--no-legend", f"{WORKER_SCOPE_UNIT_PREFIX}*.scope"],
            capture_output=True, text=True, timeout=15, check=False,
            stdin=subprocess.DEVNULL, env=systemd_user_bus_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    units = (line.split(None, 1)[0] for line in (proc.stdout or "").splitlines() if line.strip())
    return [unit for unit in units if WORKER_SCOPE_UNIT_RE.match(unit)]


def _stop_worker_scope_unit(unit_name: str) -> bool:
    """``systemctl --user stop`` one worker scope; True if stopped or already gone.

    The stop is the one every other scope-kill path shares (``tools/process_registry``); this
    wrapper is the sweep's single seam for tests.
    """
    from tools.process_registry import _stop_systemd_unit

    return _stop_systemd_unit(unit_name)


def reap_terminal_run_scopes(conn: sqlite3.Connection, *, stop_fn: Optional[Callable[[str], bool]] = None) -> list[str]:
    """Stop the transient worker scope of every run that has already ended.

    A worker scope is a cgroup, so stopping it reaps whatever the worker left running inside it.
    Those children outlive the agent process that started them, hold their port, and answer to
    no signal sent to the worker's own pid, which is why ``reap_terminal_workers`` (keyed on the
    run's retained pid, which a reclaim may already have cleared) cannot clear them.

    Only a scope whose own run row is closed is ever stopped, so a live run is untouched; a scope
    whose run is not in this board's DB (scopes are named by task id alone) is left to the board
    that owns it; and a run that ended less than ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago is left
    alone so a worker still finalising after its own transition is not killed. One scope's failure
    is logged and skips only that scope. Returns the unit names stopped.
    """
    units = _loaded_worker_scope_units()
    if not units:
        return []
    stop = stop_fn if stop_fn is not None else _stop_worker_scope_unit
    cutoff = int(time.time()) - TERMINAL_WORKER_REAP_GRACE_SECONDS
    stopped: list[str] = []
    for unit in units:
        try:
            if _reap_run_scope(conn, unit, cutoff, stop):
                stopped.append(unit)
        except Exception:  # health: allow BLE001 -- per-scope boundary like reap_terminal_workers; logs exc_info
            _kb._log.debug("kanban dispatch: worker scope reap failed for %s", unit, exc_info=True)
    return stopped


def _reap_run_scope(conn: sqlite3.Connection, unit: str, cutoff: int, stop: Callable[[str], bool]) -> bool:
    match = WORKER_SCOPE_UNIT_RE.match(unit)
    if match is None:
        return False
    task_id, run_id = match.group("task_id"), int(match.group("run_id"))
    row = conn.execute(
        "SELECT ended_at FROM task_runs WHERE id = ? AND task_id = ?", (run_id, task_id),
    ).fetchone()
    if row is None or row["ended_at"] is None or int(row["ended_at"]) > cutoff:
        return False
    if not stop(unit):
        return False
    with _kb.write_txn(conn):
        _kb._append_event(
            conn, task_id, "terminal_run_scope_reaped", {"unit": unit, "run_id": run_id}, run_id=run_id,
        )
    return True


# Late-bound origin namespace (see module docstring).
from hermes_cli import kanban_db as _kb
