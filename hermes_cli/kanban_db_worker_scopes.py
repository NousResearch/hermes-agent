"""Transient worker scopes: what keeps a leftover child alive, and what a kernel OOM kill looks like.

A worker runs in a transient ``hermes-worker-kanban-<task>-run-<n>.scope`` cgroup
(``tools/process_registry``), and both questions about it are answered by systemd rather than by the
board's run row. ``--collect`` removes a scope only once its cgroup is empty, so a worker that ends
leaving a background child behind (a ``bun run dev``, a ``python3 -m http.server``, an ``Xvfb``, a
``while pgrep ...; do sleep 15; done`` loop) keeps the scope loaded and holds the child's port and
memory; and because a scope runs with the default ``OOMPolicy=stop``, a kernel OOM kill stops the
whole scope, so an OOM death reaches the dispatcher as a clean rc=0 or as a vanished pid.

Split out of ``hermes_cli.kanban_db_dispatch`` (a 2,000-line facade may not grow); origin-resident
helpers are reached late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import time

# A healthy worker is still alive for a while after kanban_complete / kanban_request_review
# returns (final assistant turn, session persistence), so a run's retained worker and its scope
# are only reaped once ended_at is at least this old (two default dispatch ticks).
TERMINAL_WORKER_REAP_GRACE_SECONDS = 120

WORKER_SCOPE_UNIT_PREFIX = "hermes-worker-kanban-"
WORKER_SCOPE_UNIT_RE = re.compile(
    rf"^{re.escape(WORKER_SCOPE_UNIT_PREFIX)}(?P<task_id>.+)-run-(?P<run_id>\d+)\.scope$"
)

_SCOPE_OOM_MARKERS = (
    "Failed with result 'oom-kill'",
    "The kernel OOM killer killed some processes in this unit",
)


def _loaded_worker_scope_units() -> list[str]:
    """Unit names of the worker scopes the user manager still has loaded.

    One ``systemctl --user list-units`` call per tick: the loaded set is the
    authoritative answer to "which worker cgroups still hold a process", and it
    reaches a scope whose run row predates this sweep (or whose retained worker
    evidence a previous tick already cleared), which a per-run SELECT cannot.
    ``systemctl`` missing, or a user bus that cannot be reached, is a silent
    no-op: scope reaping must never be the reason a tick fails.
    """
    import shutil

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
    units: list[str] = []
    for line in (proc.stdout or "").splitlines():
        if not line.strip():
            continue
        unit = line.split(None, 1)[0]
        if WORKER_SCOPE_UNIT_RE.match(unit):
            units.append(unit)
    return units


def _stop_worker_scope_unit(unit_name: str) -> bool:
    """``systemctl --user stop`` one transient worker scope; True if stopped or
    already gone. Thin wrapper (the stop itself is shared with every other
    scope-kill path in ``tools/process_registry``), so the sweep has one seam to
    fake in tests."""
    from tools.process_registry import _stop_systemd_unit

    return _stop_systemd_unit(unit_name)


def reap_terminal_run_scopes(conn: sqlite3.Connection, *, stop_fn=None) -> list[str]:
    """Stop the transient worker scope of every run that has ended.

    A worker scope is a cgroup, so stopping it reaps whatever the worker left
    running inside it: the dev and preview servers, the ``Xvfb``, the
    ``while pgrep ...; do sleep 15; done`` loop that had nothing left to wait for.
    All of those outlive the agent process that started them, hold their port, and
    answer to no signal sent to the worker's own pid, which is exactly why
    ``reap_terminal_workers`` cannot clear them.

    Only a scope whose own run row is closed is ever stopped, so a live run is
    untouched; a scope whose run id is not in this board's DB (scopes are named by
    task id alone) is left to the board that owns it; and a run that ended less
    than ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago is left alone so a worker
    still finalising after its own transition is not killed. ``stop_fn`` is the
    test hook. One scope's failure is logged and skips only that scope. Returns
    the unit names stopped.
    """
    units = _loaded_worker_scope_units()
    if not units:
        return []
    stop = stop_fn if stop_fn is not None else _stop_worker_scope_unit
    cutoff = int(time.time()) - TERMINAL_WORKER_REAP_GRACE_SECONDS
    stopped: list[str] = []
    for unit in units:
        match = WORKER_SCOPE_UNIT_RE.match(unit)
        if match is None:  # pragma: no cover - the lister applies the same pattern
            continue
        task_id, run_id = match.group("task_id"), int(match.group("run_id"))
        try:
            row = conn.execute(
                "SELECT ended_at FROM task_runs WHERE id = ? AND task_id = ?",
                (run_id, task_id),
            ).fetchone()
        except sqlite3.Error:  # pragma: no cover - defensive
            _kb._log.debug(
                "kanban dispatch: worker scope reap lookup failed for %s", unit, exc_info=True,
            )
            continue
        if row is None or row["ended_at"] is None or int(row["ended_at"]) > cutoff:
            continue
        try:
            if not stop(unit):
                continue
        except Exception:  # health: allow BLE001 -- a stop seam any failure can come from; one scope must not abort the sweep
            _kb._log.debug(
                "kanban dispatch: stopping worker scope %s failed for task %s",
                unit, task_id, exc_info=True,
            )
            continue
        stopped.append(unit)
        try:
            with _kb.write_txn(conn):
                _kb._append_event(
                    conn, task_id, "terminal_run_scope_reaped",
                    {"unit": unit, "run_id": run_id}, run_id=run_id,
                )
        except (sqlite3.Error, ValueError, TypeError):  # the stop has already happened
            _kb._log.debug(
                "kanban dispatch: scope reap event failed for %s", unit, exc_info=True,
            )
    return stopped


def _worker_scope_oom_killed(task_id: str, *, window: str = "-3h") -> bool:
    """True when a transient worker scope of ``task_id`` was stopped by the kernel OOM killer.

    A worker scope runs with the default ``OOMPolicy=stop``, so when the kernel
    kills a process inside it systemd stops the whole scope and the worker is
    left with no way to report anything: its signal path exits 0, or the scope is
    already gone before a sweeper polls the pid. ``--collect`` removes the scope
    the moment it exits, so ``systemctl show`` reports ``LoadState=not-found``
    with ``Result=success`` afterwards and cannot tell an OOM death from a clean
    one. The user manager's journal keeps the record, so read it there, bounded to
    this one task's scopes and a short window (only a death that is about to be
    booked asks).
    """
    try:
        proc = subprocess.run(
            [
                "journalctl", "--user",
                "-u", f"hermes-worker-kanban-{task_id}-run-*.scope",
                "--since", window, "-o", "cat", "--no-pager",
            ],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = proc.stdout or ""
    return any(marker in output for marker in _SCOPE_OOM_MARKERS)


from hermes_cli import kanban_db as _kb
