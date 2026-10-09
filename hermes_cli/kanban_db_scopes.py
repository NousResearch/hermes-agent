"""Kanban worker systemd scopes: the unit name a worker's transient scope gets,
wrapping the worker command in it, and stopping scopes that outlived their run.

Split out of ``hermes_cli.kanban_db_dispatch``; the facade is late-imported
inside functions (import-cycle breaking).
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_log = logging.getLogger(__name__)

# Each stop is a blocking ``systemctl --user stop`` under the dispatch lock: a
# backlog of leaked scopes drains over several ticks instead of stalling one.
MAX_SCOPE_STOPS_PER_TICK = 5


def kanban_worker_unit_suffix(task_id: str, run_id: "int | str") -> str:
    """Systemd unit suffix of a Kanban worker's scope (``hermes-worker-<suffix>.scope``).

    The one place the format lives: the spawn path names the scope with it and
    :func:`stop_ended_worker_scopes` parses scope names back with it."""
    return f"kanban-{task_id}-run-{run_id}"


# Parses scope names back into (task, run) by escaping the helper's own output
# around two placeholders, so the format is never written out a second time.
_KANBAN_WORKER_SCOPE_RE = re.compile(
    "^hermes-worker-"
    + re.escape(kanban_worker_unit_suffix("TASKID", "RUNID"))
    .replace("TASKID", r"(?P<task>[A-Za-z0-9_]+)")
    .replace("RUNID", r"(?P<run>[0-9]+)")
    + r"\.scope$"
)


def _active_kanban_worker_scopes() -> list[tuple[str, str, int]]:
    """``(unit, task_id, run_id)`` for every loaded ``hermes-worker-kanban-*`` scope of
    this user's systemd manager; empty when systemctl is missing or fails."""
    from tools.process_registry import systemd_user_bus_env

    binary = shutil.which("systemctl")
    if binary is None:
        return []
    proc = subprocess.run(
        [binary, "--user", "list-units", "--type=scope", "--state=active",
         "--no-legend", "--plain", "hermes-worker-kanban-*"],
        capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        env=systemd_user_bus_env(),
    )
    if proc.returncode != 0:
        return []
    scopes = []
    for line in proc.stdout.splitlines():
        unit = line.split(maxsplit=1)[0] if line.strip() else ""
        m = _KANBAN_WORKER_SCOPE_RE.match(unit)
        if m:
            scopes.append((unit, m["task"], int(m["run"])))
    return scopes


def stop_ended_worker_scopes(conn: sqlite3.Connection) -> list[str]:
    """Stop Kanban worker scopes that outlived their run.

    A worker's descendants (e.g. a browser-harness daemon reparented to the user
    manager) keep its transient ``--collect`` scope alive after the worker itself
    is gone, so the run's processes leak indefinitely; ``reap_terminal_workers``
    cannot see them once the run's ``worker_pid`` is cleared. Each tick stops,
    via ``systemctl --user stop`` (whole cgroup), every scope whose run in THIS
    board's DB ended more than ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago, at
    most ``MAX_SCOPE_STOPS_PER_TICK`` per call. The scope list comes from this
    host's own user manager; a run whose claim_lock names another host is still
    skipped (reclaimed runs have none). Open runs, fresh runs and unknown runs
    (other boards) are never touched. Runs on ``dry_run`` ticks too, like
    :func:`reap_terminal_workers`. Best-effort: systemctl missing or failing just
    returns fewer units. Returns the stopped units."""
    if sys.platform != "linux":
        return []
    try:
        scopes = _active_kanban_worker_scopes()
    except (OSError, subprocess.SubprocessError):
        _log.debug("kanban dispatch: listing worker scopes failed", exc_info=True)
        return []
    if not scopes:
        return []
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import TERMINAL_WORKER_REAP_GRACE_SECONDS
    from tools.process_registry import _stop_systemd_unit

    cutoff = int(time.time()) - TERMINAL_WORKER_REAP_GRACE_SECONDS
    host_prefix = _kb._host_prefix()
    stopped: list[str] = []
    for unit, task_id, run_id in scopes:
        if len(stopped) >= MAX_SCOPE_STOPS_PER_TICK:
            break
        row = conn.execute(
            "SELECT ended_at, claim_lock FROM task_runs WHERE id = ? AND task_id = ?",
            (run_id, task_id),
        ).fetchone()
        if row is None or row["ended_at"] is None or row["ended_at"] > cutoff:
            continue
        # A reclaimed run has its claim_lock cleared; any lock left must be ours.
        if row["claim_lock"] and not str(row["claim_lock"]).startswith(host_prefix):
            continue
        if _stop_systemd_unit(unit):  # never raises: False on any failure
            stopped.append(unit)
    if stopped:
        _log.info("kanban dispatch: stopped %d scope(s) of ended runs: %s", len(stopped), ", ".join(stopped))
    return stopped


def _restart_safe_worker_argv(task: Task, command: list[str]) -> list[str]:
    """Wrap a systemd-hosted dispatcher's worker in the shared restart-safe scope.

    Kanban workers are long-lived agentic runs that outlive the dispatcher
    tick, so they never take cron's degraded mode under the managed gateway:
    ``require_restart_safe_scope=True`` makes the helper raise
    ``RestartSafeScopeUnavailable`` there (an infrastructure spawn failure the
    dispatcher does not charge to the card). Under any other systemd unit
    (``Type=oneshot`` dispatch timers, #113612) ``outlives_parent=True`` gets the
    worker its own scope so the unit's cgroup teardown cannot kill it.
    """
    from tools.process_registry import restart_safe_gateway_child_argv

    if task.current_run_id is None:
        # Outside managed systemd this is harmless, but a managed dispatch must
        # never mint an untraceable worker.  Check topology through the shared
        # helper first, using a placeholder suffix that cannot be launched.
        dispatch = restart_safe_gateway_child_argv(
            command,
            unit_suffix=kanban_worker_unit_suffix(task.id, "missing"),
            require_restart_safe_scope=True,
            outlives_parent=True,
        )
        if dispatch.mode != "in_process":
            raise RuntimeError(
                "cannot create restart-safe systemd scope for Kanban worker: "
                "the claimed task has no current run id"
            )
        return command

    return restart_safe_gateway_child_argv(
        command,
        unit_suffix=kanban_worker_unit_suffix(task.id, task.current_run_id),
        require_restart_safe_scope=True,
        outlives_parent=True,
    ).argv
