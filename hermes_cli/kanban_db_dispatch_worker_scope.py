"""Kanban worker scope lifecycle: the exact systemd scope unit a worker run is spawned in (``_restart_safe_worker_argv``) and its release by the terminal-worker reaper once the run's client is gone.

Split out of ``hermes_cli.kanban_db_dispatch``; ``hermes_cli.kanban_db`` helpers
are reached late-bound inside functions (never a module-level import: that
module imports the dispatcher, which imports this sibling).
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


# The reaper releases a terminal run's systemd scope only within this window
# after task_runs.ended_at; older residual scopes are left to operator-side,
# ledger-checked exact-name cleanup.
TERMINAL_WORKER_SCOPE_RELEASE_WINDOW_SECONDS = 1800


def _worker_unit_suffix(task_id: str, run_id: int) -> str:
    """The ``unit_suffix`` a run's worker scope is spawned under; shared by spawn and release."""
    return f"kanban-{task_id}-run-{run_id}"


def _worker_scope_unit_name(task_id: str, run_id: int) -> str:
    """The exact transient scope unit a kanban worker for this run was spawned in.

    Exact names only — never glob ``hermes-worker-*``: an operator blanket stop
    killed live workers (incident 2026-10-16). Must match the ``unit_suffix``
    ``_restart_safe_worker_argv`` spawns under (``_worker_unit_suffix``) plus the
    ``hermes-worker-<suffix>`` + ``.scope`` naming in
    ``tools.process_registry._systemd_scope_argv``. Upstream
    PR #139844 introduces ``systemd_scope_unit_name()``; switch this call site to
    it once that lands."""
    return f"hermes-worker-{_worker_unit_suffix(task_id, run_id)}.scope"


def _scope_unit_is_active(unit_name: str) -> bool:
    """True only when ``systemctl --user is-active`` reports ``active`` or
    ``activating``. No systemctl, any other state or any error is False: never
    stop what we cannot see."""
    import shutil

    from hermes_cli import kanban_db as _kb
    from tools.process_registry import systemd_user_bus_env

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--user", "is-active", unit_name],
            capture_output=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            env=systemd_user_bus_env(),
            check=False,
        )
    except Exception:  # health: allow BLE001 -- never stop what we cannot see: any probe fault reads as not active
        _kb._log.warning("kanban dispatch: systemctl is-active %s failed", unit_name, exc_info=True)
        return False
    return (result.stdout or b"").decode(errors="replace").strip() in ("active", "activating")


def _release_terminal_worker_scope(conn, task_id: str, run_id: int, *, now: int, ended_at: int) -> dict:
    """Stop the scope unit of a terminal run whose client is already gone (the
    kanban mirror of the cron-side release path added in #139844; stop as #70716).

    A ``--collect`` scope stays active while any straggler (a daemonised
    grandchild, a client hung after ``kanban_complete``) keeps its cgroup
    non-empty, so the reaper stops it by exact name. Best-effort and only within
    ``TERMINAL_WORKER_SCOPE_RELEASE_WINDOW_SECONDS`` of ``ended_at``: older
    residuals are left to operator-side, ledger-checked exact-name cleanup.
    Returns an audit dict and never raises."""
    from hermes_cli import kanban_db as _kb

    try:
        if now - ended_at > TERMINAL_WORKER_SCOPE_RELEASE_WINDOW_SECONDS:
            return {"skipped": "outside_window"}
        unit = _worker_scope_unit_name(task_id, run_id)
        if not _scope_unit_is_active(unit):
            return {"skipped": "not_active", "unit": unit}
        from tools.process_registry import _stop_systemd_unit

        if _stop_systemd_unit(unit):
            _kb._log.info("kanban dispatch: released worker scope %s (task %s)", unit, task_id)
            return {"released": True, "unit": unit}
        _kb._log.warning("kanban dispatch: failed to release worker scope %s (task %s)", unit, task_id)
        return {"released": False, "unit": unit}
    except Exception:  # health: allow BLE001 -- best-effort release must never abort the reaper row
        _kb._log.debug(
            "kanban dispatch: worker scope release failed for run %s (task %s)",
            run_id, task_id, exc_info=True,
        )
        return {"skipped": "error"}


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
            unit_suffix=f"kanban-{task.id}-run-missing",
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
        unit_suffix=_worker_unit_suffix(task.id, task.current_run_id),
        require_restart_safe_scope=True,
        outlives_parent=True,
    ).argv
