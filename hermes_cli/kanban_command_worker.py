"""Run a profile-owned fixed argv as a Kanban worker.

The dispatcher starts this module as the recorded worker PID.  The declared
argv is carried in host-built environment state, while the command itself is
the direct child and inherits the task workspace as its cwd.  A normal exit is
reported through the existing canonical transitions before the supervisor
exits.  A supervisor termination is deliberately not reported: the dispatcher
owns timeout/retry/failure accounting and uses ``expected_run_id`` as the
stale-run fence.
"""

from __future__ import annotations

import json
import logging
import ntpath
import os
import signal
import subprocess
import sys
import time


_TERM_GRACE_SECONDS = 3.0
_WAIT_POLL_SECONDS = 0.2
_WORKER_COMPLETION_MODE_ENV = "HERMES_KANBAN_WORKER_COMPLETION_MODE"
_WORKER_COMPLETION_MODES = frozenset({"exit_code", "self_reported"})
_SELF_REPORTED_PROTOCOL_REASON = "worker_protocol_transition_required"
_log = logging.getLogger(__name__)
_CHILD_KANBAN_ENV = frozenset(
    {
        "HERMES_KANBAN_TASK_ID",
        # Keep the established spelling for existing command integrations.
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    }
)


def _fail(message: str) -> int:
    _log.error("kanban_command_worker: %s", message)
    return 1


def _term_grace() -> float:
    raw = os.environ.get("HERMES_KANBAN_COMMAND_TERM_GRACE", "")
    try:
        value = float(raw)
    except ValueError:
        value = _TERM_GRACE_SECONDS
    return min(value, 4.0) if value > 0 else _TERM_GRACE_SECONDS


def _terminate_child_tree(child: subprocess.Popen, *, force: bool = False) -> None:
    """Terminate the command tree without relying on POSIX-only APIs."""
    from hermes_cli import _subprocess_compat as compat

    if compat.IS_WINDOWS:
        # ``os.kill`` cannot address descendants on Windows.  The shared
        # helper uses taskkill /T and therefore remains safe if the supervisor
        # receives a direct termination request there.
        compat.kill_process_tree(child)
        return
    compat.terminate_process_tree(child.pid, force=force)


def _child_env() -> dict[str, str]:
    """Re-sanitize the supervisor environment before arbitrary code runs."""
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    for key in list(env):
        if key.startswith("HERMES_KANBAN_") and key not in _CHILD_KANBAN_ENV:
            env.pop(key, None)
    # The supervisor is an implementation detail, not command input.
    for key in _CHILD_KANBAN_ENV:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _absolute_argv0(value: str) -> bool:
    if os.path.isabs(value):
        return True
    return os.name == "nt" and ntpath.isabs(value)


def _observe_self_reported_exit(conn, kb, task_id: str, run_id: int | None) -> None:
    """Preserve a worker-owned transition or fail closed on no transition."""
    if run_id is None:
        raise RuntimeError(
            "self_reported completion requires a valid expected_run_id"
        )
    task = kb.get_task(conn, task_id)
    run = kb.get_run(conn, run_id)
    if task is None or run is None or run.task_id != task_id:
        raise RuntimeError(
            f"self_reported completion could not inspect task/run {task_id}/{run_id}"
        )

    # A durable outcome is authoritative even if a concurrent observer has not
    # yet made the task pointer/status look terminal. Never replace it with the
    # supervisor's generic exit metadata.
    if run.outcome is not None:
        _log.info(
            "kanban_command_worker: preserving self-reported outcome=%s for %s run=%s",
            run.outcome,
            task_id,
            run_id,
        )
        return

    # A changed pointer or status means this supervisor is stale (or another
    # actor won the transition race). The expected-run CAS must remain the only
    # authority; do not create a transition for a later run.
    if task.status != "running" or task.current_run_id != run_id:
        _log.info(
            "kanban_command_worker: preserving non-active self-reported run for %s run=%s",
            task_id,
            run_id,
        )
        return

    # The fixed command exited successfully without making a canonical
    # transition. Turn that protocol violation into a durable non-success
    # outcome instead of falsely marking the task done.
    if not kb.block_task(
        conn,
        task_id,
        reason=_SELF_REPORTED_PROTOCOL_REASON,
        expected_run_id=run_id,
    ):
        _log.info(
            "kanban_command_worker: self-reported protocol transition raced for %s run=%s",
            task_id,
            run_id,
        )


def main() -> int:
    raw_argv = os.environ.get("HERMES_KANBAN_WORKER_COMMAND", "")
    task_id = os.environ.get("HERMES_KANBAN_TASK_ID") or os.environ.get(
        "HERMES_KANBAN_TASK", ""
    )
    if not raw_argv or not task_id:
        return _fail("missing worker command or Kanban task id")
    try:
        argv = json.loads(raw_argv)
    except (TypeError, ValueError):
        return _fail("HERMES_KANBAN_WORKER_COMMAND is not valid JSON")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(part, str) and part for part in argv)
    ):
        return _fail("worker command must be a JSON list of non-empty strings")
    if not _absolute_argv0(argv[0]):
        return _fail("worker command argv[0] must be an absolute executable path")
    completion_mode = os.environ.get(_WORKER_COMPLETION_MODE_ENV, "exit_code")
    if completion_mode not in _WORKER_COMPLETION_MODES:
        return _fail(
            f"{_WORKER_COMPLETION_MODE_ENV} must be one of "
            f"{sorted(_WORKER_COMPLETION_MODES)}"
        )

    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "")
    try:
        run_id = int(raw_run_id) if raw_run_id else None
    except ValueError:
        run_id = None

    from hermes_cli import _subprocess_compat as compat

    try:
        child = subprocess.Popen(
            argv,
            env=_child_env(),
            **compat.windows_detach_popen_kwargs(),
        )
    except FileNotFoundError:
        return _fail(f"worker command executable not found: {argv[0]!r}")
    except OSError as exc:
        return _fail(f"worker command could not start: {exc}")

    terminated = False
    term_at: list[float] = []

    def _forward_term(_signum, _frame):
        nonlocal terminated
        if not term_at:
            terminated = True
            term_at.append(time.monotonic())
            _terminate_child_tree(child)

    signal.signal(signal.SIGTERM, _forward_term)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _forward_term)

    returncode: int | None = None
    while returncode is None:
        try:
            returncode = child.wait(timeout=_WAIT_POLL_SECONDS)
        except subprocess.TimeoutExpired:
            if term_at and time.monotonic() - term_at[0] >= _term_grace():
                _terminate_child_tree(child, force=True)
                returncode = child.wait()

    if terminated:
        # No canonical transition here.  In particular, a dispatcher timeout
        # must remain ``timed_out`` and keep its retry/failure-counter owner.
        return 128 + int(getattr(signal, "SIGTERM", 15))

    try:
        from hermes_cli import kanban_db as kb

        with kb.connect() as conn:
            if returncode == 0:
                if completion_mode == "self_reported":
                    _observe_self_reported_exit(conn, kb, task_id, run_id)
                else:
                    kb.complete_task(
                        conn,
                        task_id,
                        summary="worker command exited 0",
                        metadata={"exit_code": 0, "worker_kind": "command"},
                        expected_run_id=run_id,
                    )
            else:
                if returncode < 0:
                    reason = f"worker command terminated by signal {-returncode}"
                else:
                    reason = f"worker command exited with code {returncode}"
                kb.block_task(conn, task_id, reason=reason, expected_run_id=run_id)
    except Exception as exc:
        return _fail(f"could not report exit {returncode} for {task_id}: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
