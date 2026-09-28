"""Detached supervisor for a configured Kanban pre-spawn command.

The dispatcher starts this process quickly while holding its board lock.  This
process runs the potentially slow external gate with a deliberately small,
non-secret environment and only then replaces itself with the real worker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any


_GATE_ENV_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SYSTEMROOT",
    "TMPDIR",
    "WINDIR",
)

_SQLITE_MAX_INTEGER = 2**63 - 1
_BLOCK_RETRY_INITIAL_SECONDS = 0.25
_BLOCK_RETRY_MAX_SECONDS = 5.0
_OWNERSHIP_HANDSHAKE_SECONDS = 5.0
_OWNERSHIP_RETRY_SECONDS = 0.02


def _valid_run_id(value: Any) -> bool:
    """Return whether *value* is a canonical SQLite run primary key."""
    return type(value) is int and 0 < value <= _SQLITE_MAX_INTEGER


def _gate_env(source: dict[str, str]) -> dict[str, str]:
    """Return the non-secret process environment exposed to the gate."""
    return {key: source[key] for key in _GATE_ENV_KEYS if key in source}


def _block_launch(request: dict[str, Any], reason: str) -> bool:
    """Durably stop this run so a denied/uncertain gate is never retried."""
    run_id = request.get("run_id") if isinstance(request, dict) else None
    if not _valid_run_id(run_id):
        print(
            "[kanban_spawn_supervisor] invalid request: run_id must be a "
            "positive SQLite integer",
            file=sys.stderr,
            flush=True,
        )
        return False
    retry_delay = _BLOCK_RETRY_INITIAL_SECONDS
    while True:
        try:
            from hermes_cli import kanban_db

            with kanban_db.connect(board=request.get("board")) as conn:
                # The dispatcher may still hold its own connection when this
                # runs; give SQLite enough runway to wait for the write lock.
                try:
                    conn.execute("PRAGMA busy_timeout = 30000")
                except Exception:
                    pass
                result = kanban_db.block_task(
                    conn,
                    str(request["task_id"]),
                    reason=reason,
                    kind="capability",
                    expected_run_id=run_id,
                )
                print(
                    "[kanban_spawn_supervisor] "
                    f"block_task({request.get('task_id')!r}) → {result}",
                    flush=True,
                )
                if result:
                    return True

                # A failed compare-and-swap is final only when this run has
                # already been superseded or otherwise resolved.  If the same
                # run is still active, retain ownership and retry rather than
                # exiting and letting dead-PID recovery launch the gate again.
                task = kanban_db.get_task(conn, str(request["task_id"]))
                if (
                    task is None
                    or task.current_run_id != run_id
                    or task.status not in {"running", "ready"}
                ):
                    return False
        except Exception as exc:
            print(
                "[kanban_spawn_supervisor] _block_launch failed; retrying: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, _BLOCK_RETRY_MAX_SECONDS)


def _still_owns_run(request: dict[str, Any]) -> bool:
    """Verify this supervisor still owns the fenced run before worker exec."""
    run_id = request.get("run_id") if isinstance(request, dict) else None
    if not _valid_run_id(run_id):
        return False
    deadline = time.monotonic() + _OWNERSHIP_HANDSHAKE_SECONDS
    try:
        from hermes_cli import kanban_db

        while True:
            with kanban_db.connect(board=request.get("board")) as conn:
                task = kanban_db.get_task(conn, str(request["task_id"]))
            if (
                task is None
                or task.status != "running"
                or task.current_run_id != run_id
            ):
                return False
            if task.worker_pid == os.getpid():
                return True
            # Popen returns in the dispatcher before its subsequent DB write.
            # An approving gate can therefore finish while worker_pid is still
            # NULL. Wait only for that one known handshake state; any other PID
            # means ownership has moved and must fail immediately.
            if task.worker_pid is not None or time.monotonic() >= deadline:
                return False
            time.sleep(_OWNERSHIP_RETRY_SECONDS)
    except Exception as exc:
        print(
            "[kanban_spawn_supervisor] ownership check failed; launch denied: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False


def supervise(
    gate_command: list[str],
    timeout_seconds: float,
    request: dict[str, Any],
    worker_command: list[str],
) -> int:
    """Run one gate and exec the worker only after an explicit zero exit."""
    try:
        completed = subprocess.run(
            gate_command,
            input=json.dumps(request),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_gate_env(dict(os.environ)),
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _block_launch(request, "pre-spawn command outcome uncertain (timed out); launch denied")
        return 78
    except OSError:
        _block_launch(request, "pre-spawn command could not start; launch denied")
        return 78
    if completed.returncode != 0:
        _block_launch(request, "pre-spawn command rejected worker launch")
        return 78
    if not _still_owns_run(request):
        print(
            "[kanban_spawn_supervisor] run ownership changed or could not be "
            "confirmed; launch denied",
            file=sys.stderr,
            flush=True,
        )
        _block_launch(
            request,
            "pre-spawn approval could not confirm supervisor ownership; launch denied",
        )
        return 78
    try:
        os.execvpe(worker_command[0], worker_command, os.environ)
    except OSError:
        _block_launch(request, "worker could not start after pre-spawn approval")
    return 78  # pragma: no cover - exec only returns on failure


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4:
        return 64
    gate_command = json.loads(args[0])
    timeout_seconds = float(args[1])
    request = json.loads(args[2])
    worker_command = json.loads(args[3])
    if not gate_command or not worker_command:
        return 64
    run_id = request.get("run_id") if isinstance(request, dict) else None
    if not _valid_run_id(run_id):
        print(
            "[kanban_spawn_supervisor] invalid request: run_id must be a "
            "positive SQLite integer",
            file=sys.stderr,
            flush=True,
        )
        return 64
    return supervise(gate_command, timeout_seconds, request, worker_command)


if __name__ == "__main__":
    raise SystemExit(main())
