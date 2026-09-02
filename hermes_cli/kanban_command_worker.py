"""Supervisor for direct-command kanban workers (``worker.command``).

The dispatcher spawns this module (``python -P -m
hermes_cli.kanban_command_worker`` — the ``-P`` matters: the supervisor
runs with the task's workspace as its cwd, and without it Python would
put that workspace first on ``sys.path``, letting workspace content
shadow this very module) instead of the command itself. It runs the
declared argv as its direct child, waits for it, and reports the exit
code through the same canonical transitions a well-behaved agent worker
makes — while this process is still alive, so no reaper ever has to
guess what happened:

* rc=0  → ``complete_task`` (unless the command already moved the task
  itself through a canonical channel — then the transition is a no-op and
  the command's own word stands)
* rc!=0 → ``block_task`` with the code (or signal) in the reason.
  Deliberately no retry: a deterministic pipeline that failed is a fact
  for a human, and any retry loop belongs to the pipeline itself.
  Unblocking re-runs it.

Being the command's direct parent is the whole point: the exit code is
observed in every dispatcher constitution (resident gateway, ``kanban
daemon``, throwaway cron ticks), the task leaves ``running`` before this
process exits so stale-claim reclaim can never race the report, and no
worker-kind bookkeeping has to be reconstructed at reap time.

Everything this process needs arrives in the environment the dispatcher
built (host-side values only — nothing here is card-controlled):

* ``HERMES_KANBAN_WORKER_COMMAND`` — JSON argv list, resolved once at
  spawn time from the assignee profile's config
* ``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_RUN_ID`` — the task to report on
* ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` — the board to report to
* ``HERMES_KANBAN_WORKSPACE`` — the working directory (already the cwd)

Termination: the child runs in its own session (process group), so a
SIGTERM here — ``enforce_max_runtime``'s first shot — is forwarded to
the whole group, and after a short grace the group is SIGKILLed. The
handler only forwards and returns; the grace is enforced by the main
wait loop (calling ``Popen.wait`` inside the handler would deadlock
against the main thread's own wait and burn the whole grace, observed in
review). The grace must stay SHORTER than the ~5 seconds
``enforce_max_runtime`` waits before SIGKILLing the supervisor itself,
or the report below never runs and the child is orphaned. If the
supervisor is SIGKILLed anyway, it said nothing — the ordinary crash
path covers the task without inventing a result.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

# Must stay shorter than enforce_max_runtime's term-to-kill window (~5s in
# hermes_cli.kanban_db) — see the module docstring. Overridable for tests
# and unusual pipelines via HERMES_KANBAN_COMMAND_TERM_GRACE (seconds).
_TERM_GRACE_SECONDS = 3.0
_WAIT_POLL_SECONDS = 0.2


def _fail(message: str) -> "int":
    print(f"kanban_command_worker: {message}", file=sys.stderr, flush=True)
    return 1


def _term_grace() -> float:
    raw = os.environ.get("HERMES_KANBAN_COMMAND_TERM_GRACE", "")
    try:
        value = float(raw)
        if value > 0:
            # Clamp: a grace at or beyond enforce_max_runtime's ~5s
            # term-to-kill window would mean the report never runs and the
            # child is orphaned — the exact failure the grace exists to
            # prevent. 4.0 keeps ~1s of reporting headroom.
            return min(value, 4.0)
    except ValueError:
        pass
    return _TERM_GRACE_SECONDS


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except OSError:
        pass


def main() -> int:
    raw_argv = os.environ.get("HERMES_KANBAN_WORKER_COMMAND", "")
    task_id = os.environ.get("HERMES_KANBAN_TASK", "")
    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "")
    if not raw_argv or not task_id:
        return _fail("missing HERMES_KANBAN_WORKER_COMMAND / HERMES_KANBAN_TASK")
    try:
        argv = json.loads(raw_argv)
    except ValueError:
        return _fail("HERMES_KANBAN_WORKER_COMMAND is not valid JSON")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(part, str) and part for part in argv)
    ):
        return _fail("HERMES_KANBAN_WORKER_COMMAND must be a JSON list of strings")
    try:
        run_id = int(raw_run_id) if raw_run_id else None
    except ValueError:
        run_id = None

    try:
        # Own session: SIGTERM/SIGKILL forwarding below reaches the whole
        # process group, grandchildren included, not just the direct child.
        child = subprocess.Popen(argv, start_new_session=True)  # noqa: S603 -- host-side config
    except FileNotFoundError:
        return _fail(f"worker.command executable not found: {argv[0]!r}")
    except OSError as exc:
        return _fail(f"worker.command could not start: {exc}")

    term_at: "list[float]" = []

    def _forward_term(_signum, _frame):
        # Forward and return. No waiting in the handler: the handler runs
        # on the main thread, whose own Popen.wait holds the waitpid lock —
        # a timeout wait here can never acquire it and would just burn the
        # whole grace (found in review). The main loop below enforces the
        # grace and the SIGKILL escalation.
        if not term_at:
            term_at.append(time.monotonic())
            _signal_group(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, _forward_term)
    signal.signal(signal.SIGINT, _forward_term)

    grace = _term_grace()
    returncode = None
    while returncode is None:
        try:
            returncode = child.wait(timeout=_WAIT_POLL_SECONDS)
        except subprocess.TimeoutExpired:
            if term_at and time.monotonic() - term_at[0] >= grace:
                _signal_group(child.pid, signal.SIGKILL)
                # No timeout: after a group SIGKILL only an uninterruptible
                # (D-state) child can stall this, and then the dispatcher's
                # own SIGKILL ends the supervisor — crash path, safe side.
                returncode = child.wait()

    # Report through the canonical transitions. Everything from here on is
    # inside the try: an unreportable exit must end in _fail (message in
    # the task log), leaving the task to the ordinary crash path.
    try:
        from hermes_cli import kanban_db as kb

        with kb.connect() as conn:
            if returncode == 0:
                kb.complete_task(
                    conn,
                    task_id,
                    summary="worker command exited 0",
                    metadata={"exit_code": 0, "worker_kind": "command"},
                    expected_run_id=run_id,
                )
            else:
                if returncode < 0:
                    reason = (
                        f"worker command terminated by signal {-returncode}"
                    )
                else:
                    reason = f"worker command exited with code {returncode}"
                kb.block_task(
                    conn,
                    task_id,
                    reason=reason,
                    expected_run_id=run_id,
                )
    except Exception as exc:
        return _fail(f"could not report exit {returncode} for {task_id}: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
