"""Identity and process-group ownership for Kanban worker reclamation."""

from __future__ import annotations

import os
import shlex

import psutil


def pid_matches_task_worker(pid: int, task_id: str) -> bool:
    from hermes_cli.update_cmd_windows import _hermes_holder_subcommand

    try:
        argv = psutil.Process(pid).cmdline()
    except (psutil.Error, OSError, ValueError):
        return False
    if _hermes_holder_subcommand(shlex.join(argv)) != "chat":
        return False
    query = f"work kanban task {task_id}"
    return any(
        flag in {"-q", "--query"} and value == query
        for flag, value in zip(argv, argv[1:])
    )


def claim_is_host_local(claim_lock, *, pid=None, task_id=None) -> bool:
    from hermes_cli import kanban_db as kb

    if not claim_lock:
        return False
    if str(claim_lock).startswith(kb._host_prefix()):
        return True
    return bool(pid and task_id and pid_matches_task_worker(int(pid), str(task_id)))


def signal_worker_tree(pid: int, sig: int) -> None:
    """Reach children in the worker's own session, never the gateway's group."""
    pid = int(pid)
    if os.name != "nt" and hasattr(os, "killpg"):
        try:
            pgid = os.getpgid(pid)
            if pgid == pid and pgid != os.getpgrp():
                os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only guarded path
                return
        except ProcessLookupError:
            if pid != os.getpgrp():
                os.killpg(pid, sig)  # windows-footgun: ok — POSIX-only guarded path
                return
        except OSError:
            pass
    os.kill(pid, sig)
