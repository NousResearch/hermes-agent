"""Synchronous end for workers displaced by a terminal close.

``archive_task`` (#76196) and ``complete_task`` (2026-11-05 zombie-writer
incident, fleet card t_2da8548a) must not leave the card's executor running
behind: a supersede/force close is pure bookkeeping in the closer's process,
while the displaced worker keeps writing into a workspace nothing tracks
anymore. The deferred backstop (``kanban_db_dispatch.reap_terminal_workers``,
#111791) only sees closed run rows whose pid evidence survived, and only fires
on a dispatcher tick — this module is the synchronous path, called post-commit
so the ~5 s TERM→KILL poll never holds the write lock.

The signalling safety rails live in ``_terminate_reclaimed_worker``: host-local
claims only, never a recycled PID (spawn-fingerprint match), never an
UNVERIFIED spawn. On top of those, ``complete_task`` must never signal the
worker that closed its OWN card — the normal completion path runs inside the
worker process, and killing it there would cut the post-commit finalisation
(lifecycle hook, gateway notification, session persist) that the reaper's grace
window exists to protect.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Optional


def close_initiated_by_own_worker(task_id: str, worker_pid: Optional[int]) -> bool:
    """True when the closing process IS the card's worker (or its descendant).

    The in-process ``kanban_complete`` tool runs under the worker's own pid; a
    worker that closes its card via ``hermes kanban complete`` in its terminal
    tool runs the CLI as a child that inherits ``HERMES_KANBAN_TASK``, so the
    env check covers the whole process tree without walking it.
    """
    if worker_pid and int(worker_pid) == os.getpid():
        return True
    return (os.environ.get("HERMES_KANBAN_TASK") or "").strip() == task_id


def terminate_displaced_worker(
    conn: sqlite3.Connection,
    task_id: str,
    task_row: Any,
    *,
    event_kind: str,
    run_id: Optional[int] = None,
    signal_fn=None,
    own_worker_guard: bool = True,
) -> Optional[dict]:
    """SIGTERM (then SIGKILL) the worker a terminal close displaced; record it.

    ``task_row`` carries the pre-close ``claim_lock`` / ``worker_pid`` /
    ``worker_started_at`` (callers snapshot them inside the close txn, before
    the UPDATE wipes the task row). Returns the termination info recorded as
    ``event_kind``, or None when there was no worker to displace or the closer
    IS the worker (``own_worker_guard``). A cross-host claim is recorded but
    not signalled — only the worker's own host may kill it, and that host's
    dispatcher sweep owns the deferred kill.
    """
    if task_row is None:
        return None
    pid = task_row["worker_pid"]
    if not pid:
        return None
    if own_worker_guard and close_initiated_by_own_worker(task_id, int(pid)):
        return None
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import _terminate_reclaimed_worker
    termination = _terminate_reclaimed_worker(
        pid, task_row["claim_lock"], signal_fn=signal_fn,
        started_at=task_row["worker_started_at"],
    )
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, event_kind, termination, run_id=run_id)
    return termination
