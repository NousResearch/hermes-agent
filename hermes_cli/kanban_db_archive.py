"""Archiving a task, and withdrawing a card its own filer created by mistake.

Split out of ``hermes_cli.kanban_db``; facade helpers are late-imported inside the
functions, so monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def archive_task(
    conn: sqlite3.Connection, task_id: str, *, signal_fn=None,
    refuse_statuses: tuple[str, ...] = (),
) -> bool:
    """Archive a task; a *running* task's host-local worker is terminated.

    ``refuse_statuses`` makes the transition conditional on the row NOT being in
    one of those statuses at write time (compare-and-set in the archive txn), so
    a caller that must not overwrite e.g. ``done`` cannot race a concurrent
    completion. A refused row returns False with no side effects.

    Clearing ``worker_pid`` in the DB alone left the OS process running past its
    own archive — it kept executing (and pushing work) against a task nothing
    tracked anymore (#76196). Snapshot pid+claim inside the archive txn so the
    kill is contingent on THIS caller winning the archive transition (a losing
    concurrent archiver must never signal the pid); the kill itself runs after
    commit — ``_poll_worker_exit`` can wait ~5 s and must not hold the write
    lock. Post-release kill is safe here because ``archived`` is terminal: no
    dispatcher can spawn a duplicate worker off the released claim. The
    termination outcome lands as its own ``archive_worker_termination`` event so
    the ``archived`` event stays atomic with the status flip.
    """
    from hermes_cli.kanban_db import (
        _append_event,
        _cleanup_workspace,
        _end_run,
        _terminate_reclaimed_worker,
        recompute_ready,
        write_txn,
    )

    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid, worker_started_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return False
        was_running = row["status"] == "running"
        prev_pid, prev_lock, prev_started = row["worker_pid"], row["claim_lock"], row["worker_started_at"]
        excluded = ("archived", *refuse_statuses)
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', "
            "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL "
            f"WHERE id = ? AND status NOT IN ({', '.join('?' * len(excluded))})",
            (task_id, *excluded),
        )
        if cur.rowcount != 1:
            return False
        # Archived mid-run (dashboard): close the run so history isn't orphaned.
        run_id = _end_run(
            conn, task_id, outcome="reclaimed", status="reclaimed",
            summary="task archived with run still active",
        )
        _append_event(conn, task_id, "archived", None, run_id=run_id)
    if was_running:
        termination = _terminate_reclaimed_worker(prev_pid, prev_lock, signal_fn=signal_fn, started_at=prev_started)
        with write_txn(conn):
            _append_event(conn, task_id, "archive_worker_termination", termination, run_id=run_id)
    # ``archived`` parents no longer block children; promote them now.
    recompute_ready(conn)
    # Reap the workspace on archive too (never-completed tasks kept it forever).
    _cleanup_workspace(conn, task_id)
    return True


_WITHDRAW_TERMINAL = ("done", "archived")


class WithdrawRefused(ValueError):
    """``withdraw_task`` refused: no such card, already terminal, or filed by
    another run. A ``ValueError`` so tool error handlers report it as a
    recoverable refusal rather than a traceback."""


def withdraw_task(
    conn: sqlite3.Connection, task_id: str, *, actor: Optional[str] = None,
    reason: Optional[str] = None, creator_task_id: Optional[str] = None, signal_fn=None,
) -> bool:
    """Retract a card its filer filed by mistake; lands the terminal ``archived`` state.

    The missing piece was authorisation, not the transition. A card is filed by
    ONE run and, in the reported case, retracted by that same run seconds later —
    but that run is scoped to the card it is *working* (``HERMES_KANBAN_TASK``),
    so every lifecycle verb refuses to touch the card it just created (a guard
    that must stay: a prompt-injected ``task_id`` must not corrupt sibling runs),
    and the filer can only watch the dispatcher claim its own duplicate.
    ``archive_task`` already lands the state — terminating a host-local worker
    mid-run, closing the run, reaping the workspace, and promoting the children
    an archived parent no longer gates (#76196) — so this adds the proof of
    authorship, not a second transition.

    ``creator_task_id`` is that proof: the calling run's OWN task id, matched
    against the provenance ``create_task`` stamped at insert time (never against
    a caller-supplied value, which would let any worker retract another run's
    card). ``None`` = no provenance required — the CLI/dashboard operator path,
    exactly as ``archive`` behaves today.

    Refuses rather than archives when the card is already ``done``/``archived``
    (a finished duplicate is already out of dispatch, and its record is the
    evidence of what happened) or when the provenance does not match.

    Returns True on the transition; raises ``WithdrawRefused`` otherwise.
    """
    from hermes_cli.kanban_db import (
        _append_event,
        get_task,
        write_txn,
    )

    def _finished(status: str) -> WithdrawRefused:
        return WithdrawRefused(
            f"{task_id} is {status}: nothing left to withdraw (a finished card is "
            "already out of dispatch; archive it from the CLI/dashboard if it must leave "
            "the board)")

    task = get_task(conn, task_id)
    if task is None:
        raise WithdrawRefused(f"unknown task: {task_id}")
    if task.status in _WITHDRAW_TERMINAL:
        raise _finished(task.status)
    if creator_task_id is not None and task.creator_task_id != creator_task_id:
        raise WithdrawRefused(
            f"{task_id} was not filed by {creator_task_id} "
            f"(provenance: {task.creator_task_id or 'none recorded'}). Only the run that "
            "filed a card may withdraw it: ask its filer, or archive it from the "
            "CLI/dashboard (`hermes kanban archive`)")
    # The read above is advisory: a concurrent completion can land before the
    # archive, so the refusal is re-applied as a CAS inside the archive txn.
    if not archive_task(conn, task_id, signal_fn=signal_fn, refuse_statuses=("done",)):
        current = get_task(conn, task_id)
        if current is None:
            raise WithdrawRefused(f"unknown task: {task_id}")
        raise _finished(current.status)
    # After the transition, so a losing concurrent withdrawer cannot leave a
    # `withdrawn` event on a card it never moved.
    with write_txn(conn):
        _append_event(
            conn, task_id, "withdrawn",
            {"actor": actor, "reason": reason,
             "provenance": creator_task_id or "operator", "status": "archived"},
        )
    return True
