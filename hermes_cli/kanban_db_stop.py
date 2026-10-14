"""Stop-note and ship-command guards for a live kanban worker.

A comment matching :func:`looks_like_stop_note` is an operator HARD STOP: the
kernel SIGTERMs the host-local worker the same way archive/reclaim already do,
parks the card in ``blocked`` (or archives on ``CLOSE``), and
:func:`claim_task` refuses to respawn until the hold lifts. Ordinary comments
are still steered mid-run; they do not cancel anything.

A task created with ``initial_status=blocked`` (R3 / human-ops hold) stays
blocked until ``unblock_task(..., force=True)``. A bare ``hermes kanban unblock``
— the path that released gated cards ~5 s after create — is not enough.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Optional

# Operator stop notes. Anchored / phrased so a progress comment mentioning
# "stopped" or "closed the file" does not trip it.
_STOP_NOTE_RE = re.compile(
    r"(?i)(?:^\s*HARD STOP\b|^\s*CLOSE\b|\bstop work here\b|\bstop the run\b|"
    r"\bdo not (?:merge|deploy|continue)\b)"
)
_CLOSE_NOTE_RE = re.compile(r"(?i)^\s*CLOSE\b")

_GH_MERGE_RE = re.compile(r"(?i)^(?:gh\s+pr\s+merge|gh\s+merge)\b")
_LAB_DEPLOY_RE = re.compile(r"(?i)^(?:\./)?lab\s+deploy\b")
_GIT_PUSH_RE = re.compile(r"(?i)^git\s+push\b")
_SHELL_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\||&)\s*")


def looks_like_stop_note(body: Optional[str]) -> bool:
    """True when ``body`` is an operator stop/close, not ordinary steering."""
    if not body or not str(body).strip():
        return False
    return bool(_STOP_NOTE_RE.search(str(body)))


def looks_like_close_note(body: Optional[str]) -> bool:
    """True when ``body`` is a CLOSE note (archive, not just block)."""
    if not body or not str(body).strip():
        return False
    return bool(_CLOSE_NOTE_RE.search(str(body)))


def looks_like_ship_command(command: Optional[str]) -> bool:
    """True when ``command`` would merge a PR, deploy, or push to main/master."""
    if not command or not str(command).strip():
        return False
    for part in _SHELL_SPLIT_RE.split(str(command)):
        s = part.strip()
        if not s:
            continue
        if _GH_MERGE_RE.match(s) or _LAB_DEPLOY_RE.match(s):
            return True
        if _GIT_PUSH_RE.match(s) and _git_push_targets_main(s):
            return True
    return False


def _git_push_targets_main(segment: str) -> bool:
    """True when a ``git push …`` segment names ``main`` or ``master`` as a ref."""
    tokens = segment.split()
    # ``git push`` [options] [remote] [refspec…] — skip the two-word prefix.
    for raw in tokens[2:]:
        if raw.startswith("-"):
            continue
        ref = raw.split(":", 1)[-1]  # HEAD:main / +refs/heads/main
        name = ref.rstrip("/").split("/")[-1]
        if name in {"main", "master"}:
            return True
    return False


def stop_note_holds(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when a stop-note comment is more recent than the last lift.

    A lift is a later non-stop comment or an ``unblocked`` event. Bare unblock
    is refused while this is true (see :func:`unblock_task`), so a 10:14-style
    empty unblock cannot respawn a card that still has ``Stop work here`` as
    its latest operator note.
    """
    from hermes_cli.kanban_db import list_comments

    last_stop_at: Optional[int] = None
    last_non_stop_at: Optional[int] = None
    for comment in list_comments(conn, task_id):
        stamp = int(comment.created_at)
        if looks_like_stop_note(comment.body):
            last_stop_at = stamp
        else:
            last_non_stop_at = stamp
    if last_stop_at is None:
        return False
    last_lift_at = last_non_stop_at
    row = conn.execute(
        "SELECT created_at FROM task_events "
        "WHERE task_id = ? AND kind = 'unblocked' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is not None:
        unblocked_at = int(row["created_at"])
        if last_lift_at is None or unblocked_at > last_lift_at:
            last_lift_at = unblocked_at
    if last_lift_at is None:
        return True
    # Unblock at the same second as the stop lifts it — tests and a human
    # `/kanban unblock --force` both land in that window. A later stop re-holds.
    return last_stop_at > last_lift_at


def initial_status_hold(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the newest block/unblock event is the create-time R3 hold."""
    from hermes_cli.kanban_db import _json_dict

    row = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row or row["kind"] != "blocked":
        return False
    return _json_dict(row["payload"]).get("reason") == "initial_status"


def apply_stop_note(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    comment_id: Optional[int] = None,
    body: Optional[str] = None,
    signal_fn=None,
) -> Optional[dict[str, Any]]:
    """Park ``task_id`` in ``blocked`` (or archive on CLOSE) and SIGTERM a live worker.

    Mirrors :func:`hermes_cli.kanban_db.archive_task`: the status flip and run
    close commit first so a losing concurrent writer never signals the pid; the
    kill runs after release because ``_poll_worker_exit`` must not hold the
    write lock. Ready/todo cards with no worker are just parked. Done/archived
    cards are left alone. Returns the termination dict when a worker was
    signalled, ``{}`` when the card was only parked, ``None`` when nothing
    applied.
    """
    from hermes_cli.kanban_db import (
        _append_event,
        _end_run,
        _terminate_reclaimed_worker,
        archive_task,
        write_txn,
    )

    if looks_like_close_note(body):
        return {} if archive_task(conn, task_id, signal_fn=signal_fn) else None

    with write_txn(conn, allow_nested=True):
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid, worker_started_at "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row or row["status"] in {"done", "archived"}:
            return None
        was_running = row["status"] == "running"
        prev_pid, prev_lock, prev_started = (
            row["worker_pid"], row["claim_lock"], row["worker_started_at"],
        )
        cur = conn.execute(
            "UPDATE tasks SET status = 'blocked', "
            "    claim_lock = NULL, claim_expires = NULL, "
            "    worker_pid = NULL, worker_started_at = NULL "
            "WHERE id = ? AND status NOT IN ('done', 'archived')",
            (task_id,),
        )
        if cur.rowcount != 1:
            return None
        run_id = None
        if was_running:
            run_id = _end_run(
                conn, task_id,
                outcome="cancelled", status="cancelled",
                summary="stop note cancelled the run",
            )
        _append_event(
            conn, task_id, "blocked",
            {
                "reason": "stop_note",
                "comment_id": comment_id,
                "status": "blocked",
            },
            run_id=run_id,
        )
    if not was_running:
        return {}
    termination = _terminate_reclaimed_worker(
        prev_pid, prev_lock, signal_fn=signal_fn, started_at=prev_started,
    )
    with write_txn(conn, allow_nested=True):
        _append_event(
            conn, task_id, "stop_worker_termination", termination, run_id=run_id,
        )
    return termination


def unblock_task(conn: sqlite3.Connection, task_id: str, *, force: bool = False) -> bool:
    """``blocked``/``scheduled`` -> its resumable phase (parent re-gated; ``review``
    when that is where it left off), closing any leaked run first.

    A card created with ``initial_status=blocked`` (R3 / human-ops hold) stays
    blocked unless ``force=True``. A bare unblock is how gated cards used to
    leak onto the dispatcher ~5 s after create. A stop note also refuses a
    bare unblock so CLOSE/HARD STOP cannot be undone by an empty unblock.
    """
    import time

    from hermes_cli.kanban_db import (
        _append_event,
        _landing_status_after_parents,
        _reclaim_dangling_run,
        _resume_status_from_events,
        _task_status,
        write_txn,
    )

    if not force and (initial_status_hold(conn, task_id) or stop_note_holds(conn, task_id)):
        return False
    now = int(time.time())
    with write_txn(conn):
        resume_status = (
            _resume_status_from_events(conn, task_id)
            if _task_status(conn, task_id) == "blocked"
            else "ready"
        )
        _reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        # Re-gate on parent completion before restoring the source phase.
        landing_status = _landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        # ``block_kind``/``block_recurrences`` deliberately survive the unblock:
        # resetting them is the amnesia that let cron-unblock <-> re-block loop
        # unbounded; only complete_task clears them. ``consecutive_failures``
        # (the dispatcher's spawn/crash counter) IS reset — a deliberate unblock
        # is a fresh start for the retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "unblocked",
            (
                {"status": new_status, "resume_status": resume_status}
                if new_status != "ready" or resume_status != "ready"
                else None
            ),
        )
        return True
