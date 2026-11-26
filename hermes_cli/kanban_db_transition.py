"""Armed-board release plumbing for the kanban transition writers.

The transition functions (``unblock_task``, ``promote_task``, the dashboard's direct status
write) live in modules that are over their line caps, so their guard hooks live here and
are re-exported from ``hermes_cli.kanban_db``. Everything is late-bound through
:mod:`hermes_cli.kanban_transition_guard`, which imports ``kanban_db`` back at call time.
"""

from __future__ import annotations

import os
import secrets
import sqlite3


def _ktg():
    """The board transition guard (lazy: it imports this module back at call time)."""
    from hermes_cli import kanban_transition_guard as ktg

    return ktg


def _transition_release(conn: sqlite3.Connection, task_id: str, interpretation: str):
    """The release an armed board demands for this transition, or ``None`` (guard open)."""
    return _ktg().release_request_for(conn, task_id, interpretation)


def _spend_release(conn: sqlite3.Connection, request) -> None:
    """Spend the release found before the write txn; no-op when the guard is open.

    Inside the caller's txn, so the spend and the flip commit together — and a caller
    that loses the status CAS leaves the release for the one that does change the row.
    """
    if request is not None:
        _ktg().consume_release(conn, request)


def require_armed_board_release(conn: sqlite3.Connection, task_id: str, interpretation: str) -> None:
    """Public check for writers that flip status themselves (dashboard drag-drop/direct
    status write). Raises ``BoardTransitionRefused`` when an armed board has no release;
    the caller spends it with :func:`spend_armed_board_release` once the row is known to
    be changeable."""
    request = _transition_release(conn, task_id, interpretation)
    if request is not None:
        _ktg().refuse_without_release(conn, request)


def spend_armed_board_release(conn: sqlite3.Connection, task_id: str, interpretation: str) -> None:
    """Spend the release :func:`require_armed_board_release` found; no-op when not armed.
    Must run inside the writer's own ``write_txn`` so the spend commits with the flip."""
    _spend_release(conn, _transition_release(conn, task_id, interpretation))

def _new_task_id() -> str:
    """``t_`` + 4 hex bytes (collision ~1e-3 at 100k tasks; 2 bytes would hit 50%
    by 10k). Idempotency belongs to ``idempotency_key``, not id uniqueness."""
    return "t_" + secrets.token_hex(4)


def _claimer_id() -> str:
    """Return a ``host:pid`` string that identifies this claimer."""
    import socket
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


def _host_prefix() -> str:
    """``"<host>:"`` prefix shared by every claim lock issued from this host."""
    return f"{_claimer_id().split(':', 1)[0]}:"


def _resume_status_from_events(conn: sqlite3.Connection, task_id: str) -> str:
    """``review`` when the newest lifecycle event carries a review
    ``resume_status``/``retry_status``/``source_status``, else ``ready`` (legacy)."""
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind IN ("
        "'blocked', 'block_loop_detected', 'dependency_wait', 'gave_up', "
        "'unblocked', 'changes_requested', 'review_reopened', 'status', 'reclaimed', "
        "'stale', 'timed_out', 'crashed', 'spawn_failed', 'rate_limited'"
        ") ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    from hermes_cli.kanban_db import _json_dict, _row_get  # late-bound origin helpers

    payload = _json_dict(_row_get(row, "payload"))
    for key in ("resume_status", "retry_status", "source_status"):
        if payload.get(key) == "review":
            return "review"
    return "ready"
