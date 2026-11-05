"""Kanban review lifecycle: reopening a review card back to its implementer.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import sqlite3
import time


def reopen_review_task(
    conn: sqlite3.Connection, task_id: str, *, reason: str | None = None
) -> bool:
    """``review`` -> ``ready``/``todo`` so the implementer re-runs on the new
    comments; restores the implementer from the ``review_requested`` event.
    ``reason`` (already redacted by the caller) rides on the event payload so
    the notifier can surface it without a second write.
    Preserves ``consecutive_failures`` and the block loop counter (review is
    not a block; only :func:`complete_task` clears them)."""
    now = int(time.time())
    with _kb.write_txn(conn):
        _kb._reclaim_dangling_run(
            conn, task_id, statuses=("review",), now=now,
            note="invariant recovery on review reopen",
        )
        new_status = _kb._landing_status_after_parents(conn, task_id)
        review_event = _kb._latest_event(conn, task_id, "review_requested")
        handoff = _kb._json_dict(_kb._row_get(review_event, "payload"))
        implementer = _kb._nonblank_str(handoff.get("implementer"))
        return _kb._apply_review_reopen(conn, task_id, new_status, implementer, reason)


from hermes_cli import kanban_db as _kb
