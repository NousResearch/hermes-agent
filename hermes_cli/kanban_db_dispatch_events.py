"""Durable, rate-bounded diagnostics for repeated dispatcher holds."""

from __future__ import annotations

import sqlite3
import time

_RESPAWN_GUARD_REFRESH_SECONDS = 24 * 60 * 60
_TRANSITION_FIELDS = {
    "assigned": ("from", "assignee"),
    "status": ("previous_status", "status"),
}


def record_respawn_guard(conn: sqlite3.Connection, task_id: str, reason: str) -> None:
    """Record entry/reason changes immediately, refreshing continuous holds daily.

    Lifecycle transitions end an episode even when the task returns between
    ticks. Comments, attachments and heartbeats do not: polling a held PR must
    not manufacture a new episode. The check and append share the writer lock.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        # The time range uses idx_events_task and avoids walking all history.
        # Event IDs break same-second ties; reconnects share this durable state.
        recent = conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? AND created_at > ? "
            "AND kind IN ('respawn_guarded', 'status', 'claimed', 'assigned', "
            "'promoted', 'promoted_manual', 'unblocked', 'reclaimed', 'reconciled', "
            "'review_requested', 'changes_requested', 'review_reopened', "
            "'blocked', 'dependency_wait', 'block_loop_detected', 'scheduled', "
            "'completed', 'archived', 'specified', 'descendant_invalidated', "
            "'crashed', 'stale', 'timed_out', 'spawn_failed', 'gave_up') "
            "ORDER BY id DESC",
            (task_id, int(time.time()) - _RESPAWN_GUARD_REFRESH_SECONDS),
        )
        for previous in recent:
            payload = kb._json_dict(previous["payload"])
            fields = _TRANSITION_FIELDS.get(previous["kind"])
            if fields and all(key in payload for key in fields) and payload[fields[0]] == payload[fields[1]]:
                # Idempotent writes do not end a hold. Keep looking: a real
                # transition may precede the no-op. Legacy unknowns reset it.
                continue
            if previous["kind"] == "respawn_guarded" and payload.get("reason") == reason:
                return
            break
        kb._append_event(conn, task_id, "respawn_guarded", {"reason": reason})
