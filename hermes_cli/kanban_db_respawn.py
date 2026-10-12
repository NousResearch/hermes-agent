"""Respawn-guard helpers: which events lift ``active_pr``, and the one notice a
subscriber gets when the guard keeps a ready card back for a long time.

``respawn_guarded`` is written every dispatch tick and is not a notifier kind,
so without ``respawn_held`` a held card sits in Ready with nobody told.
Origin-resident helpers are reached late-bound via ``_kb``, like the other
``kanban_db_*`` siblings.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

# Overridable via ``HERMES_KANBAN_HOLD_NOTIFY_SECONDS``; 0 disables.
DEFAULT_HOLD_NOTIFY_SECONDS = 600  # 10 minutes

# Guard reasons ``unblock_task`` lifts. The cooldowns lift themselves.
_HOLD_NOTIFY_REASONS = ("blocker_auth", "recent_success", "active_pr")


def lifts_active_pr(kind: str, payload: Optional[str]) -> bool:
    """Every kind ``check_respawn_guard`` queries lifts ``active_pr`` except an
    ``assigned`` event that does not hand the card to a DIFFERENT profile. A
    no-op re-assign (dev→dev via CLI/dashboard/``reassign --reclaim``), an
    unassign, or the dispatcher's own ``kanban.default_assignee`` write would
    otherwise lift ``active_pr`` for the very implementer that opened the PR.
    Events without ``from`` (written before it was recorded) are not trusted as
    handoffs — fail closed."""
    if kind != "assigned":
        return True
    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


def note_long_hold(conn: sqlite3.Connection, task_id: str, reason: str) -> None:
    """Append one ``respawn_held`` event once a ready card has been held for
    ``HERMES_KANBAN_HOLD_NOTIFY_SECONDS``. Runs in the txn that wrote this
    tick's ``respawn_guarded``. The hold is the run of ``respawn_guarded`` rows
    since the card's last other event; comments and the notice itself don't
    end it, so a held card is announced once per hold."""
    threshold = _kb._env_int("HERMES_KANBAN_HOLD_NOTIFY_SECONDS", DEFAULT_HOLD_NOTIFY_SECONDS)
    if threshold <= 0 or reason not in _HOLD_NOTIFY_REASONS:
        return
    hold = conn.execute(
        "SELECT MIN(created_at) AS since, MAX(kind = 'respawn_held') AS noted FROM task_events "
        "WHERE task_id = ? AND kind IN ('respawn_guarded', 'respawn_held') AND id > COALESCE("
        "  (SELECT MAX(id) FROM task_events WHERE task_id = ? "
        "   AND kind NOT IN ('respawn_guarded', 'respawn_held', 'commented')), 0)",
        (task_id, task_id),
    ).fetchone()
    if hold["since"] is None or hold["noted"]:
        return
    held_seconds = int(time.time()) - int(hold["since"])
    if held_seconds >= threshold:
        _kb._append_event(conn, task_id, "respawn_held", {"reason": reason, "held_seconds": held_seconds})


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
