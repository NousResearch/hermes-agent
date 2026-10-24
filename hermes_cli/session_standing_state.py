"""Retire a session's standing state (``/goal`` and ``/loop``) when the session ends.

A goal and a loop are keyed by the session that owns them
(``goal:<session_id>`` / ``loop:<session_id>`` in ``SessionDB.state_meta``), so a
row outlives its session only as a frozen artifact: nothing reads it, but it
still reports ``status="active"`` and counts as live in every scan that looks
for standing work.

The explicit session boundary (``/new`` / ``/reset`` in the CLI, the gateway and
the TUI) is where that row must be retired. Compression is different: it mints a
child session for the *same* conversation, so ``migrate_goal_to_session`` carries
the goal across and archives the parent row (``hermes_cli/goals.py``). Here the
conversation is over, so the standing state goes with it.

``clear_goal`` / ``clear_loop`` are the existing primitives: they mark the row
``cleared`` and keep it for audit, which is what ``/goal clear`` and
``/loop stop`` already do.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Statuses that still describe live standing work. A row in any other state is
# already terminal and needs no retirement.
_LIVE_GOAL_STATUSES = frozenset({"active", "paused"})
_LIVE_LOOP_STATUSES = frozenset({"active", "paused"})


def retire_standing_state(session_id: str, *, reason: str = "session_reset") -> tuple[bool, bool]:
    """Retire ``session_id``'s standing ``/goal`` and ``/loop`` rows.

    Returns ``(goal_retired, loop_retired)``. Best-effort per row: a failure to
    clear one must never keep the session boundary from completing, so each
    clear is individually guarded and logged.
    """
    if not session_id:
        return (False, False)
    goal_retired = _retire_goal(session_id, reason)
    loop_retired = _retire_loop(session_id, reason)
    return (goal_retired, loop_retired)


def _retire_goal(session_id: str, reason: str) -> bool:
    try:
        from hermes_cli.goals import clear_goal, load_goal

        state = load_goal(session_id)
        if state is None or state.status not in _LIVE_GOAL_STATUSES:
            return False
        clear_goal(session_id)
        logger.debug("Retired standing goal for session %s (%s)", session_id, reason)
        return True
    except Exception:
        logger.debug("Failed to retire standing goal for session %s", session_id, exc_info=True)
        return False


def _retire_loop(session_id: str, reason: str) -> bool:
    try:
        from hermes_cli.loops import clear_loop, load_loop

        state = load_loop(session_id)
        if state is None or state.status not in _LIVE_LOOP_STATUSES:
            return False
        clear_loop(session_id)
        logger.debug("Retired standing loop for session %s (%s)", session_id, reason)
        return True
    except Exception:
        logger.debug("Failed to retire standing loop for session %s", session_id, exc_info=True)
        return False


def retired_summary(goal_retired: bool, loop_retired: bool) -> Optional[str]:
    """One-line user-facing notice when standing state was retired, else None."""
    parts = []
    if goal_retired:
        parts.append("standing /goal")
    if loop_retired:
        parts.append("standing /loop")
    if not parts:
        return None
    return " and ".join(parts) + " cleared for the previous session."
