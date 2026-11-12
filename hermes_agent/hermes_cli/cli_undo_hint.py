"""Classic CLI ``/undo``: the read-only line naming the files the undone turns changed and pointing at ``/rollback``.

The in-process CLI asks its agent's checkpoint manager over the same cwd its ``/rollback`` lists; a
CLI rewired onto the shared gateway asks the owner (``rollback.list`` with ``undo_since``), whose
tools took the checkpoints. Either way the line is advisory: any failure prints nothing.
"""
from __future__ import annotations

import contextlib
import os


def _undone_span(before: list, turns_undone: int):
    """``(target stamp, kept prefix)`` of the oldest undone user turn in the pre-undo history."""
    from hermes_agent.hermes_cli.cli_session_mixin import _user_turn_indices
    indices = _user_turn_indices(before)
    if not indices or turns_undone < 1:
        return None, []
    cut = indices[max(len(indices) - turns_undone, 0)]
    return before[cut].get("timestamp"), before[:cut]


def print_undo_rollback_hint(cli, before: list, turns_undone: int) -> None:
    from hermes_agent.hermes_cli.gateway_side import connection_for
    from hermes_agent.tools.checkpoint_undo_hint import undo_hint_payload, undo_span_start
    target_ts, kept = _undone_span(before, turns_undone)
    if target_ts is None:
        return
    hint = None
    if (connection := connection_for(cli)) is not None:
        with contextlib.suppress(Exception):  # an older owner refuses the param: no line, as on main
            hint = connection.call("rollback.list", session_id=cli.session_id,
                                   undo_since=undo_span_start(target_ts, kept)).get("undo_hint")
    else:
        mgr = getattr(getattr(cli, "agent", None), "_checkpoint_mgr", None)
        hint = undo_hint_payload(mgr, os.getenv("TERMINAL_CWD", os.getcwd()), undo_span_start(target_ts, kept))
    if text := (hint or {}).get("text"):
        print(f"  {text}")
