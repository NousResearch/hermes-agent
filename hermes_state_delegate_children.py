"""Marked delegate subagents owned by one session's lineage (#150198).

Kept beside hermes_state_sessions (at its line cap): the TUI kanban poller uses it to reclaim
subscriptions that ``kanban_create`` bound to a subagent while that subagent's row lacked its
``_delegate_from`` marker.
"""

from __future__ import annotations

from hermes_state_sessions import _collect_delegate_child_ids, _parse_model_config


def delegate_descendant_ids(db, session_id: str) -> list[str]:
    """Marked delegate subagents (recursive) of ``session_id`` and the compression segments it continues.

    The walk up stops at a delegate or branch row: those are not continuations of their parent, and a
    subagent must never claim its parent's children. A delegate ``session_id`` itself has none.
    """
    seeds: list[str] = []
    for sid in reversed(db._session_lineage_root_to_tip(session_id)):
        row = db._read_one("SELECT model_config FROM sessions WHERE id = ?", (sid,))
        config = _parse_model_config(row[0]) if row is not None else {}
        if config.get("_delegate_from"):
            break
        seeds.append(sid)
        if config.get("_branched_from"):
            break
    if not seeds:
        return []
    with db._read_ctx() as conn:
        return _collect_delegate_child_ids(conn, seeds)
