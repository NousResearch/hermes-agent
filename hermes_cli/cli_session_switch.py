"""Session-switch plumbing shared by /resume, /sessions and /branch.

The switch itself lives in ``cli_commands_mixin``; the handoff it needs sits here: end the row
being left, then repoint an already-built agent.
"""
from __future__ import annotations

from contextlib import suppress


def _end_current_session(cli, reason: str) -> None:
    """Flush un-persisted messages, end the current session row with ``reason``, then prune that
    row when the switch abandoned it before its first message. Best-effort on every step (the
    switch proceeds even if the DB write fails)."""
    if cli.agent:
        with suppress(Exception):
            cli.agent._flush_messages_to_session_db(
                cli.conversation_history, conversation_history=cli.conversation_history)
    with suppress(Exception):
        cli._session_db.end_session(cli.session_id, reason)
    # A switch away from a session that never gained content leaves a titeless row that /resume
    # and `hermes sessions list` then offer forever (#146165). /new and the TUI quit path prune
    # theirs right after end_session; this path must too. The prune belongs HERE, before the
    # caller swaps cli.session_id: _discard_session_if_empty reads cli.conversation_history as
    # the authoritative transcript and probes the id it is handed.
    with suppress(Exception):
        cli._discard_session_if_empty(cli.session_id)


def _sync_agent_to_session(cli, session_id: str, *, parent_session_id: str, reason: str) -> None:
    """Point an already-built agent at ``session_id`` after a /resume or /branch switch: reset
    per-session state, re-anchor the DB flush index, and notify memory providers with
    reset=False (their state stays valid and just targets the new id; parent keeps lineage)."""
    if not cli.agent:
        return
    cli.agent.session_id = session_id
    cli.agent.reset_session_state()
    if hasattr(cli.agent, "_last_flushed_db_idx"):
        cli.agent._last_flushed_db_idx = len(cli.conversation_history)
    if hasattr(cli.agent, "_todo_store"):
        with suppress(Exception):
            from tools.todo_tool import TodoStore
            cli.agent._todo_store = TodoStore()
    if hasattr(cli.agent, "_invalidate_system_prompt"):
        cli.agent._invalidate_system_prompt()
    with suppress(Exception):
        _mm = getattr(cli.agent, "_memory_manager", None)
        # reset=False: provider state follows the new id; parent_session_id keeps lineage (#6672).
        if _mm is not None:
            _mm.on_session_switch(
                session_id, parent_session_id=parent_session_id or "", reset=False, reason=reason)
