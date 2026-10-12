"""Classic CLI adapter for native plugin session commands."""

from __future__ import annotations

from contextlib import suppress


def _sync_agent_to_session(cli, session_id: str, *, parent_session_id: str, reason: str) -> None:
    """Point an already-built agent at ``session_id`` after a /resume or /branch switch: reset
    per-session state, re-anchor the DB flush index, and notify memory providers with
    reset=False (their state stays valid and just targets the new id; parent keeps lineage)."""
    if not cli.agent:
        return
    cli.agent.session_id = session_id
    cli.agent.reset_session_state()
    # Rehydrate the exact target row so a session switch cannot retain the
    # previous session's plugin policy overrides.
    with suppress(Exception):
        from agent.session_policies import hydrate_session_policies

        hydrate_session_policies(
            cli.agent,
            session_db=cli._session_db,
            session_id=session_id,
        )
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
        # Notify memory providers that session_id rotated to a fresh conversation. reset=True signals
        # providers to flush accumulated per-session state (_session_turns, _turn_counter, _document_id).
        # Fires BEFORE the plugin on_session_reset hook (shell hooks only see the new id; Python providers
        # see the transition). See #6672. When the old session has history, end-of-session extraction
        # (LLM-bound, seconds) and this switch are queued as ONE task on the memory manager's serialized
        # worker — end strictly before switch, without blocking /new (#16454). With no history there is
        # nothing to extract; switch inline as before.
        # Notify memory providers that session_id rotated to a resumed session. reset=False — the provider's
        # accumulated state is still valid; it just needs to target the new session_id for subsequent
        # writes. See #6672.
        # Notify memory providers that session_id forked to a new branch. reset=False — the branched session
        # carries the transcript forward, so provider state tracks the lineage. parent_session_id links the
        # branch back to the original. See #6672.
        if _mm is not None:
            _mm.on_session_switch(
                session_id, parent_session_id=parent_session_id or "", reset=False, reason=reason)


def run_cli_session_command(cli, handler, raw_args: str):
    """Parse before touching the foreground session and use its canonical writer."""
    from hermes_cli.session_commands import (
        make_session_context, prepare_session_command, run_session_command,
    )
    from hermes_constants import get_hermes_home

    try:
        request = prepare_session_command(handler, raw_args)
    except ValueError as exc:
        return str(exc)

    agent = getattr(cli, "agent", None)
    if agent is None:
        raise ValueError("the foreground agent is not initialized")
    session_id = str(getattr(agent, "session_id", "") or "")
    if not session_id.strip():
        raise ValueError("a non-empty exact session ID is required")
    if str(getattr(cli, "session_id", session_id) or "") != session_id:
        raise ValueError("foreground session identity mismatch; refusing to mutate")
    session_db = getattr(agent, "_session_db", None)
    context = make_session_context(
        agent=agent,
        session_db=session_db,
        session_id=session_id,
        ensure_session=getattr(agent, "_ensure_db_session", None),
        profile_home=get_hermes_home(),
    )
    return run_session_command(handler, request, context)
