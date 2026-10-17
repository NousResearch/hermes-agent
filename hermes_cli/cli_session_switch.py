"""Session-switch plumbing shared by /resume, /sessions and /branch.

The switch itself lives in ``cli_commands_mixin``; the handoff it needs sits here: end the row
being left, open the row being entered on this viewer, then repoint an already-built agent.
"""
from __future__ import annotations

from contextlib import suppress

from hermes_cli.gateway_connection import RESUME_GONE


def _end_current_session(cli, reason: str) -> None:
    """Flush un-persisted messages, then end the current session row with ``reason``.
    Best-effort on both steps (the switch proceeds even if the DB write fails)."""
    if cli.agent:
        with suppress(Exception):
            cli.agent._flush_messages_to_session_db(
                cli.conversation_history, conversation_history=cli.conversation_history)
    with suppress(Exception):
        cli._session_db.end_session(cli.session_id, reason)


def _open_switched_session(cli, session_id: str) -> str | None:
    """Subscribe ``session_id`` on this viewer's connection before a switch commits.

    Reads at the switch are local by design, the write side is a gateway verb: the gateway
    refuses ``prompt.submit`` for a session its connection never resumed
    (``gateway/session_controls.py``: ``submit`` -> ``permission_denied``). Startup resume and
    the agent's ``session_id`` setter reach the store's ``attach``; a switch typed before the
    first turn reaches neither, because ``_sync_agent_to_session`` only repoints an agent that
    already exists, so the first prompt after the switch was refused instead of answering.

    Returns ``None`` when the switch may proceed, else the refusal reason for the caller to
    report. Printing stays with the caller: its copy and its seams are the handler's.
    """
    store = getattr(cli, "_session_db", None)
    attach = getattr(store, "attach", None)
    if not callable(attach):
        return None
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
    try:
        attach(session_id)
    except GatewayRPCError as exc:
        # The owner answered: only an ownership refusal stops the switch.
        reason = str(exc)
        return reason if reason in RESUME_GONE else None
    except GatewayClientError:
        # No answer (unavailable, disconnected, timeout). The switch's reads are local, so it
        # proceeds as before; the turn reports the lost connection itself.
        return None
    return None


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
