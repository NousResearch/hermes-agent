"""CLI /branch handler, split from cli_commands_mixin without changing session-switch order."""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from datetime import datetime

from agent.message_metadata import message_identity
from agent.turn_context import extract_api_content_sidecar
from hermes_cli.cli_commands_session_tools import _command_arg, _gt, _t, _tn
from hermes_state_ids import new_session_id as mint_session_id

logger = logging.getLogger("hermes_cli.cli_commands_mixin")


def handle_branch_command(
    cli, cmd_original: str, *, db_unavailable_line, sync_agent_to_session,
    end_current_session, branch_copy_keys, print_line,
) -> None:
    """Fork the full transcript into an independent session while preserving the warm prompt."""
    # An in-flight agent run would flush through the rotating session identity: the branch
    # ends the parent row and repoints agent.session_id, so refuse mid-turn like /handoff.
    if getattr(cli, "_agent_running", False):
        return print_line(f"  {_t('shared.agent_busy', command='/branch')}")
    from cli import _sync_process_session_id
    if not cli.conversation_history:
        return print_line(f"  {_gt('branch.no_conversation')}")
    if not cli._session_db:
        return print_line(db_unavailable_line())
    # CLI has no threads: always in place; strip the gateway's ``--here`` so it is never a title.
    from gateway.slash_commands_branch_thread import parse_branch_args
    _, branch_name = parse_branch_args(_command_arg(cmd_original))
    now = datetime.now()
    new_session_id = mint_session_id(now)
    branch_title = branch_name or cli._session_db.get_next_title_in_lineage(
        cli._session_db.get_session_title(cli.session_id) or "branch")
    parent_session_id = cli.session_id
    # PostgreSQL owns the durable branch publication as one transaction. Flush idle CLI
    # messages first, then switch process/agent/memory only after commit.
    from cli_session_store import PostgreSQLCLISessionStore
    if isinstance(cli._session_db, PostgreSQLCLISessionStore):
        if cli.agent:
            with suppress(Exception):
                cli.agent._flush_messages_to_session_db(
                    cli.conversation_history, conversation_history=cli.conversation_history)
        try:
            cli._session_db.branch_session(
                parent_session_id=parent_session_id, child_session_id=new_session_id,
                source=os.environ.get("HERMES_SESSION_SOURCE", "cli"), model=cli.model,
                model_config={"max_iterations": cli.max_turns, "reasoning_config": cli.reasoning_config},
                title=branch_title,
            )
        except Exception as e:
            logger.exception("Failed to create branch session")
            return print_line(f"  Failed to create branch session: {e}")
        cli._transfer_session_yolo(cli.session_id, new_session_id)
        cli.session_id, cli.session_start, cli._pending_title = new_session_id, now, None
        cli._resumed = True
        _sync_process_session_id(new_session_id)
        if cli.agent:
            cli.agent.session_start = now
        sync_agent_to_session(cli, new_session_id, parent_session_id=parent_session_id, reason="branch")
        msg_count = len([m for m in cli.conversation_history if m.get("role") == "user"])
        return print_line("  " + _tn("branch.branched", msg_count, title=branch_title),
                   f"  {_t('branch.original_session', session_id=parent_session_id)}",
                   f"  {_t('branch.branch_session', session_id=new_session_id)}")
    # Create the child BEFORE ending the parent: a failed create_session must leave the session the
    # user is still on open, not ended with end_reason="branched" and no branch (#11030).
    # The stable ``_branched_from`` marker keeps the branch visible in /resume + /sessions
    # even after the parent is re-ended with a different end_reason.
    # The child sends the parent's exact system prompt: a row without one makes the branch's first
    # turn rebuild (re-probing the workspace), so the warm cache the copied transcript buys is
    # lost at byte 0 whenever the repo moved since the parent's session start.
    parent_prompt = getattr(cli.agent, "_cached_system_prompt", None)
    if not isinstance(parent_prompt, str) or not parent_prompt:
        with suppress(Exception):
            parent_prompt = (cli._session_db.get_session(parent_session_id) or {}).get("system_prompt")
    try:
        cli._session_db.create_session(
            session_id=new_session_id, source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
            model=cli.model, parent_session_id=parent_session_id, system_prompt=parent_prompt or None,
            model_config={"max_iterations": cli.max_turns, "reasoning_config": cli.reasoning_config,
                          "_branched_from": parent_session_id})
    except Exception as e:
        logger.exception("Failed to create SQLite branch session")
        return print_line(f"  {_gt('branch.create_failed', error=e)}")
    end_current_session(cli, "branched")
    # Best-effort chunked copy (a failed copy still yields a usable branch); the api_content
    # sidecar lets the branch's first turn replay the parent's exact wire bytes (warm cache).
    with suppress(Exception):
        cli._session_db.append_messages_batch(new_session_id, [
            {"role": msg.get("role", "user"), "tool_name": msg.get("tool_name") or msg.get("name"),
             "api_content": extract_api_content_sidecar(msg),
             **{k: msg.get(k) for k in branch_copy_keys}, **message_identity(msg, with_tool_uids=True)}
            for msg in cli.conversation_history], chunk_rows=500)
    with suppress(Exception):
        cli._session_db.set_session_title(new_session_id, branch_title)
    # Switch to the new session
    cli._transfer_session_yolo(cli.session_id, new_session_id)
    cli.session_id, cli.session_start, cli._pending_title = new_session_id, now, None
    cli._resumed = True  # Prevents auto-title generation
    _sync_process_session_id(new_session_id)
    if cli.agent:
        cli.agent.session_start = now
    sync_agent_to_session(cli, new_session_id, parent_session_id=parent_session_id, reason="branch")
    msg_count = len([m for m in cli.conversation_history if m.get("role") == "user"])
    print_line("  " + _tn("branch.branched", msg_count, title=branch_title),
        f"  {_t('branch.original_session', session_id=parent_session_id)}",
        f"  {_t('branch.branch_session', session_id=new_session_id)}")
