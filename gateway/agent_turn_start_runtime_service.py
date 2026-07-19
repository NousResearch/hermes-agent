"""Shared runtime helpers for gateway per-turn session/bootstrap setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

from gateway.direct_shortcut_runtime_service import (
    try_handle_direct_gateway_shortcuts,
)
from gateway.message_turn_context_runtime_service import (
    prepare_gateway_message_turn_context,
)
from gateway.session import build_session_context


@dataclass(slots=True)
class GatewayPreparedAgentTurnStart:
    """Prepared gateway turn state after session/context bootstrap.

    ``context_prompt`` is stable session context only; must-deliver notes are
    in ``turn_sidecar_notes`` (sidecar-only contract).
    """

    session_entry: Any
    session_key: str
    context: Any
    history: list[dict[str, Any]]
    history_for_agent: list[dict[str, Any]]
    context_prompt: str
    immediate_response: str | None = None
    turn_sidecar_notes: list[str] | None = None


async def prepare_gateway_agent_turn_start(
    *,
    runner: Any,
    event: Any,
    source: Any,
    config_path: Path,
    hermes_home: Path,
    logger: Any,
    explicit_group_reply_note: str = "",
    visible_limit: int = 20,
    runtime_agent_kwargs_loader: Callable[[], dict[str, Any]] | None = None,
    build_session_context_fn: Callable[..., Any] = build_session_context,
    prepare_message_turn_context_fn: Callable[..., Awaitable[Any]] = prepare_gateway_message_turn_context,
    direct_shortcut_handler_fn: Callable[..., str | None] = try_handle_direct_gateway_shortcuts,
    build_session_context_prompt_fn: Callable[..., str] | None = None,
) -> GatewayPreparedAgentTurnStart:
    """Prepare session state, transcript history, direct shortcuts, and turn context."""

    session_entry = runner.session_store.get_or_create_session(source)
    session_key = session_entry.session_key
    is_new_session = (
        session_entry.created_at == session_entry.updated_at
        or getattr(session_entry, "was_auto_reset", False)
    )

    if is_new_session:
        await runner.hooks.emit(
            "session:start",
            {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            },
        )

    admin_user_ids = runner._configured_admin_user_ids(source.platform)
    is_admin_user = runner._is_admin_user(source) if admin_user_ids else None
    context = build_session_context_fn(
        source,
        runner.config,
        session_entry,
        admin_user_ids=admin_user_ids,
        is_admin_user=is_admin_user,
    )

    runner._current_session_env_event = event
    try:
        runner._set_session_env(context)
    finally:
        runner._current_session_env_event = None
    history = runner.session_store.load_transcript(session_entry.session_id)

    direct_shortcut_response = direct_shortcut_handler_fn(
        runner,
        event,
        conversation_history=list(history or []),
        logger=logger,
    )
    if direct_shortcut_response is not None:
        return GatewayPreparedAgentTurnStart(
            session_entry=session_entry,
            session_key=session_key,
            context=context,
            history=history,
            history_for_agent=[],
            context_prompt="",
            immediate_response=direct_shortcut_response,
            turn_sidecar_notes=[],
        )

    prepared_turn_context = await prepare_message_turn_context_fn(
        runner=runner,
        event=event,
        source=source,
        context=context,
        session_entry=session_entry,
        session_key=session_key,
        history=history,
        is_new_session=is_new_session,
        config_path=config_path,
        hermes_home=hermes_home,
        logger=logger,
        explicit_group_reply_note=explicit_group_reply_note,
        visible_limit=visible_limit,
        runtime_agent_kwargs_loader=runtime_agent_kwargs_loader,
        build_session_context_prompt_fn=build_session_context_prompt_fn,
    )

    immediate_response = prepared_turn_context.auto_background_response
    return GatewayPreparedAgentTurnStart(
        session_entry=session_entry,
        session_key=session_key,
        context=context,
        history=prepared_turn_context.history,
        history_for_agent=prepared_turn_context.history_for_agent,
        context_prompt=prepared_turn_context.context_prompt,
        immediate_response=immediate_response,
        turn_sidecar_notes=list(getattr(prepared_turn_context, "turn_sidecar_notes", None) or []),
    )
