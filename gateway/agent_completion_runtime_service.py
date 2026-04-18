"""Shared runtime helpers for gateway agent-completion handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gateway.agent_response_runtime_service import normalize_gateway_agent_response


@dataclass(slots=True)
class GatewayPreparedAgentCompletion:
    """Prepared post-run agent state ready for transcript persistence/delivery."""

    response: str
    suppress_reply: bool
    response_state: str
    agent_messages: list[dict[str, Any]]


async def stop_gateway_typing_indicator(
    *,
    adapters: dict[Any, Any],
    platform: Any,
    chat_id: str | None,
) -> None:
    """Best-effort typing-indicator shutdown used on both success and error paths."""

    try:
        typing_adapter = adapters.get(platform)
        if typing_adapter and hasattr(typing_adapter, "stop_typing"):
            await typing_adapter.stop_typing(chat_id)
    except Exception:
        pass


def log_gateway_response_ready(
    *,
    logger: Any,
    platform_name: str,
    chat_id: str | None,
    msg_start_time: float,
    agent_result: dict[str, Any],
    response: str,
    response_state: str,
) -> None:
    """Emit the standard gateway response-ready log line."""

    response_time = time.time() - msg_start_time
    api_calls = agent_result.get("api_calls", 0)
    response_len = len(response)
    logger.info(
        "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars state=%s",
        platform_name,
        chat_id or "unknown",
        response_time,
        api_calls,
        response_len,
        response_state,
    )


def sync_gateway_session_entry_id(
    *,
    session_entry: Any,
    agent_result: dict[str, Any],
) -> bool:
    """Apply any session split/compression session_id change to the live entry."""

    new_session_id = agent_result.get("session_id")
    if new_session_id and new_session_id != session_entry.session_id:
        session_entry.session_id = new_session_id
        return True
    return False


def build_gateway_agent_end_payload(
    *,
    hook_ctx: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    """Build the payload for the gateway agent:end hook."""

    return {
        **hook_ctx,
        "response": (response or "")[:500],
    }


def apply_gateway_reasoning_display(
    *,
    response: str,
    show_reasoning: bool,
    last_reasoning: Any,
) -> str:
    """Optionally prepend collapsed reasoning to the visible response."""

    if not show_reasoning or not response or not last_reasoning:
        return response

    lines = str(last_reasoning).strip().splitlines()
    if len(lines) > 15:
        display_reasoning = "\n".join(lines[:15])
        display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
    else:
        display_reasoning = str(last_reasoning).strip()
    return f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"


def drain_pending_process_watchers(
    *,
    process_registry: Any,
    run_process_watcher: Callable[[dict[str, Any]], Awaitable[None]],
    create_task: Callable[[Awaitable[None]], Any],
    logger: Any | None = None,
    resumed_log_template: str | None = None,
) -> int:
    """Schedule all pending background-process watchers and return the count."""

    scheduled = 0
    while process_registry.pending_watchers:
        watcher = process_registry.pending_watchers.pop(0)
        create_task(run_process_watcher(watcher))
        scheduled += 1
        if logger is not None and resumed_log_template:
            logger.info(resumed_log_template, watcher.get("session_id"))
    return scheduled


async def prepare_gateway_agent_completion(
    *,
    agent_result: dict[str, Any],
    history_len: int,
    empty_response_fallback: Callable[[str], str | None],
    session_entry: Any,
    show_reasoning: bool,
    hook_ctx: dict[str, Any],
    hooks: Any,
    logger: Any,
    platform_name: str,
    chat_id: str | None,
    msg_start_time: float,
    process_registry: Any | None = None,
    run_process_watcher: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    create_task: Callable[[Awaitable[None]], Any] | None = None,
) -> GatewayPreparedAgentCompletion:
    """Normalize and finalize the non-delivery success path after _run_agent()."""

    normalized_response = normalize_gateway_agent_response(
        agent_result=agent_result,
        history_len=history_len,
        empty_response_fallback=empty_response_fallback,
    )
    response = normalized_response.response
    suppress_reply = normalized_response.suppress_reply
    response_state = normalized_response.response_state
    agent_messages = agent_result.get("messages", [])
    agent_result["response_state"] = response_state
    agent_result["synthetic_fallback"] = bool(normalized_response.synthetic_fallback)

    log_gateway_response_ready(
        logger=logger,
        platform_name=platform_name,
        chat_id=chat_id,
        msg_start_time=msg_start_time,
        agent_result=agent_result,
        response=response,
        response_state=response_state,
    )
    sync_gateway_session_entry_id(
        session_entry=session_entry,
        agent_result=agent_result,
    )

    response = apply_gateway_reasoning_display(
        response=response,
        show_reasoning=show_reasoning,
        last_reasoning=agent_result.get("last_reasoning"),
    )

    await hooks.emit(
        "agent:end",
        build_gateway_agent_end_payload(
            hook_ctx=hook_ctx,
            response=response,
        ),
    )

    if process_registry is not None and run_process_watcher is not None and create_task is not None:
        drain_pending_process_watchers(
            process_registry=process_registry,
            run_process_watcher=run_process_watcher,
            create_task=create_task,
        )

    return GatewayPreparedAgentCompletion(
        response=response,
        suppress_reply=suppress_reply,
        response_state=response_state,
        agent_messages=agent_messages,
    )
