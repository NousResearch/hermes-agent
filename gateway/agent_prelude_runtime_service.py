"""Shared runtime helpers for gateway agent prelude assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gateway.config import Platform


@dataclass(slots=True)
class GatewayAgentPreludeResult:
    """Prepared gateway agent prelude state before invoking _run_agent()."""

    hook_ctx: dict[str, Any]
    message_text: str
    blocked: bool = False


def append_discord_voice_channel_context(
    context_prompt: str,
    *,
    platform: Platform | None,
    guild_id: int | None,
    adapter: Any,
) -> str:
    """Append Discord voice-channel awareness text when available."""

    if platform != Platform.DISCORD or not guild_id or adapter is None:
        return context_prompt
    if not hasattr(adapter, "get_voice_channel_context"):
        return context_prompt

    voice_context = adapter.get_voice_channel_context(guild_id)
    if not voice_context:
        return context_prompt
    return f"{context_prompt}\n\n{voice_context}"


def build_agent_start_hook_context(
    *,
    platform: Platform | None,
    user_id: str | None,
    session_id: str,
    message_text: str,
    message_preview_chars: int = 500,
) -> dict[str, Any]:
    """Build the context payload emitted to `agent:start` hooks."""

    return {
        "platform": platform.value if platform else "",
        "user_id": user_id,
        "session_id": session_id,
        "message": str(message_text or "")[:message_preview_chars],
    }


async def run_gateway_agent_prelude(
    *,
    hooks: Any,
    hook_ctx: dict[str, Any],
    message_text: str,
    should_expand_context_references: bool,
    expand_context_references: Callable[[], Awaitable[Any]] | None = None,
    send_blocked_warning: Callable[[str], Awaitable[None]] | None = None,
) -> GatewayAgentPreludeResult:
    """Run the gateway prelude flow before the main agent invocation."""

    await hooks.emit("agent:start", hook_ctx)

    if not should_expand_context_references or expand_context_references is None:
        return GatewayAgentPreludeResult(
            hook_ctx=hook_ctx,
            message_text=message_text,
        )

    outcome = await expand_context_references()
    blocked_warning = getattr(outcome, "blocked_warning", None)
    if blocked_warning:
        if send_blocked_warning is not None:
            await send_blocked_warning(str(blocked_warning))
        return GatewayAgentPreludeResult(
            hook_ctx=hook_ctx,
            message_text=message_text,
            blocked=True,
        )

    return GatewayAgentPreludeResult(
        hook_ctx=hook_ctx,
        message_text=str(getattr(outcome, "message_text", message_text) or message_text),
    )
