"""Shared runtime helpers for gateway agent prelude assembly."""

from __future__ import annotations

from typing import Any

from gateway.config import Platform


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
