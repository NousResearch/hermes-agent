"""Shared runtime helpers for gateway post-response delivery."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


async def finalize_gateway_agent_delivery(
    *,
    agent_result: dict[str, Any],
    suppress_reply: bool,
    response: str,
    agent_messages: list[dict[str, Any]] | None,
    event: Any,
    platform: Any,
    adapters: dict[Any, Any],
    should_send_voice_reply: Callable[..., bool],
    send_voice_reply: Callable[[Any, str], Awaitable[None]],
    deliver_media_from_response: Callable[[str, Any, Any], Awaitable[None]],
) -> str | None:
    """Handle voice/media side effects and return the final gateway reply."""

    resolved_messages = agent_messages or []
    already_sent = bool(agent_result.get("already_sent"))

    if (
        not suppress_reply
        and should_send_voice_reply(
            event,
            response,
            resolved_messages,
            already_sent=already_sent,
        )
    ):
        await send_voice_reply(event, response)

    if already_sent:
        if response:
            media_adapter = adapters.get(platform)
            if media_adapter:
                await deliver_media_from_response(response, event, media_adapter)
        return None

    if suppress_reply:
        return None

    return response
