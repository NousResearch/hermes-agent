"""Model-inaccessible gateway metadata for the AgentCrew MCP transport."""
from __future__ import annotations

from typing import Optional

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def agentcrew_transport(
    event: MessageEvent, source: SessionSource, session_id: str
) -> Optional[dict[str, str]]:
    """Return authoritative Telegram identity, or None when the event lacks transport proof."""
    if (
        event.internal
        or source.platform != Platform.TELEGRAM
        or not event.message_id
        or event.platform_update_id is None
        or not event._transport_received_at
    ):
        return None
    return {
        "version": "hermes-agentcrew-trusted-transport/0.1",
        "platform": "telegram",
        "profile": str(source.profile or ""),
        "session_id": str(session_id),
        "chat_id": str(source.chat_id),
        "thread_id": str(source.thread_id or ""),
        "message_id": str(event.message_id),
        "update_id": str(event.platform_update_id),
        "received_at": event._transport_received_at,
    }


def bind_agentcrew_transport(event: MessageEvent, source: SessionSource, session_id: str) -> None:
    """Bind the trusted envelope after the ordinary session context has been established."""
    from gateway.session_context import set_trusted_transport
    set_trusted_transport(agentcrew_transport(event, source, session_id))
