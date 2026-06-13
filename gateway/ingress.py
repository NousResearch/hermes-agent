"""Shared helpers for HTTP-style ingress sources.

This module is intentionally narrow: it defines a small normalized envelope
for gateway-style ingress events and the common boilerplate for turning those
into ``MessageEvent`` objects and scheduling background dispatch.

The first concrete consumers are webhook-like adapters.  API-server request
handlers still own their direct ``AIAgent`` execution path for now, but can
progressively reuse the request/envelope helpers added here as the ingress
surfaces converge.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from gateway.platforms.base import MessageEvent, MessageType, SessionSource


class IngressAdapter(Protocol):
    """Minimal adapter surface required by the shared ingress helpers."""

    _background_tasks: set[asyncio.Task]

    def build_source(
        self,
        *,
        chat_id: str,
        chat_name: str | None = None,
        chat_type: str = "dm",
        user_id: str | None = None,
        user_name: str | None = None,
        **kwargs: Any,
    ) -> SessionSource: ...

    async def handle_message(self, event: MessageEvent): ...


@dataclass(frozen=True)
class IngressEnvelope:
    """Normalized internal representation for HTTP-originated message ingress."""

    text: str
    message_id: str | None
    chat_id: str
    chat_name: str | None = None
    chat_type: str = "webhook"
    user_id: str | None = None
    user_name: str | None = None
    raw_payload: Any = None
    internal: bool = False


def build_ingress_message_event(adapter: IngressAdapter, envelope: IngressEnvelope) -> MessageEvent:
    """Build a ``MessageEvent`` for an ingress envelope using an adapter's source factory."""

    source = adapter.build_source(
        chat_id=envelope.chat_id,
        chat_name=envelope.chat_name,
        chat_type=envelope.chat_type,
        user_id=envelope.user_id,
        user_name=envelope.user_name,
    )
    return MessageEvent(
        text=envelope.text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=envelope.raw_payload,
        message_id=envelope.message_id,
        internal=envelope.internal,
    )


def schedule_ingress_event(adapter: IngressAdapter, event: MessageEvent) -> asyncio.Task | None:
    """Schedule a background ingress event and register it with the adapter."""

    task = asyncio.create_task(adapter.handle_message(event))
    try:
        adapter._background_tasks.add(task)
    except TypeError:
        # Mirror the base adapter's test-friendly behavior: some tests stub
        # task creation with lightweight sentinels that aren't hashable.
        return None
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(adapter._background_tasks.discard)
    return task


def schedule_ingress_envelope(adapter: IngressAdapter, envelope: IngressEnvelope) -> asyncio.Task | None:
    """Build and schedule an ingress envelope in one step."""

    event = build_ingress_message_event(adapter, envelope)
    return schedule_ingress_event(adapter, event)
