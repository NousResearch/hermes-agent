"""Shared helpers for HTTP-originated gateway ingress."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from gateway.platforms.base import MessageEvent, MessageType


class IngressAdapter(Protocol):
    platform: Any
    _background_tasks: set

    def build_source(
        self,
        chat_id: str,
        chat_name: str | None = None,
        chat_type: str = "dm",
        user_id: str | None = None,
        user_name: str | None = None,
        message_id: str | None = None,
    ) -> Any: ...

    async def handle_message(self, event: MessageEvent) -> None: ...


@dataclass(frozen=True)
class IngressEnvelope:
    text: str
    message_id: str
    chat_id: str
    chat_name: str | None = None
    chat_type: str = "webhook"
    user_id: str | None = None
    user_name: str | None = None
    raw_payload: Any = None
    internal: bool = False


@dataclass(frozen=True)
class HttpIngressRequestContext:
    remote: str = ""
    peer_ip: str = ""
    forwarded_for: str = ""
    real_ip: str = ""
    method: str = ""
    path: str = ""
    user_agent: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedIngressRequest:
    mode: str
    request_id: str
    user_message: Any
    conversation_history: list[dict[str, Any]]
    session_id: str | None = None
    session_key: str | None = None
    ephemeral_system_prompt: str | None = None
    metadata: dict[str, Any] | None = None
    reply_target: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_http_ingress_value(value: Any, *, max_len: int = 200) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:max_len]


def extract_http_ingress_request_context(request: Any) -> HttpIngressRequestContext:
    peer_ip = ""
    try:
        transport = getattr(request, "transport", None)
        peer = transport.get_extra_info("peername") if transport else None
        if isinstance(peer, (tuple, list)) and peer:
            peer_ip = str(peer[0])
    except Exception:
        peer_ip = ""

    headers = getattr(request, "headers", {}) or {}
    remote = getattr(request, "remote", "") or peer_ip
    return HttpIngressRequestContext(
        remote=sanitize_http_ingress_value(remote),
        peer_ip=sanitize_http_ingress_value(peer_ip),
        forwarded_for=sanitize_http_ingress_value(headers.get("X-Forwarded-For", "")),
        real_ip=sanitize_http_ingress_value(headers.get("X-Real-IP", "")),
        method=sanitize_http_ingress_value(getattr(request, "method", ""), max_len=16),
        path=sanitize_http_ingress_value(getattr(request, "path_qs", ""), max_len=500),
        user_agent=sanitize_http_ingress_value(headers.get("User-Agent", ""), max_len=300),
    )


def format_http_ingress_request_context(context: HttpIngressRequestContext) -> str:
    fields = [f"{key}={value!r}" for key, value in context.to_dict().items() if value]
    return " ".join(fields) if fields else "source='unknown'"


def build_http_ingress_origin(*, platform: str, chat_id: str, context: HttpIngressRequestContext) -> dict[str, str]:
    origin = {"platform": platform, "chat_id": chat_id}
    if context.remote:
        origin["source_ip"] = context.remote
    if context.peer_ip:
        origin["peer_ip"] = context.peer_ip
    if context.forwarded_for:
        origin["forwarded_for"] = context.forwarded_for
    if context.real_ip:
        origin["real_ip"] = context.real_ip
    if context.user_agent:
        origin["user_agent"] = context.user_agent
    return origin


def build_ingress_message_event(adapter: IngressAdapter, envelope: IngressEnvelope) -> MessageEvent:
    source = adapter.build_source(
        chat_id=envelope.chat_id,
        chat_name=envelope.chat_name,
        chat_type=envelope.chat_type,
        user_id=envelope.user_id,
        user_name=envelope.user_name,
        message_id=envelope.message_id,
    )
    return MessageEvent(
        text=envelope.text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=envelope.raw_payload,
        message_id=envelope.message_id,
        internal=envelope.internal,
    )


def schedule_ingress_event(adapter: IngressAdapter, event: MessageEvent) -> asyncio.Task:
    task = asyncio.create_task(adapter.handle_message(event))
    try:
        adapter._background_tasks.add(task)
    except Exception:
        pass
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(adapter._background_tasks.discard)
    return task


def schedule_ingress_envelope(adapter: IngressAdapter, envelope: IngressEnvelope) -> asyncio.Task:
    return schedule_ingress_event(adapter, build_ingress_message_event(adapter, envelope))
