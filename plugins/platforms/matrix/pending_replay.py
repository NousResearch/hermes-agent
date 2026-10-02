"""Current native Matrix evidence for persisted pending input."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from gateway.pending_native import PendingNativeInput
from gateway.session import SessionSource
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.matrix.client_events import Method, raw_state_event
from plugins.platforms.matrix.effective_event import effective_event
from plugins.platforms.matrix.reply_context import MatrixEventContextCache


class MatrixPendingReplayMixin(BasePlatformAdapter):
    _client: Any
    _user_id: str
    _event_context_cache: MatrixEventContextCache
    _build_inbound_event: Callable[..., Awaitable[MessageEvent | None]]
    _resolve_message_context: Callable[..., Awaitable[tuple | None]]

    def pending_native_input(self, event: MessageEvent) -> PendingNativeInput | None:
        if not isinstance(event.raw_message, dict) or not isinstance(
            event.raw_message.get("msgtype"), str
        ):
            return None
        return PendingNativeInput.capture(event, event.raw_message)

    async def revalidate_pending_event(
        self,
        event: MessageEvent,
        *,
        authorize: Callable[[SessionSource], bool] | None = None,
    ) -> MessageEvent | None:
        if authorize is not None and not authorize(event.source):
            return None
        from hermes_constants import get_hermes_home
        from plugins.platforms.matrix.adapter import _normalize_matrix_bang_command

        if (
            self._client is None
            or event.internal
            or event._merged_parts
            or not event.message_id
        ):
            return None
        cache = get_hermes_home().resolve() / "cache"
        native = event._pending_native_input
        room_id, event_id = event.source.chat_id, event.message_id
        path = (
            f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}"
            f"/event/{quote(event_id, safe='')}"
        )
        try:
            current, membership = await asyncio.wait_for(
                asyncio.gather(
                    self._client.api.request(Method.GET, path),
                    raw_state_event(
                        self._client, room_id, "m.room.member", self._user_id
                    ),
                ),
                self._event_context_cache.timeout_seconds,
            )
            if (
                not isinstance(current, dict)
                or current.get("room_id") != room_id
                or current.get("event_id") != event_id
                or current.get("sender") != event.source.user_id
                or current.get("type")
                not in {"m.room.message", "m.room.encrypted", "m.sticker"}
                or not isinstance(membership, dict)
                or membership.get("membership") != "join"
            ):
                return None
            state = await effective_event(
                self._client, current, cache=self._event_context_cache, room_id=room_id
            )
            content = state.content
            if state.redacted or state.error or not isinstance(content, dict):
                return None
            if native is not None and state.plain_original_content != native.content:
                return None
            kind = content.get("msgtype")
            media_types = {
                "m.image": MessageType.PHOTO,
                "m.sticker": MessageType.PHOTO,
                "m.audio": MessageType.VOICE,
                "m.video": MessageType.VIDEO,
                "m.file": MessageType.DOCUMENT,
            }
            body = content.get("body")
            if (
                not isinstance(body, str)
                or not isinstance(kind, str)
                or kind not in {"m.text", "m.emote", *media_types}
            ):
                return None
            body = _normalize_matrix_bang_command(body) if kind == "m.text" else body
            relates_to = content.get("m.relates_to") or {}
            ctx = await self._resolve_message_context(
                room_id, event.source.user_id, event_id, body, content, relates_to
            )
            if ctx is None:
                return None
            if event.media_urls and (
                native is None
                or not all(
                    Path(path).resolve().is_relative_to(cache)
                    for path in event.media_urls
                )
                or not native.attachments_available(event.media_urls)
            ):
                return None
            extras: dict[str, Any] = {}
            if kind in media_types:
                if native is None or content != native.content:
                    return None
                authored = event.authored_media()
                if not authored.media_urls:
                    return None
                extras = {
                    "media_msgtype": kind,
                    "message_type": media_types[kind],
                    "media_urls": list(authored.media_urls),
                    "media_types": list(authored.media_types),
                    "media_text_inlined": list(authored.media_text_inlined),
                }
            elif native is None and (
                kind != "m.text"
                or body != event.text
                or event._inbound_context_dependencies
            ):
                return None
            verified = await self._build_inbound_event(
                room_id,
                event.source.user_id,
                event_id,
                body,
                content,
                relates_to,
                ctx=ctx,
                **extras,
            )
            if verified is None or verified.is_command():
                return None
            verified.timestamp = event.timestamp
            verified._pending_native_input = PendingNativeInput.capture(
                verified, state.plain_original_content
            )
            return verified
        except Exception:
            return None
