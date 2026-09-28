"""Opt-in Matrix corrections queued as new conversational turns."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable
from urllib.parse import quote

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.matrix.effective_event import (
    _decrypt, _encrypted_replacement_content, event_content,
)
from plugins.platforms.matrix.relations import MatrixRelation

try:
    from mautrix.api import Method
except ImportError:
    from plugins.platforms.matrix.read_context import Method

logger = logging.getLogger(__name__)


def edit_followup_rooms(config: PlatformConfig) -> frozenset[str]:
    policy = config.extra.get("process_edits", {})
    if not isinstance(policy, dict):
        raise ValueError("matrix.process_edits must map exact room IDs to true or false")
    for room_id, enabled in policy.items():
        if (not isinstance(room_id, str) or not room_id.startswith("!")
                or ":" not in room_id or not isinstance(enabled, bool)):
            raise ValueError("matrix.process_edits must map exact room IDs to true or false")
    return frozenset(room_id for room_id, enabled in policy.items() if enabled)


class MatrixEditFollowupsMixin:
    _client: Any
    _event_context_cache: Any
    _process_edits: frozenset[str]
    _process_notices: bool
    _build_inbound_event: Callable[..., Awaitable[MessageEvent | None]]
    _resolve_message_context: Callable[..., Awaitable[tuple | None]]
    _is_sender_authorized: Callable[..., bool | None]
    _is_allowed_matrix_room_event: Callable[[str], Awaitable[bool]]
    handle_message: Callable[[MessageEvent], Awaitable[None]]

    async def validate_inbound_event(self, event: MessageEvent) -> bool:
        if not event._queue_at_turn_boundary:
            return True
        source = event.source
        room_id = source.chat_id
        target = event.metadata.get("edited_message_original_id")
        if not isinstance(source.user_id, str) or not isinstance(event.message_id, str):
            return False
        if (room_id not in self._process_edits or not isinstance(target, str)
                or self._event_context_cache.is_redacted(room_id, event.message_id)
                or not await self._is_allowed_matrix_room_event(room_id)):
            return False
        original = await self._edit_original_content(room_id, source.user_id, target)
        if original is None or (original.get("msgtype") == "m.notice" and not self._process_notices):
            return False
        raw = await self._edit_raw_event(room_id, event.message_id)
        if raw is None or raw.get("sender") != source.user_id:
            return False
        content = await self._edit_new_content(raw, target)
        if content is None:
            return False
        relation = original.get("m.relates_to")
        relation = relation if isinstance(relation, dict) else {}
        ctx = await self._resolve_message_context(
            room_id, source.user_id, target, str(content.get("body") or ""), content, relation,
            allow_gateway_control=False,
        )
        return ctx is not None and ctx[0] == event.text and ctx[-1].thread_id == source.thread_id

    async def _edit_original_content(self, room_id: str, sender: str, target: str) -> dict | None:
        if self._client is None or self._event_context_cache.is_redacted(room_id, target):
            return None
        raw = await self._edit_raw_event(room_id, target)
        unsigned = raw.get("unsigned") if raw is not None else None
        if (raw is None or raw.get("sender") != sender or "state_key" in raw
                or raw.get("type") not in {"m.room.message", "m.room.encrypted"}
                or (isinstance(unsigned, dict) and unsigned.get("redacted_because"))):
            return None
        content = event_content(raw)
        if raw.get("type") == "m.room.encrypted":
            decrypted, error = await _decrypt(self._client, raw)
            if error is not None:
                return None
            content = event_content(decrypted)
        if (self._event_context_cache.is_redacted(room_id, target)
                or MatrixRelation.from_content(content.get("m.relates_to")).is_edit
                or content.get("msgtype") not in {"m.text", "m.notice"}):
            return None
        return content

    async def _edit_raw_event(self, room_id: str, event_id: str) -> dict | None:
        if self._client is None:
            return None
        path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
        try:
            raw = await asyncio.wait_for(self._client.api.request(Method.GET, path), timeout=10.0)
        except Exception as exc:
            logger.debug("Matrix: could not validate edit event %s: %s", event_id, exc)
            return None
        if (not isinstance(raw, dict) or raw.get("event_id") != event_id
                or raw.get("room_id", room_id) != room_id):
            return None
        return {**raw, "room_id": room_id}

    async def _edit_new_content(self, raw: dict, target: str) -> dict | None:
        unsigned = raw.get("unsigned")
        room_id, event_id = raw.get("room_id"), raw.get("event_id")
        if ("state_key" in raw or (isinstance(unsigned, dict) and unsigned.get("redacted_because"))
                or self._event_context_cache.is_redacted(room_id, event_id)):
            return None
        source_content = event_content(raw)
        if raw.get("type") == "m.room.encrypted":
            decrypted, error = await _decrypt(self._client, raw)
            relation = event_content(decrypted).get("m.relates_to")
            if (error is not None or not isinstance(relation, dict)
                    or relation.get("rel_type") != "m.replace" or relation.get("event_id") != target):
                return None
            try:
                revised = await _encrypted_replacement_content(self._client, raw)
            except Exception as exc:
                logger.debug("Matrix: could not inspect encrypted edit %s: %s", event_id, exc)
                return None
        elif raw.get("type") == "m.room.message":
            relation = source_content.get("m.relates_to")
            if (not isinstance(relation, dict) or relation.get("rel_type") != "m.replace"
                    or relation.get("event_id") != target):
                return None
            revised = source_content.get("m.new_content")
        else:
            return None
        if not isinstance(revised, dict):
            return None
        msgtype, body = revised.get("msgtype"), revised.get("body")
        if (msgtype != "m.text" and not (msgtype == "m.notice" and self._process_notices)):
            return None
        if not isinstance(body, str) or not body.strip():
            return None
        if self._event_context_cache.is_redacted(room_id, event_id):
            return None
        return revised

    async def _handle_edit_message(
        self, room_id: str, sender: str, event_id: str, source_content: dict, relates_to: dict,
        *, typed_content: bool = False,
    ) -> None:
        if room_id not in self._process_edits:
            return
        target = relates_to.get("event_id")
        if not isinstance(target, str) or not target:
            return
        if typed_content:
            raw = await self._edit_raw_event(room_id, event_id)
            if raw is None or raw.get("sender") != sender:
                return
            revised = await self._edit_new_content(raw, target)
            source_content = {"m.new_content": revised}
        revised = source_content.get("m.new_content")
        if not isinstance(revised, dict):
            return
        msgtype = revised.get("msgtype")
        if msgtype != "m.text" and not (msgtype == "m.notice" and self._process_notices):
            return
        body = revised.get("body")
        if not isinstance(body, str) or not body.strip():
            return
        original = await self._edit_original_content(room_id, sender, target)
        if original is None or (original.get("msgtype") == "m.notice" and not self._process_notices):
            return
        relation = original.get("m.relates_to")
        relation = relation if isinstance(relation, dict) else {}
        revised = {key: value for key, value in revised.items() if key != "m.relates_to"}
        if relation:
            revised["m.relates_to"] = relation
        ctx = await self._resolve_message_context(
            room_id, sender, target, body, revised, relation, allow_gateway_control=False,
        )
        if ctx is None:
            return
        ctx[-1].message_id = event_id
        event = await self._build_inbound_event(
            room_id, sender, event_id, body, revised, relation, ctx=ctx,
            reply_anchor_override=target,
            metadata={"edited_message": True, "edited_message_original_id": target},
            allow_gateway_control=False,
        )
        if event is None:
            return
        if self._is_sender_authorized(sender, chat_type=event.source.chat_type, chat_id=room_id) is not True:
            return
        event.message_type = MessageType.TEXT
        event._queue_at_turn_boundary = True
        event._pending_coalesce_key = ("matrix-edit", room_id, sender, target)
        await self.handle_message(event)
