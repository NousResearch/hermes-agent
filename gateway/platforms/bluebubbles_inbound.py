"""BlueBubbles webhook hydration, routing, and admission ownership."""

import asyncio
import json
import logging
import re
import time
from copy import deepcopy
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote

import httpx

from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger(__name__)

_TAPBACK_CODES = {*range(2000, 2006), *range(3000, 3006)}
_MESSAGE_EVENTS = {"new-message", "message", "updated-message"}
_HYDRATE_RETRY_DELAYS = (0.0, 0.05, 0.15)
_INBOUND_CACHE_SIZE = 2000
_INBOUND_CACHE_TTL = 300
_PHONE_RE = re.compile(r"\+?\d{7,15}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


@dataclass
class _InboundMessage:
    record: Dict[str, Any] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    touched: float = field(default_factory=time.monotonic)
    accepted: bool = False
    delivered: set[str] = field(default_factory=set)
    downloaded: Dict[str, tuple[str, str]] = field(default_factory=dict)


def _ok():
    from aiohttp import web
    return web.Response(text="ok")


class BlueBubblesInboundMixin:
    def _extract_payload_record(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and (first := next((i for i in data if isinstance(i, dict)), None)):
            return first
        if isinstance(payload.get("message"), dict):
            return payload.get("message")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _value(*candidates: Any) -> Optional[str]:
        return next((c.strip() for c in candidates if isinstance(c, str) and c.strip()), None)

    @staticmethod
    def _parse_webhook_body(raw: bytes) -> Any:
        """Decode a webhook body: JSON, else form-encoded with a JSON field."""
        body = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            form = parse_qs(body)
            payload_str = (form.get("payload") or form.get("data") or form.get("message") or [""])[0]
            return json.loads(payload_str) if payload_str else {}

    async def _collect_attachments(self, record: Dict[str, Any], *,
                                   state: Optional[_InboundMessage] = None):
        """Download inbound attachments and report the GUIDs included in this delivery."""
        media_urls: List[str] = []
        media_types: List[str] = []
        collected: set[str] = set()
        msg_type = MessageType.TEXT
        for att in record.get("attachments") or []:
            if not isinstance(att, dict):
                continue
            att_guid = att.get("guid", "")
            if not att_guid or (state is not None and att_guid in state.delivered):
                continue
            if att.get("transferState") not in (None, 5, "5"):
                continue
            mime = (att.get("mimeType") or "").lower()
            prior = state.downloaded.get(att_guid) if state is not None else None
            cached = prior[0] if prior and prior[1] == mime else None
            if not cached:
                for delay in _HYDRATE_RETRY_DELAYS:
                    if delay:
                        await asyncio.sleep(delay)
                    cached = await self._download_attachment(att_guid, att)
                    if cached:
                        break
            if not cached:
                continue
            mime = (att.get("mimeType") or "").lower()
            if state is not None:
                state.downloaded[att_guid] = cached, mime
            media_urls.append(cached)
            media_types.append(mime)
            collected.add(att_guid)
            is_voice = mime.startswith("audio/") or (att.get("uti") or "").endswith("caf")
            msg_type = (MessageType.PHOTO if mime.startswith("image/") else MessageType.VOICE if is_voice
                        else MessageType.VIDEO if mime.startswith("video/") else MessageType.DOCUMENT)
        if len(media_urls) > 1 and any(m.split("/")[0] == "image" for m in media_types):  # any image → PHOTO
            msg_type = MessageType.PHOTO
        return media_urls, media_types, msg_type, collected

    def _webhook_token(self, request) -> Optional[str]:
        return (request.query.get("password") or request.query.get("guid") or request.headers.get("x-password")
                or request.headers.get("x-guid") or request.headers.get("x-bluebubbles-guid"))

    def _resolve_chat_and_sender(self, payload: Dict[str, Any], record: Dict[str, Any]):
        """Return ``(chat_guid, chat_identifier, sender)`` across BlueBubbles payload shapes."""
        chat_guid = self._value(record.get("chatGuid"), payload.get("chatGuid"), record.get("chat_guid"),
                                payload.get("chat_guid"))
        # ``guid`` on payload/record is the message GUID, never a chat GUID. BlueBubbles v1.9+
        # commonly nests chat identity under data.chats[0], including private API ``[auth-key]``.
        chats = record.get("chats") or []
        first_chat = chats[0] if chats and isinstance(chats[0], dict) else {}
        if not chat_guid:
            chat_guid = self._value(first_chat.get("guid"), first_chat.get("chatGuid"),
                                    first_chat.get("[auth-key]"))
        chat_identifier = self._value(
            record.get("chatIdentifier"), record.get("identifier"),
            payload.get("chatIdentifier"), payload.get("identifier"),
            first_chat.get("chatIdentifier"), first_chat.get("identifier"))
        handle = record.get("handle")
        sender = (self._value(handle.get("address") if isinstance(handle, dict) else None, record.get("sender"),
                              record.get("from"), record.get("address")) or chat_identifier or chat_guid)
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        return chat_guid, chat_identifier, sender

    @staticmethod
    def _is_group_record(record: Dict[str, Any], chat_guid: Optional[str]) -> bool:
        if record.get("isGroup") or ";+;" in (chat_guid or ""):
            return True
        for chat in record.get("chats") or []:
            if not isinstance(chat, dict):
                continue
            with suppress(TypeError, ValueError):
                # BlueBubbles v1.9.9: group chats are style 43; one-to-one DMs are style 45.
                if int(chat.get("style") or 0) == 43:
                    return True
        return False

    @staticmethod
    def _canonical_session_chat_id(chat_guid: Optional[str], chat_identifier: Optional[str],
                                   sender: Optional[str], is_group: bool) -> Optional[str]:
        """Collapse BlueBubbles DM GUID aliases onto the address while preserving group GUIDs."""
        if is_group:
            return chat_guid
        address = chat_guid.split(";-;", 1)[-1] if chat_guid and ";-;" in chat_guid else None
        if chat_identifier and (chat_identifier in (sender, address)
                                or _PHONE_RE.fullmatch(chat_identifier) or _EMAIL_RE.fullmatch(chat_identifier)):
            return chat_identifier
        if sender and ";" not in sender:
            return sender
        if address:
            return address
        return chat_guid

    @staticmethod
    def _has_real_chat_guid(chat_guid: Optional[str]) -> bool:
        return bool(chat_guid and (";-;" in chat_guid or ";+;" in chat_guid))

    @staticmethod
    def _is_receipt_only_update(event_type: str, record: Dict[str, Any]) -> bool:
        # Receipt snapshots can also carry previously unseen content and completed attachments.
        return (event_type == "updated-message"
                and not any(record.get(key) for key in ("text", "message", "body", "attachments")))

    def _has_pending_inbound_content(self, message_id: Optional[str]) -> bool:
        state = self._inbound_messages.get(message_id)
        if state is None:
            return False
        if not state.accepted:
            return any(state.record.get(key) for key in ("text", "message", "body", "attachments"))
        return any(isinstance(att, dict) and att.get("guid") and att["guid"] not in state.delivered
                   for att in state.record.get("attachments") or [])

    async def _hydrate_inbound_record(self, message_id: Optional[str]) -> Dict[str, Any]:
        """Fetch routing and attachment relationships with bounded in-request retries.

        BlueBubbles does not retry failed webhook POSTs, so transient API/relationship lag has to be
        absorbed before this request is acknowledged.
        """
        if not message_id or not self.client:
            return {}
        last_error: Optional[Exception] = None
        last_record: Dict[str, Any] = {}
        for delay in _HYDRATE_RETRY_DELAYS:
            if delay:
                await asyncio.sleep(delay)
            try:
                data = (await self._api_get(
                    f"/api/v1/message/{quote(message_id, safe='')}?with=chats,attachments")).get("data")
                last_error = None
            except (httpx.HTTPError, OSError, ValueError) as exc:
                last_error = exc
                continue
            if isinstance(data, dict):
                last_record = data
                chats = [chat for chat in (data.get("chats") or []) if isinstance(chat, dict)]
                if chats and all(att.get("transferState") in (None, 5, "5")
                                 for att in (data.get("attachments") or []) if isinstance(att, dict)):
                    return data
        if last_error is not None:
            raise last_error
        return last_record

    @staticmethod
    def _merge_inbound_record(target: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        """Refresh metadata by identity without erasing fields omitted by sparse echoes."""
        for key, value in incoming.items():
            if value in (None, "", []):
                continue
            if key in {"attachments", "chats"} and isinstance(value, list):
                current = target.setdefault(key, [])
                seen = {item.get("guid") or item.get("[auth-key]") or repr(item): item
                        for item in current if isinstance(item, dict)}
                for item in value:
                    identity = (item.get("guid") or item.get("[auth-key]") or repr(item)
                                if isinstance(item, dict) else repr(item))
                    if identity in seen and isinstance(item, dict):
                        BlueBubblesInboundMixin._merge_inbound_record(seen[identity], item)
                    else:
                        copied = deepcopy(item)
                        current.append(copied)
                        if isinstance(copied, dict):
                            seen[identity] = copied
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                BlueBubblesInboundMixin._merge_inbound_record(target[key], value)
            else:
                target[key] = deepcopy(value)

    def _get_inbound_message(self, message_id: Optional[str], record: Dict[str, Any]) -> _InboundMessage:
        now = time.monotonic()
        for key, state in list(self._inbound_messages.items()):
            if not state.users and now - state.touched >= _INBOUND_CACHE_TTL:
                self._inbound_messages.pop(key)
        state = self._inbound_messages.get(message_id) if message_id else None
        if state is None:
            state = _InboundMessage()
            if message_id:
                for key, old in list(self._inbound_messages.items()):
                    if len(self._inbound_messages) < _INBOUND_CACHE_SIZE:
                        break
                    if not old.users:
                        self._inbound_messages.pop(key)
                self._inbound_messages[message_id] = state
        if message_id:
            self._inbound_messages.move_to_end(message_id)
        state.touched = now
        state.users += 1
        # Merge before waiting: an update can enrich the owner across hydration/download awaits.
        self._merge_inbound_record(state.record, record)
        return state

    def _reserve_inbound_turn(self, chat_id: str):
        previous = self._inbound_chat_tails.get(chat_id)
        current = asyncio.get_running_loop().create_future()
        self._inbound_chat_tails[chat_id] = current
        return previous, current

    def _finish_inbound_turn(self, chat_id: str, previous, current) -> None:
        def release(_=None):
            if not current.done():
                current.set_result(None)
            if self._inbound_chat_tails.get(chat_id) is current:
                self._inbound_chat_tails.pop(chat_id, None)
        # A cancelled waiter must not let its successor overtake a still-running predecessor.
        if previous is not None and not previous.done():
            previous.add_done_callback(release)
        else:
            release()

    async def _route_inbound_message(self, payload, state, message_id):
        record = state.record
        chat_guid, chat_identifier, sender = self._resolve_chat_and_sender(payload, record)
        if not self._has_real_chat_guid(chat_guid):
            try:
                hydrated = await self._hydrate_inbound_record(message_id)
                self._merge_inbound_record(record, hydrated)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                logger.warning("[bluebubbles] inbound hydration failed: %s", type(exc).__name__)
            chat_guid, chat_identifier, sender = self._resolve_chat_and_sender(payload, record)
        if not self._has_real_chat_guid(chat_guid):
            logger.warning("[bluebubbles] ignoring message with no resolvable chat GUID")
            return None
        is_group = self._is_group_record(record, chat_guid)
        chat_id = self._canonical_session_chat_id(chat_guid, chat_identifier, sender, is_group)
        if not sender or not chat_id:
            return None
        if not is_group:
            self._remember_chat_guid(chat_id, chat_guid)
        return chat_guid, chat_id, chat_identifier, sender, is_group

    async def _deliver_inbound_message(self, payload, state, message_id, route):
        from aiohttp import web

        chat_guid, chat_id, chat_identifier, sender, is_group = route
        record = state.record
        text = self._value(record.get("text"), record.get("message"), record.get("body")) or ""
        # Known unmentioned group text needs no media I/O; sparse text can still hydrate below.
        if is_group and self.require_mention and text and not self._message_matches_mention_patterns(text):
            return _ok()
        if record.get("attachments") and self.client:
            try:
                self._merge_inbound_record(record, await self._hydrate_inbound_record(message_id))
            except (httpx.HTTPError, OSError, ValueError) as exc:
                logger.warning("[bluebubbles] attachment hydration failed: %s", type(exc).__name__)
        text = self._value(record.get("text"), record.get("message"), record.get("body")) or ""
        if is_group and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                return _ok()
            text = self._clean_mention_text(text)
        media_urls, media_types, msg_type, collected = await self._collect_attachments(record, state=state)
        attachments = {att.get("guid") for att in record.get("attachments") or []
                       if isinstance(att, dict) and att.get("guid")} - state.delivered
        if not attachments <= collected:
            # Keep successful downloads for a later completion; don't consume the caption alone.
            return web.json_response({"error": "attachments not ready"}, status=503)
        if state.accepted:
            if not media_urls:
                return _ok()
            text = "(attachment)"
        elif not text and media_urls:
            text = "(attachment)"
        if not text:
            return web.json_response({"error": "missing message fields"}, status=400)
        chats = record.get("chats") or []
        chat_name = self._value(chats[0].get("displayName") if chats and isinstance(chats[0], dict) else None,
                                chat_identifier, chat_id)
        source = self.build_source(chat_id=chat_id, chat_name=chat_name,
                                   chat_type="group" if is_group else "dm", user_id=sender, user_name=sender,
                                   chat_id_alt=chat_identifier)
        event = MessageEvent(
            text=text, message_type=msg_type, source=source,
            raw_message={**payload, "data": deepcopy(record)}, message_id=message_id,
            reply_to_message_id=self._value(record.get("threadOriginatorGuid"), record.get("associatedMessageGuid")),
            media_urls=media_urls, media_types=media_types)
        try:
            await self.handle_message(event)
        finally:
            # Admission, not model/output completion, is the point of no replay.
            if event._gateway_accepted:
                state.accepted = True
                state.delivered.update(attachments)
                for guid in attachments:
                    state.downloaded.pop(guid, None)
        if not event._gateway_accepted:
            return web.json_response({"error": "gateway did not accept message"}, status=503)
        if self.send_read_receipts:
            task = asyncio.create_task(self.mark_read(chat_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        return _ok()

    async def _handle_webhook(self, request):
        from aiohttp import web

        if self._webhook_token(request) != self.password:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            payload = self._parse_webhook_body(await request.read())
            if not isinstance(payload, dict):
                raise ValueError("webhook body must be an object")
        except Exception:
            logger.exception("[bluebubbles] webhook parse error")
            return web.json_response({"error": "invalid payload"}, status=400)
        event_type = self._value(payload.get("type"), payload.get("event")) or ""
        if event_type and event_type not in _MESSAGE_EVENTS:
            return _ok()
        record = self._extract_payload_record(payload) or {}
        if record.get("isFromMe") or record.get("fromMe") or record.get("is_from_me"):
            return _ok()
        assoc_type = record.get("associatedMessageType")
        if isinstance(assoc_type, int) and assoc_type in _TAPBACK_CODES:
            return _ok()
        message_id = self._value(record.get("guid"), record.get("messageGuid"), record.get("id"))
        # A routing-only update can complete a retained message after hydration failed.
        if (self._is_receipt_only_update(event_type, record)
                and not self._has_pending_inbound_content(message_id)):
            return _ok()
        state = self._get_inbound_message(message_id, record)
        try:
            async with state.lock:
                attachments = {att.get("guid") for att in state.record.get("attachments") or []
                               if isinstance(att, dict) and att.get("guid")}
                if state.accepted and attachments <= state.delivered:
                    return _ok()
                # Resolve sparse routes in order; attachment work runs outside the routing lock.
                async with self._inbound_routing_lock:
                    route = await self._route_inbound_message(payload, state, message_id)
                    if route is None:
                        return _ok()
                    previous, current = self._reserve_inbound_turn(route[1])
                try:
                    if previous is not None:
                        await asyncio.shield(previous)
                    return await self._deliver_inbound_message(payload, state, message_id, route)
                finally:
                    self._finish_inbound_turn(route[1], previous, current)
        finally:
            state.users -= 1
            state.touched = time.monotonic()
