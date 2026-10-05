"""BlueBubbles inbound policy, attachment recovery and GUID admission."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger(__name__)

# Tapback reaction codes (BlueBubbles associatedMessageType values)
_TAPBACK_ADDED = {
    2000: "love",
    2001: "like",
    2002: "dislike",
    2003: "laugh",
    2004: "emphasize",
    2005: "question",
}
_TAPBACK_REMOVED = {
    3000: "love",
    3001: "like",
    3002: "dislike",
    3003: "laugh",
    3004: "emphasize",
    3005: "question",
}
# Only new-message (plus the legacy message alias) starts an agent turn.
# BlueBubbles emits updated-message for receipt, delivery, and attachment state
# changes, often with a different chat GUID shape for the same iMessage.
_MESSAGE_EVENTS = {"new-message", "message"}


def _ok():
    """Plain ``ok`` acknowledgement for webhook events we accept but don't process."""
    from aiohttp import web

    return web.Response(text="ok")


def _attachment_message_type(attachment, mime):
    if mime.startswith("image/"):
        return MessageType.PHOTO
    if mime.startswith("audio/") or (attachment.get("uti") or "").endswith("caf"):
        return MessageType.VOICE
    if mime.startswith("video/"):
        return MessageType.VIDEO
    return MessageType.DOCUMENT


class BlueBubblesInboundMixin:
    def _extract_payload_record(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and (
            first := next((i for i in data if isinstance(i, dict)), None)
        ):
            return first
        if isinstance(payload.get("message"), dict):
            return payload.get("message")
        return payload if isinstance(payload, dict) else None

    def _claim_inbound_message(
        self, message_id: str
    ) -> tuple[Optional[asyncio.Future], bool]:
        """Reserve a GUID, or return the existing in-flight outcome."""
        if self._message_dedup.contains(message_id):
            return None, False
        existing = self._inflight_message_ids.get(message_id)
        if existing is not None:
            return existing, False
        claim = asyncio.get_running_loop().create_future()
        self._inflight_message_ids[message_id] = claim
        return claim, True

    def _finish_inbound_claim(
        self,
        message_id: Optional[str],
        claim: Optional[asyncio.Future],
        *,
        accepted: bool,
    ) -> None:
        """Commit a successful handoff, or release a failed reservation."""
        if not message_id or claim is None:
            return
        if self._inflight_message_ids.get(message_id) is not claim:
            return
        self._inflight_message_ids.pop(message_id, None)
        if accepted:
            self._message_dedup.is_duplicate(message_id)
        if not claim.done():
            claim.set_result(accepted)

    @staticmethod
    def _value(*candidates: Any) -> Optional[str]:
        return next(
            (c.strip() for c in candidates if isinstance(c, str) and c.strip()), None
        )

    @staticmethod
    def _parse_webhook_body(raw: bytes) -> Any:
        """Decode a webhook body: JSON, else form-encoded with a JSON field."""
        body = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            form = parse_qs(body)
            payload_str = (
                form.get("payload") or form.get("data") or form.get("message") or [""]
            )[0]
            return json.loads(payload_str) if payload_str else {}

    def _webhook_token(self, request) -> Optional[str]:
        return (
            request.query.get("password")
            or request.query.get("guid")
            or request.headers.get("x-password")
            or request.headers.get("x-guid")
            or request.headers.get("x-bluebubbles-guid")
        )

    def _resolve_chat_and_sender(self, payload: Dict[str, Any], record: Dict[str, Any]):
        """Returns ``(chat_guid, chat_identifier, sender)`` from the many BlueBubbles payload shapes."""
        chat_guid = self._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        # BlueBubbles v1.9+ payloads omit top-level chatGuid; it's nested under data.chats[0].guid.
        _chats = record.get("chats") or []
        if not chat_guid and _chats and isinstance(_chats[0], dict):
            chat_guid = _chats[0].get("guid") or _chats[0].get("chatGuid")
        chat_identifier = self._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        handle = record.get("handle")
        sender = (
            self._value(
                handle.get("address") if isinstance(handle, dict) else None,
                record.get("sender"),
                record.get("from"),
                record.get("address"),
            )
            or chat_identifier
            or chat_guid
        )
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        return chat_guid, chat_identifier, sender

    async def _handle_webhook(self, request):
        from aiohttp import web

        if self._webhook_token(request) != self.password:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            payload = self._parse_webhook_body(await request.read())
        except (ValueError, TypeError, OSError) as exc:
            logger.error("[bluebubbles] webhook parse error: %s", exc)
            return web.json_response({"error": "invalid payload"}, status=400)
        event_type = self._value(payload.get("type"), payload.get("event")) or ""
        if (
            event_type and event_type not in _MESSAGE_EVENTS
        ):  # ack non-message events silently
            return _ok()
        record = self._extract_payload_record(payload) or {}
        if self._ignore_record(record):
            return _ok()
        message_id = self._value(
            record.get("guid"),
            record.get("messageGuid"),
            record.get("id"),
        )
        claim: Optional[asyncio.Future] = None
        if message_id:
            claim, is_owner = self._claim_inbound_message(message_id)
            if claim is None:
                logger.info("[bluebubbles] duplicate inbound message ignored")
                return web.Response(text="ok")
            if not is_owner:
                accepted = await asyncio.shield(claim)
                return web.Response(
                    text="ok" if accepted else "handoff unavailable",
                    status=200 if accepted else 503,
                )

        chat_guid, chat_identifier, sender = self._resolve_chat_and_sender(
            payload, record
        )
        is_group = bool(record.get("isGroup")) or (";+;" in (chat_guid or ""))
        text = self._inbound_text(record, is_group)
        if text is None:
            self._finish_inbound_claim(message_id, claim, accepted=True)
            return _ok()

        try:
            content = await self._inbound_content(record, text)
        except asyncio.CancelledError:
            self._finish_inbound_claim(message_id, claim, accepted=False)
            raise
        except ValueError as exc:
            self._finish_inbound_claim(message_id, claim, accepted=False)
            return web.json_response({"error": str(exc)}, status=400)
        text = content[0]

        if not sender or not (chat_guid or chat_identifier) or not text:
            self._finish_inbound_claim(message_id, claim, accepted=False)
            return web.json_response({"error": "missing message fields"}, status=400)
        return await self._handoff_inbound(
            payload,
            record,
            chat_guid,
            chat_identifier,
            sender,
            is_group,
            content,
            message_id,
            claim,
        )

    @staticmethod
    def _ignore_record(record):
        if record.get("isFromMe") or record.get("fromMe") or record.get("is_from_me"):
            return True
        assoc_type = record.get("associatedMessageType")
        if isinstance(assoc_type, int) and (
            assoc_type in _TAPBACK_ADDED or assoc_type in _TAPBACK_REMOVED
        ):  # tapback additions/removals delivered as messages
            return True
        return False

    def _inbound_text(self, record, is_group):
        text = (
            self._value(record.get("text"), record.get("message"), record.get("body"))
            or ""
        )
        if is_group and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                return None
            return self._clean_mention_text(text)
        return text

    async def _inbound_content(self, record, text):
        """Recover attachment siblings within the provider's single delivery."""
        attachments = record.get("attachments") or []
        if not isinstance(attachments, list):
            raise ValueError("invalid attachments")
        media_urls: List[str] = []
        media_types: List[str] = []
        msg_type = MessageType.TEXT
        attachment_failed = False

        for att in attachments:
            try:
                if not isinstance(att, dict):
                    attachment_failed = True
                    continue
                att_guid = att.get("guid", "")
                if not att_guid:
                    attachment_failed = True
                    continue
                cached = await self._download_attachment_with_retries(att_guid, att)
                if not cached:
                    attachment_failed = True
                    continue
                mime = (att.get("mimeType") or "").lower()
                media_urls.append(cached)
                media_types.append(mime)
                msg_type = _attachment_message_type(att, mime)
            except asyncio.CancelledError:
                raise
            except Exception:
                attachment_failed = True
                logger.exception(
                    "[bluebubbles] inbound attachment failed; preserving other content"
                )

        if attachment_failed:
            logger.warning(
                "[bluebubbles] one or more inbound attachments remained unavailable "
                "after bounded retries; preserving recoverable message content"
            )

        # With multiple attachments, prefer PHOTO if any images present
        if len(media_urls) > 1:
            mime_prefixes = {(m or "").split("/")[0] for m in media_types}
            if "image" in mime_prefixes:
                msg_type = MessageType.PHOTO

        if not text and media_urls:
            text = "(attachment)"
        if attachments and not text and not media_urls:
            # BlueBubbles will not redeliver this webhook. Preserve the user
            # turn even when every attachment remains unavailable so the agent
            # can acknowledge the failed media instead of silently losing it.
            text = "(attachment unavailable)"

        return text, msg_type, media_urls, media_types

    async def _handoff_inbound(
        self,
        payload,
        record,
        chat_guid,
        chat_identifier,
        sender,
        is_group,
        content,
        message_id,
        claim,
    ):
        """Settle the reserved GUID against the real gateway admission receipt."""
        from aiohttp import web

        text, msg_type, media_urls, media_types = content
        session_chat_id = chat_guid or chat_identifier
        event: Optional[MessageEvent] = None
        try:
            source = self.build_source(
                chat_id=session_chat_id,
                chat_name=chat_identifier or sender,
                chat_type="group" if is_group else "dm",
                user_id=sender,
                user_name=sender,
                chat_id_alt=chat_identifier,
            )
            event = MessageEvent(
                text=text,
                message_type=msg_type,
                source=source,
                raw_message=payload,
                message_id=message_id,
                reply_to_message_id=self._value(
                    record.get("threadOriginatorGuid"),
                    record.get("associatedMessageGuid"),
                ),
                media_urls=media_urls,
                media_types=media_types,
            )
            # BasePlatformAdapter.handle_message returns after accepting the
            # handoff and spawning agent work; awaiting it lets us release a
            # failed claim without waiting for the agent turn itself.
            await self.handle_message(event)
        except Exception:
            logger.exception("[bluebubbles] failed to hand off inbound message")
            return web.Response(text="handoff unavailable", status=503)
        finally:
            # Cancellation can arrive after admission while an inline reply is
            # sending. Retain that GUID so replay cannot execute the command again.
            accepted = event is not None and event._gateway_accepted is True
            self._finish_inbound_claim(message_id, claim, accepted=accepted)

        if not accepted:
            return web.Response(text="handoff unavailable", status=503)

        # Fire-and-forget read receipt
        if self.send_read_receipts and session_chat_id:
            receipt = asyncio.create_task(self.mark_read(session_chat_id))
            self._background_tasks.add(receipt)
            receipt.add_done_callback(self._finish_read_receipt)
        return _ok()

    def _finish_read_receipt(self, task):
        self._background_tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("[bluebubbles] read receipt failed", exc_info=error)
