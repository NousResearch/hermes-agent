"""Telegram bodies for ``gateway_ingress_observer`` events (machinery: ``gateway/ingress_observer.py``)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gateway.ingress_observer import MAX_CONTENT_BYTES, IngressEpoch

_UPDATE_BYTES = 96  # accounted per fetched update entry
# Attribute order matters: an animation also carries ``document`` and a venue ``location``.
_MEDIA_KINDS = (
    "animation", "audio", "contact", "dice", "game", "invoice", "paid_media", "photo", "poll", "sticker", "story",
    "venue", "location", "video", "video_note", "voice", "document")


def fetched_fields(epoch: IngressEpoch, event_no: int, result: Any) -> tuple[dict[str, Any], int]:
    """One ``{update_id, raw_sha256}`` per element of the parsed raw ``getUpdates`` result, fields
    PTB does not model included: sha256 of ``json.dumps(element, sort_keys=True, separators=(",", ":"))``."""
    updates = []
    for raw in result if isinstance(result, list) else ():
        raw_sha256 = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        update_id = raw.get("update_id") if isinstance(raw, dict) else None
        if type(update_id) is int:
            epoch.associate(update_id, event_no, raw_sha256)
        else:
            update_id = None
        updates.append({"update_id": update_id, "raw_sha256": raw_sha256})
    return {"updates": updates}, len(updates) * _UPDATE_BYTES


def observed_fields(epoch: IngressEpoch, event_no: int, adapter: Any, update: Any) -> tuple[dict[str, Any], int]:
    """An update reaching the group-99 handler. ``authorized`` is the runner-installed sender
    check (synchronous, read-only) for a message or edit, else ``None``; content only when ``True``."""
    chat, user = update.effective_chat, update.effective_user
    message = update.message or update.edited_message
    authorized = None
    if message is not None:
        source = adapter._source_from_message_for_auth(message)
        authorized = adapter._is_sender_authorized(
            source.user_id, source.chat_type, source.chat_id, is_bot=source.is_bot, thread_id=source.thread_id)
    fields = {
        "update_id": update.update_id, **epoch.resolve(update.update_id),
        "chat_id": str(chat.id) if chat else None, "user_id": str(user.id) if user else None,
        "authorized": authorized, "message": None,
    }
    if not authorized:
        return fields, 0
    fields["message"], size = _message_fields(message)
    return fields, size


def _message_fields(message: Any) -> tuple[dict[str, Any], int]:
    text, caption = message.text, message.caption
    size = sum(len(part.encode("utf-8", "surrogatepass")) for part in (text, caption) if part)
    omitted = size > MAX_CONTENT_BYTES
    if omitted:
        text = caption = None
        size = 0
    reply = message.reply_to_message
    return {
        "message_id": str(message.message_id), "date": int(message.date.timestamp()),
        "edit_date": int(message.edit_date.timestamp()) if message.edit_date else None,
        "text": text, "caption": caption, "content_omitted": omitted,
        "reply_to_message_id": str(reply.message_id) if reply else None,
        "is_forward": message.forward_origin is not None, "has_quote": message.quote is not None,
        "media_kind": next((kind for kind in _MEDIA_KINDS if getattr(message, kind, None)), None),
    }, size
