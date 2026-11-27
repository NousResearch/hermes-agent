"""Bounded plain-dict Telegram events; authorization remains gateway-owned."""

from datetime import UTC, datetime
from typing import Any, Optional


def _is_id_like(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (str, int))


def _id(value: Any) -> str | None:
    return str(value)[:128] if _is_id_like(value) else None


def _actor_metadata(update, event) -> dict[str, Any]:
    user = getattr(event, "user", None) or getattr(event, "from_user", None)
    actor_chat = getattr(event, "actor_chat", None) or getattr(
        event, "sender_chat", None
    )
    is_bot = getattr(user, "is_bot", None)
    date = getattr(event, "date", None)
    occurred_at = None
    if isinstance(date, datetime) and date.utcoffset() is not None:
        occurred_at = date.astimezone(UTC).isoformat()
    return {
        "update_id": _id(getattr(update, "update_id", None)),
        "actor_id": _id(getattr(user, "id", None)),
        "actor_is_bot": is_bot if isinstance(is_bot, bool) else None,
        "actor_chat_id": _id(getattr(actor_chat, "id", None)),
        "occurred_at": occurred_at,
    }


def normalize_reaction_event(update) -> Optional[dict[str, Any]]:
    """``message_reaction`` → ``reaction`` event: emojis (unicode), custom_emoji_ids, chat_id,
    message_id, thread_id (always None — reactions carry none)."""
    mr = getattr(update, "message_reaction", None)
    if mr is None:
        return None
    chat = getattr(mr, "chat", None)
    new_reaction = getattr(mr, "new_reaction", None) or []
    if not isinstance(new_reaction, (list, tuple)):
        return None
    chat_id = getattr(chat, "id", None) if chat is not None else None
    message_id = getattr(mr, "message_id", None)
    if not _is_id_like(chat_id) or not _is_id_like(message_id):
        return None
    emojis: list[str] = []
    custom_emoji_ids: list[str] = []
    for r in new_reaction[:64]:
        emoji = getattr(r, "emoji", None)
        if isinstance(emoji, str) and emoji:
            emojis.append(emoji[:64])
        custom_id = getattr(r, "custom_emoji_id", None)
        if _is_id_like(custom_id):
            custom_emoji_ids.append(str(custom_id)[:128])
    return {
        "platform": "telegram",
        "event_type": "reaction",
        "payload": {
            **_actor_metadata(update, mr),
            "emojis": emojis,
            "custom_emoji_ids": custom_emoji_ids,
            "chat_id": str(chat_id)[:128],
            "message_id": str(message_id)[:128],
            "thread_id": None,
        },
    }


def normalize_message_reply_event(update) -> dict[str, Any] | None:
    """New ``message`` that replies to another message → ``message_reply`` event. Other new
    messages produce no event: core handlers own them, and observers get no general message feed."""
    message = getattr(update, "message", None)
    if message is None:
        return None
    reply_to_message_id = _id(
        getattr(getattr(message, "reply_to_message", None), "message_id", None)
    )
    if reply_to_message_id is None:
        return None
    chat_id = _id(getattr(getattr(message, "chat", None), "id", None))
    message_id = _id(getattr(message, "message_id", None))
    if chat_id is None or message_id is None:
        return None
    text = getattr(message, "text", None)
    thread_id = (
        _id(getattr(message, "message_thread_id", None))
        if getattr(message, "is_topic_message", False)
        else None
    )
    return {
        "platform": "telegram",
        "event_type": "message_reply",
        "payload": {
            **_actor_metadata(update, message),
            "chat_id": chat_id,
            "message_id": message_id,
            "thread_id": thread_id,
            "reply_to_message_id": reply_to_message_id,
            "text": text[:4096] if isinstance(text, str) else None,
            "forwarded": getattr(message, "forward_origin", None) is not None
            or bool(getattr(message, "is_automatic_forward", False)),
        },
    }


def normalize_message_edited_event(update) -> Optional[dict[str, Any]]:
    """``edited_message`` → ``message_edited`` event (v1, additive): chat_id, message_id, thread_id
    (forum topic), text (edited text or caption, bounded), edited_at (ISO 8601 UTC or None)."""
    message = getattr(update, "edited_message", None)
    if message is None:
        return None
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None) if chat is not None else None
    message_id = getattr(message, "message_id", None)
    if not _is_id_like(chat_id) or not _is_id_like(message_id):
        return None
    text = getattr(message, "text", None) or getattr(message, "caption", None)
    if not isinstance(text, str):
        text = None
    thread_id = None
    thread_id_raw = getattr(message, "message_thread_id", None)
    if _is_id_like(thread_id_raw) and bool(getattr(message, "is_topic_message", False)):
        thread_id = str(thread_id_raw)[:128]
    edited_at = None
    edit_date = getattr(message, "edit_date", None)
    if isinstance(edit_date, datetime):
        edited_at = edit_date.isoformat()[:64]
    return {
        "platform": "telegram",
        "event_type": "message_edited",
        "payload": {
            "chat_id": str(chat_id)[:128],
            "message_id": str(message_id)[:128],
            "thread_id": thread_id,
            "text": text[:8192] if text is not None else None,
            "edited_at": edited_at,
        },
    }


_NORMALIZERS = {
    "message_reaction": normalize_reaction_event,
    "message": normalize_message_reply_event,
    "edited_message": normalize_message_edited_event,
}


def normalize_platform_event(update) -> dict[str, Any] | None:
    for attribute, normalize in _NORMALIZERS.items():
        if getattr(update, attribute, None) is not None:
            return normalize(update)
    return None
