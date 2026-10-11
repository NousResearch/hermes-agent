"""``gateway_platform_event`` envelopes for Telegram updates (hooks.md per-event contracts)."""

from __future__ import annotations

from typing import Any, Optional


def _is_id_like(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (str, int))


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
            "emojis": emojis, "custom_emoji_ids": custom_emoji_ids, "chat_id": str(chat_id)[:128],
            "message_id": str(message_id)[:128], "thread_id": None},
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
    try:
        if edit_date is not None and hasattr(edit_date, "isoformat"):
            edited_at = str(edit_date.isoformat())[:64]
    except Exception:  # health: allow BLE001 -- moved verbatim from the adapter; an odd edit_date leaves edited_at unset
        edited_at = None
    return {
        "platform": "telegram",
        "event_type": "message_edited",
        "payload": {
            "chat_id": str(chat_id)[:128], "message_id": str(message_id)[:128], "thread_id": thread_id,
            "text": text[:8192] if text is not None else None, "edited_at": edited_at},
    }
