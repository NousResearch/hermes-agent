"""Helpers for Telegram Bot API chat and message identifiers: a ``chat_id`` is a numeric ID (int) or an
``@username`` string for public channels/groups; a bare ``int(chat_id)`` crashes on the username form."""

from __future__ import annotations

import re
from typing import Any, Union

from gateway.platforms.event import MessageOrigin

# Usernames are 5-32 chars (letters, digits, underscores) with a leading "@"; 4-char legacy handles are tolerated.
_TELEGRAM_USERNAME_RE = re.compile(r"@[A-Za-z0-9_]{4,32}")


def normalize_telegram_chat_id(chat_id: Any) -> int | str:
    """Bot API-compatible chat_id: numeric values (incl. negative channel IDs) as ``int``, anything
    else (e.g. ``@username``) as a stripped string; never raises."""
    chat_id_str = str(chat_id).strip()
    try:
        return int(chat_id_str)
    except (TypeError, ValueError):
        return chat_id_str


def looks_like_telegram_username(chat_id: Any) -> bool:
    """True when the value is an ``@username``-format Telegram chat identifier."""
    return bool(_TELEGRAM_USERNAME_RE.fullmatch(str(chat_id).strip()))


def parse_telegram_username_target(target_ref: Any) -> str | None:
    """Return the value when it is an ``@username`` target, else ``None``."""
    value = str(target_ref).strip()
    return value if looks_like_telegram_username(value) else None


def message_origin_fields(message: Any, update_id: int | None) -> dict[str, Any]:
    """``MessageEvent`` origin fields for one inbound message revision. Its raw chat, message and
    update ids must all be present: a missing one leaves the event unidentified (the defaults),
    never a stringified ``None``. Telegram sends ``edit_date`` only for an edit, so an ordinary
    message is identified with it None."""
    chat_id, message_id = message.chat.id, message.message_id
    if chat_id is None or message_id is None or update_id is None:
        return {}
    edit_date = getattr(message, "edit_date", None)
    origin = MessageOrigin(str(chat_id), str(message_id), update_id, int(edit_date.timestamp()) if edit_date else None)
    return {"source_origins": (origin,), "source_origins_complete": True}
