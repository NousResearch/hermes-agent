"""Helpers for Telegram Bot API chat identifiers: a ``chat_id`` is a numeric ID (int) or an ``@username``
string for public channels/groups; a bare ``int(chat_id)`` crashes on the username form."""

from __future__ import annotations

import re
from typing import Any, Union

# Usernames are 5-32 chars (letters, digits, underscores) with a leading "@"; 4-char legacy handles are tolerated.
_TELEGRAM_USERNAME_RE = re.compile(r"@[A-Za-z0-9_]{4,32}")


def normalize_telegram_chat_id(chat_id: Any) -> Union[int, str]:
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


def parse_telegram_username_target(target_ref: Any) -> Union[str, None]:
    """Return the value when it is an ``@username`` target, else ``None``."""
    value = str(target_ref).strip()
    return value if looks_like_telegram_username(value) else None
