"""Helpers for Telegram Bot API chat identifiers.

Telegram's Bot API accepts a ``chat_id`` in two forms: a numeric ID (an int,
e.g. ``123456789`` for a DM or ``-1001234567890`` for a channel/supergroup) or
an ``@username`` string for public channels and groups. Hermes historically
coerced every ``chat_id`` with ``int()``, which crashes on the username form
(``ValueError: invalid literal for int()``). Normalizing here lets numeric IDs
pass through as ints while usernames pass through unchanged — both are valid
values for the Bot API.
"""

from __future__ import annotations

import re
from typing import Any, Union

# Telegram usernames are 5-32 chars: letters, digits, underscores, with a
# leading "@". (Telegram also permits 4-char usernames for some legacy/official
# accounts, but the 5-32 public rule is the safe lower bound for routing.)
_TELEGRAM_USERNAME_RE = re.compile(r"@[A-Za-z0-9_]{4,32}")


def normalize_telegram_chat_id(chat_id: Any) -> Union[int, str]:
    """Return a Bot API-compatible chat_id.

    Numeric values (incl. negative channel IDs) are returned as ``int``; any
    non-numeric value (e.g. an ``@username``) is returned as a stripped string.
    Telegram's Bot API accepts both, so this never raises on a username the way
    a bare ``int(chat_id)`` would.
    """
    chat_id_str = str(chat_id).strip()
    try:
        return int(chat_id_str)
    except (TypeError, ValueError):
        return chat_id_str


def telegram_chat_id_key(chat_id: Any) -> str:
    """Stable string key for a chat_id (for dict keys / persisted state)."""
    return str(normalize_telegram_chat_id(chat_id))


def looks_like_telegram_username(chat_id: Any) -> bool:
    """True when the value is an ``@username``-format Telegram chat identifier."""
    return bool(_TELEGRAM_USERNAME_RE.fullmatch(str(chat_id).strip()))


def parse_telegram_username_target(target_ref: Any) -> Union[str, None]:
    """Return the value when it is an ``@username`` target, else ``None``."""
    value = str(target_ref).strip()
    return value if looks_like_telegram_username(value) else None
