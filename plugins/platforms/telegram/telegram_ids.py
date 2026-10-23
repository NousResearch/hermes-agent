"""Helpers for Telegram Bot API chat identifiers: a ``chat_id`` is a numeric ID (int) or an ``@username``
string for public channels/groups; a bare ``int(chat_id)`` crashes on the username form."""

from __future__ import annotations

import re
from typing import Any, Union

from gateway.platforms.send_targets import parse_nonempty, regex_groups_parser

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


_SEND_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")


def parse_send_target(ref):
    """``target_ref_syntax_fn`` for the Telegram ``PlatformEntry``."""
    # "<chat_id>[:<topic_id>]" or an @username (usernames must not be force-int'd).
    parsed = regex_groups_parser(_SEND_TARGET_RE)(ref)
    if parsed:
        return parsed
    return parse_nonempty(parse_telegram_username_target(ref) or "")


def parse_telegram_message_id(value: Any) -> int | None:
    """A reply anchor as a Telegram message id (a positive int), or ``None`` when it is not one.

    Anchors are opaque event ids upstream: a route wake (kanban-wake, loop-tick) carries a synthetic
    JSON identity, and raw ``int()`` on it failed every reply into the session (#139611); an
    unanchored reply is strictly better than a failed send."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip() if value is not None else ""
    return int(text) if text.isascii() and text.isdigit() and int(text) > 0 else None
