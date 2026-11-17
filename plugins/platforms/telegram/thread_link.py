"""Telegram's ``thread_url``: the t.me link the session lists open. Kept apart from adapter.py,
which is over its size cap."""

import re

# Supergroups and channels carry a ``-100`` prefix the t.me/c/ form drops; basic groups and
# private chats have no public message link at all.
_SUPERGROUP_RE = re.compile(r"-100(\d+)")
_USERNAME_RE = re.compile(r"@?([A-Za-z][A-Za-z0-9_]{3,31})")
_NUMBER_RE = re.compile(r"\d+")


def _number(value) -> str:
    text = str(value or "")
    return text if _NUMBER_RE.fullmatch(text) else ""


def telegram_thread_url(source) -> str | None:
    """``https://t.me/c/<id>[/<topic>]/<message>`` for a supergroup or channel, or
    ``https://t.me/<username>/<message>`` for a public one; None for private chats and basic groups,
    which Telegram gives no link."""
    chat = str(source.chat_id or "").strip()
    message, topic = _number(source.message_id), _number(source.thread_id)
    if supergroup := _SUPERGROUP_RE.fullmatch(chat):
        path = "/".join(part for part in (topic, message) if part)
        return f"https://t.me/c/{supergroup.group(1)}/{path}" if path else None
    if chat.startswith("@") and (username := _USERNAME_RE.fullmatch(chat)):
        return f"https://t.me/{username.group(1)}" + (f"/{message}" if message else "")
    return None
