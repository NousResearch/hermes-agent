"""Per-chat gate for Telegram typing keep-alives.

``_keep_typing`` runs once per session, but Telegram meters per chat: every DM topic shares one
chat id, so N active topics sent N ``sendChatAction`` calls per tick into one chat. A typing call
also spends the same per-chat budget as sends and edits: it stays quiet while that chat is in a
flood window or its shared outbound slot is taken, and takes that slot when it fires.
"""

from __future__ import annotations

import asyncio
from typing import Any

# One typing call per chat per this many seconds, however many sessions refresh it (Telegram shows
# the indicator for ~5 s; ``_keep_typing`` ticks every 2 s per session).
TYPING_CHAT_MIN_INTERVAL_SECS = 2.0


def _typing_in_cooldown(adapter: Any, chat_id: Any) -> bool:
    """A transient typing failure (``_record_typing_cooldown``) pauses this chat's typing calls."""
    until_by_chat: dict[str, float] = adapter.__dict__.setdefault("_telegram_typing_cooldown_until", {})
    until = until_by_chat.get(str(chat_id))
    if until is None:
        return False
    if asyncio.get_running_loop().time() < until:
        return True
    until_by_chat.pop(str(chat_id), None)
    return False


def typing_may_fire(adapter: Any, chat_id: Any) -> bool:
    """True when a typing call for ``chat_id`` may go out now; reserves the chat's typing tick."""
    if _typing_in_cooldown(adapter, chat_id) or adapter._send_flood_cooldown_remaining(chat_id) is not None:
        return False
    if adapter._chat_outbound_slot_remaining(chat_id) > 0:
        return False  # a send/edit holds the chat's slot: typing never delays the reply itself
    from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

    last: dict[str, float] = adapter.__dict__.setdefault("_telegram_typing_last_sent", {})
    key = str(normalize_telegram_chat_id(chat_id))
    now = asyncio.get_running_loop().time()
    if now - last.get(key, float("-inf")) < TYPING_CHAT_MIN_INTERVAL_SECS:
        return False
    for stale in [k for k, t in last.items() if now - t >= TYPING_CHAT_MIN_INTERVAL_SECS]:
        del last[stale]  # bounded dict
    last[key] = now
    adapter._hold_chat_outbound_slot(chat_id)  # typing spends the same per-chat budget as sends/edits
    return True
