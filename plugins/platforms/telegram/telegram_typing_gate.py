"""Fair per-chat Telegram typing admission, without reserving the send/edit slot.

Typing retains the shared draft budget and hard cooldowns. Progress edits receive bounded
courtesy, not an indefinite veto; admission is not a successful refresh or client-visibility bound.
"""
from __future__ import annotations

import asyncio
from typing import Any

from gateway.platforms.helpers import typing_pause_key
from plugins.platforms.telegram.telegram_draft_budget import claim_ai_action
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

TYPING_CHAT_MIN_INTERVAL_SECS = 2.0
TYPING_BUSY_CHAT_MIN_INTERVAL_SECS = 1.0
TYPING_TOPIC_IDLE_SECS = 2.5
TYPING_SOFT_SLOT_YIELD_SECS = 3.0


def record_typing_cooldown(adapter: Any, chat_id: str, exc: Exception) -> None:
    """Suppress refreshes after transient failures; kept behind the adapter's existing method."""
    if not hasattr(adapter, "_telegram_typing_cooldown_until"):
        adapter._telegram_typing_cooldown_until = {}
    retry_after = getattr(exc, "retry_after", None)
    try:
        delay = float(retry_after) if retry_after is not None else adapter._telegram_typing_cooldown_seconds
    except (TypeError, ValueError):
        delay = adapter._telegram_typing_cooldown_seconds
    adapter._telegram_typing_cooldown_until[str(chat_id)] = (
        asyncio.get_running_loop().time() + max(1.0, min(delay, 300.0)))


def _typing_in_cooldown(adapter: Any, chat_id: Any) -> bool:
    """A transient typing failure pauses this chat's typing calls."""
    until_by_chat: dict[str, float] = adapter.__dict__.setdefault("_telegram_typing_cooldown_until", {})
    until = until_by_chat.get(str(chat_id))
    if until is None:
        return False
    if asyncio.get_running_loop().time() < until:
        return True
    until_by_chat.pop(str(chat_id), None)
    return False


def _paused(adapter: Any, chat_id: Any, metadata=None) -> bool:
    paused = getattr(adapter, '_typing_paused', ())
    return chat_id in paused or typing_pause_key(chat_id, metadata) in paused


def _waiting_turn(adapter: Any, chat_id: Any, metadata, now: float) -> tuple[dict, str | None]:
    """Keep retry position; retire paused and absent topics instead of pinning the queue head."""
    key = str(normalize_telegram_chat_id(chat_id))
    paused = getattr(adapter, '_typing_paused', ())
    queues: dict = adapter.__dict__.setdefault('_telegram_typing_waiters', {})
    for chat, waiting in list(queues.items()):
        for topic, (seen, _served, _first_seen, pause_scope) in list(waiting.items()):
            if now - seen > TYPING_TOPIC_IDLE_SECS or pause_scope[0] in paused or pause_scope in paused:
                waiting.pop(topic)
        if not waiting:
            queues.pop(chat)
    topic = adapter._metadata_thread_id(metadata)
    waiting = queues.setdefault(key, {})
    previous = waiting.get(topic, (now, float('-inf'), now))
    waiting[topic] = (now, previous[1], previous[2], typing_pause_key(chat_id, metadata))
    return waiting, topic


def forget_typing_waiter(adapter: Any, chat_id: Any, metadata=None) -> None:
    """Finish only this topic, without refunding the chat's already-spent typing allowance."""
    key = str(normalize_telegram_chat_id(chat_id))
    queues = adapter.__dict__.get('_telegram_typing_waiters', {})
    waiting = queues.get(key)
    if waiting is None:
        return
    waiting.pop(adapter._metadata_thread_id(metadata), None)
    if not waiting:
        queues.pop(key, None)


def typing_may_fire(adapter: Any, chat_id: Any, metadata=None) -> bool:
    """Reserve one chat action fairly; failures and cancellation still spend this admission."""
    if _paused(adapter, chat_id, metadata):
        forget_typing_waiter(adapter, chat_id, metadata)
        return False
    key = str(normalize_telegram_chat_id(chat_id))
    now = asyncio.get_running_loop().time()
    waiting, topic = _waiting_turn(adapter, chat_id, metadata, now)
    if _typing_in_cooldown(adapter, chat_id) or adapter._send_flood_cooldown_remaining(chat_id) is not None:
        return False
    last: dict[str, float] = adapter.__dict__.setdefault('_telegram_typing_last_sent', {})
    # Three topics at 2s/chat already need 6s per round, beyond the typing lifetime.
    interval = TYPING_BUSY_CHAT_MIN_INTERVAL_SECS if len(waiting) >= 3 else TYPING_CHAT_MIN_INTERVAL_SECS
    # Infinite newcomer priority lets short-lived topics take every tick from old work.
    first = min(waiting, key=lambda candidate: max(
        waiting[candidate][1], waiting[candidate][2] - TYPING_SOFT_SLOT_YIELD_SECS))
    if now - last.get(key, float('-inf')) < interval or first != topic:
        return False
    if (adapter._chat_outbound_slot_remaining(chat_id) > 0
            and now - waiting[topic][1] < TYPING_SOFT_SLOT_YIELD_SECS):
        return False
    if claim_ai_action(adapter, chat_id):
        return False  # hard peer budget remains shared with draft frames
    waiting[topic] = (now, now, *waiting[topic][2:])
    for stale in [k for k, t in last.items() if now - t >= TYPING_CHAT_MIN_INTERVAL_SECS]:
        del last[stale]
    last[key] = now
    return True
