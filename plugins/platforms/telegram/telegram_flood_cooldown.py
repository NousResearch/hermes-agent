"""Telegram per-chat send flood window, persisted so a gateway restart does not forget it (#89962).

A 429 ``retry_after`` arms the chat's window (``_record_send_flood_cooldown``). Inside it,
``send()``, edits, typing, drafts and deletes fail closed locally rather than firing more requests
into a penalty that Telegram lengthens while it keeps getting hit. The window used to live only
in memory, so a restart during a ban forgot it, and the new process's first stall notice, heartbeat
or reply went straight into the penalty.

Each window is now also written to ``telegram_flood_cooldown_<bot id>.json`` in the adapter's own
profile home (``_update_receipt_dir``, resolved when the adapter is constructed inside that
profile's scope, because one gateway serves many profiles and bots). The first window check after
boot loads it once. This is durable runtime state, not an ephemeral lease: it outlives the process
on purpose, and every entry expires on its own (wall-clock deadline, capped like the in-memory
window). A copy restored onto another machine or after a clock jump is therefore ignored or
clamped, never a long silence.
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

from gateway.platforms.base import SendResult
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger("plugins.platforms.telegram.adapter")

# Longest window armed by one refusal: longer penalties are handed to the delivery ledger's own timer.
_MAX_WINDOW_SECONDS = 300.0


def _bot_id(adapter: Any) -> Optional[str]:
    """The numeric bot id prefix of the token (public, no network), or None when there is none."""
    head = str(getattr(getattr(adapter, "config", None), "token", None) or "").split(":", 1)[0]
    return head if head.isdigit() else None


def _store_path(adapter: Any):
    home, bot_id = getattr(adapter, "_update_receipt_dir", None), _bot_id(adapter)
    return None if home is None or bot_id is None else home / f"telegram_flood_cooldown_{bot_id}.json"


def _windows(adapter: Any) -> dict[str, float]:
    """The chat -> loop-clock deadline map, seeded once from this bot's persisted windows."""
    until: dict[str, float] = adapter.__dict__.setdefault("_telegram_send_cooldown_until", {})
    if adapter.__dict__.setdefault("_telegram_send_cooldown_loaded", False):
        return until
    adapter._telegram_send_cooldown_loaded = True
    path = _store_path(adapter)
    if path is None:
        return until
    try:
        saved = json.loads(path.read_text(encoding="utf-8-sig")).get("until") or {}
    except FileNotFoundError:
        return until
    except (OSError, ValueError, AttributeError):
        logger.warning("[Telegram] Ignoring unreadable flood windows at %s", path, exc_info=True)
        return until
    now, loop_now = time.time(), asyncio.get_running_loop().time()
    for chat, deadline in saved.items() if isinstance(saved, dict) else ():
        if not isinstance(deadline, (int, float)) or deadline <= now:
            continue
        # Clamp: a wall-clock jump back must not turn a few minutes into hours of silence.
        resumed = loop_now + min(float(deadline) - now, _MAX_WINDOW_SECONDS)
        until[str(chat)] = max(until.get(str(chat), 0.0), resumed)
        logger.warning("[Telegram] Flood window for chat %s carried over from the previous run (%.0fs left)",
                       chat, resumed - loop_now)
    return until


def _schedule_persist(adapter: Any) -> None:
    """One coalesced writer per adapter; each pass snapshots the live map on the loop before the
    threaded write, so an older snapshot can never land after a newer one."""
    path = _store_path(adapter)
    if path is None:
        return
    adapter._telegram_send_cooldown_dirty = True
    task = adapter.__dict__.get("_telegram_send_cooldown_flush")
    if task is not None and not task.done():
        return

    async def _flush() -> None:
        from hermes_constants import mkdir_under_hermes_home
        from utils import atomic_json_write
        while adapter._telegram_send_cooldown_dirty:
            adapter._telegram_send_cooldown_dirty = False
            loop_now, now = asyncio.get_running_loop().time(), time.time()
            snapshot = {chat: now + (deadline - loop_now)
                        for chat, deadline in adapter._telegram_send_cooldown_until.items() if deadline > loop_now}
            try:
                mkdir_under_hermes_home(path.parent)
                await asyncio.to_thread(atomic_json_write, path, {"until": snapshot}, indent=None)
            except Exception:  # health: allow BLE001 -- best-effort: a failed write keeps the in-memory window
                logger.warning("[Telegram] Failed to persist flood windows to %s", path, exc_info=True)

    task = adapter._telegram_send_cooldown_flush = asyncio.get_running_loop().create_task(
        _flush(), name="telegram-flood-window-persist")
    tracked = getattr(adapter, "_background_tasks", None)
    if isinstance(tracked, set):
        tracked.add(task)
        task.add_done_callback(tracked.discard)


class TelegramFloodCooldownMixin:
    """The per-chat send flood window of ``TelegramAdapter``."""

    def _record_send_flood_cooldown(self, chat_id: Any, wait: float) -> SendResult:
        """A send refused with ``retry_after=wait`` arms a per-chat window during which ``send()`` fails
        closed locally (same ``flood_control:<s>`` result, so ledger recognition and redelivery timing are
        unchanged) instead of firing more requests into a penalty Telegram lengthens while it is hammered."""
        from plugins.platforms.telegram.adapter import _flood_cap_result
        until = _windows(self)
        until[str(normalize_telegram_chat_id(chat_id))] = (
            asyncio.get_running_loop().time() + max(1.0, min(float(wait), _MAX_WINDOW_SECONDS)))
        _schedule_persist(self)
        return _flood_cap_result(wait)

    def _send_flood_cooldown_remaining(self, chat_id: Any) -> Optional[float]:
        """Seconds left in this chat's flood window, or ``None`` when sends may go out."""
        until = _windows(self)
        key = str(normalize_telegram_chat_id(chat_id))
        deadline = until.get(key)
        if deadline is None:
            return None
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining > 0:
            return remaining
        until.pop(key, None)
        return None
