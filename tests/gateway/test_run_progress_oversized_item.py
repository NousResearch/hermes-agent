"""One tool-progress item larger than a Telegram message is shown whole, once (#151789).

A /verbose terminal call queues its full multiline command as ONE progress item. When a content
reset closed a flood-held bubble holding such an item, the owed edit was split by the adapter and a
partial delivery settled it with its tail unseen; a carried remainder holding one went out as one
send the adapter split, and a failed continuation re-sent the whole group, duplicating its head.
Drives the real ``TelegramAdapter`` and the real progress mixin; only the PTB ``Bot`` is faked.
"""

import queue
import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, RetryAfter

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

MARKERS = {f"{i:04d}" for i in range(180)}


def _bot(refuse):
    """``refuse(attempt)`` returns the PTB exception the n-th ``send_message`` raises, or None.
    ``shown`` maps message id to the text currently on screen (edits overwrite in place)."""
    bot = SimpleNamespace(sends=0, shown={}, next_id=100)

    async def send_message(**kwargs):
        bot.sends += 1
        if error := refuse(bot.sends):
            raise error
        bot.next_id += 1
        bot.shown[bot.next_id] = kwargs["text"]
        return SimpleNamespace(message_id=bot.next_id)

    async def edit_message_text(**kwargs):
        bot.shown[int(kwargs["message_id"])] = kwargs["text"]
        return True

    bot.send_message = AsyncMock(side_effect=send_message)
    bot.edit_message_text = AsyncMock(side_effect=edit_message_text)
    return bot


def _held_bubble(bot, items):
    """A progress bubble (message 7) holding ``items``, held by a flood window a content reset closes."""
    from hermes_agent.gateway.run_turn_progress import GatewayTurnProgressMixin

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", typing_indicator=False))
    adapter._bot = bot
    loop = GatewayTurnProgressMixin()
    loop._ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="-1001"), _progress_metadata={}, _progress_reply_to=None,
        _cleanup_progress=False, _run_still_current=lambda: True, progress_grouping="edit",
        agent_holder=[SimpleNamespace(is_interrupted=False)], progress_queue=queue.Queue(),
        last_progress_msg=[None], repeat_count=[0], last_was_terminal_block=[False])
    st = loop._progress_edit_state(adapter)
    st.progress_msg_id, st.flood_pending, st.flood_until = "7", True, time.monotonic() + 60
    for item in items(loop, adapter):
        loop._progress_absorb(st, item)
    loop._reset_progress_bubble(st)  # a content bubble lands while the window is open
    return loop, st


def _penalty_runs_out(st):
    st.flood_until = 0.0
    st.adapter.__dict__.get("_telegram_send_cooldown_until", {}).clear()


def _verbose_command(loop, adapter):
    command = "\n".join(f"echo MARK{i:04d} " + "x" * 35 for i in range(180))
    full, _ = loop._progress_terminal_blocks(adapter, "terminal", {"command": command}, "💻")
    return [full]  # the real producer: one queue item for the whole command


def _markers(bot):
    return re.findall(r"MARK(\d{4})", "\n".join(bot.shown.values()))


@pytest.mark.asyncio
async def test_an_oversized_item_refused_mid_way_still_reaches_the_screen():
    bot = _bot(lambda attempt: RetryAfter(20) if attempt == 1 else None)
    loop, st = _held_bubble(bot, _verbose_command)

    for _ in range(6):
        _penalty_runs_out(st)
        await loop._deliver_flood_pending_progress(st)

    assert set(_markers(bot)) == MARKERS


@pytest.mark.asyncio
async def test_a_carried_oversized_item_is_never_shown_twice():
    bot = _bot(lambda attempt: BadRequest("Chat write forbidden") if attempt == 2 else None)
    loop, st = _held_bubble(bot, lambda loop, adapter: [
        "short head", "\n".join(f"MARK{i:04d} " + "x" * 45 for i in range(180))])

    for _ in range(8):
        _penalty_runs_out(st)
        await loop._deliver_flood_pending_progress(st)

    shown = _markers(bot)
    assert len(shown) == len(set(shown)) and set(shown) == MARKERS


@pytest.mark.asyncio
async def test_a_flood_refusal_after_a_dropped_reply_anchor_still_arms_the_wait():
    bot = _bot(lambda attempt: BadRequest("Bad Request: reply message not found") if attempt == 1 else RetryAfter(20))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", typing_indicator=False))
    adapter._bot = bot

    result = await adapter._edit_overflow_split("-1001", "7", "word " * 2200, finalize=True)

    assert result.retry_after == 20 and bot.sends == 2  # no third call into the penalty
