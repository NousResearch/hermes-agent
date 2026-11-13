"""A tool-progress bubble that outgrew the cap while a flood window was open is owed once (#151789).

Inside the window the overflow roll cannot run, so a held bubble near the cap keeps absorbing lines.
When a content bubble closes it, the owed edit used to carry the whole buffer: Telegram split it,
re-edited chunk 1 and re-posted every continuation that had already landed on each idle tick, and a
flood-refused continuation came back with no wait, so no window was armed and the loop went on.
Drives the real ``TelegramAdapter`` and the real progress mixin; only the PTB ``Bot`` is faked.
"""

import queue
import re
import time
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class RetryAfter(Exception):
    """PTB's ``telegram.error.RetryAfter`` shape: the penalty rides on ``retry_after``."""

    def __init__(self, seconds: int):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _bot(refuse):
    """A PTB bot fake: ``refuse(text, attempt)`` decides which ``send_message`` calls fail: truthy raises
    a flood refusal, an exception instance is raised as is."""
    bot = SimpleNamespace(sends=[], landed=[], edits=[])

    async def send_message(**kwargs):
        bot.sends.append(kwargs["text"])
        refused = refuse(kwargs["text"], len(bot.sends))
        if isinstance(refused, Exception):
            raise refused
        if refused:
            raise RetryAfter(20)
        bot.landed.append(kwargs["text"])
        return SimpleNamespace(message_id=100 + len(bot.landed))

    async def edit_message_text(**kwargs):
        bot.edits.append(kwargs["text"])
        return True

    bot.send_message = AsyncMock(side_effect=send_message)
    bot.edit_message_text = AsyncMock(side_effect=edit_message_text)
    return bot


def _held_over_cap_bubble(bot, content_resets=1):
    from hermes_agent.gateway.run_turn_progress import GatewayTurnProgressMixin

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = bot
    loop = GatewayTurnProgressMixin()
    loop._ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="-1001"), _progress_metadata={}, _progress_reply_to=None,
        _cleanup_progress=False, _run_still_current=lambda: True, progress_grouping="edit",
        agent_holder=[SimpleNamespace(is_interrupted=False)], progress_queue=queue.Queue(),
        last_progress_msg=[None], repeat_count=[0])
    st = loop._progress_edit_state(adapter)
    st.progress_msg_id, st.flood_pending, st.flood_until = "7", True, time.monotonic() + 60
    for i in range(160):  # ~9.4k chars: well past one Telegram message
        loop._progress_absorb(st, f'💻 terminal: "build step {i:03d} running the long command here"')
    for _ in range(content_resets):  # content bubbles land while the window is still open
        loop._reset_progress_bubble(st)
    st.flood_until = 0.0  # the window closes while the agent keeps working
    return loop, st


async def _idle_ticks(loop, st, n=6):
    for _ in range(n):
        await loop._deliver_flood_pending_progress(st)


@pytest.mark.asyncio
async def test_owed_over_cap_bubble_never_reposts_a_landed_message():
    bot = _bot(lambda text, attempt: "(3/" in text)  # one continuation stays refused
    loop, st = _held_over_cap_bubble(bot)

    await _idle_ticks(loop, st)

    reposted = {text: n for text, n in Counter(bot.landed).items() if n > 1}
    assert not reposted, f"{len(bot.landed)} landed, re-posted: {list(reposted)[:1]}"
    assert len(bot.edits) == 1  # the closed bubble gets its owed edit once


@pytest.mark.asyncio
async def test_a_flood_refused_progress_message_arms_the_window_instead_of_looping():
    bot = _bot(lambda text, attempt: attempt == 1)  # the first message after the window is refused
    loop, st = _held_over_cap_bubble(bot)

    await _idle_ticks(loop, st)

    assert st.flood_until > time.monotonic() + 10  # the 20 s penalty is honoured
    assert len(bot.sends) == 1, bot.sends  # nothing else fired into it, no plain-text retry either


def _shown_steps(bot):
    return set(re.findall(r"build step (\d{3})", "\n".join(bot.edits + bot.landed)))


@pytest.mark.asyncio
async def test_a_second_content_bubble_inside_the_window_keeps_the_carried_lines():
    bot = _bot(lambda text, attempt: False)
    loop, st = _held_over_cap_bubble(bot, content_resets=2)  # text, tool, more text: one penalty

    await _idle_ticks(loop, st)

    assert _shown_steps(bot) == {f"{i:03d}" for i in range(160)}
    assert len(bot.landed) == len(set(bot.landed))


@pytest.mark.parametrize("error", [Exception("Bad Request: chat write forbidden"), ConnectionError("Connection reset")])
@pytest.mark.asyncio
async def test_a_non_flood_failure_of_the_carried_lines_is_retried(error):
    bot = _bot(lambda text, attempt: error if attempt == 1 else False)
    loop, st = _held_over_cap_bubble(bot)

    await _idle_ticks(loop, st)

    assert _shown_steps(bot) == {f"{i:03d}" for i in range(160)}
    assert len(bot.landed) == len(set(bot.landed))
