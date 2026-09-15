"""Telegram post-send typing re-arm: off the critical path, deduped and rate-limited (#111727).

Awaiting ``sendChatAction`` after every intermediate send ran its TLS round-trip on the same event
loop as the ``getUpdates`` long-polls; under concurrent streaming the polls were starved until they
rotted into CLOSE-WAIT while the adapter still reported ``connected``.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(**config_kwargs):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", **config_kwargs))
    adapter._bot = AsyncMock()
    adapter._bot.send_chat_action = AsyncMock(return_value=None)
    return adapter


async def _drain(adapter, chat_id="123"):
    """Await the scheduled re-arm so assertions see its effect."""
    task = adapter._telegram_typing_retrigger_tasks.get(str(chat_id))
    if task is not None:
        await task


@pytest.mark.asyncio
async def test_retrigger_does_not_await_the_round_trip():
    """The send path must return before sendChatAction completes — this is the whole fix."""
    adapter = _make_adapter()
    released = asyncio.Event()
    started = asyncio.Event()

    async def blocking_action(**kwargs):
        started.set()
        await released.wait()

    adapter._bot.send_chat_action = AsyncMock(side_effect=blocking_action)

    # Would hang forever if the re-arm were still awaited inline.
    await asyncio.wait_for(adapter._retrigger_typing("123", None), timeout=1.0)

    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert not adapter._telegram_typing_retrigger_tasks["123"].done()
    released.set()
    await _drain(adapter)
    assert adapter._bot.send_chat_action.await_count == 1


@pytest.mark.asyncio
async def test_streaming_burst_collapses_to_one_chat_action():
    """Every chunk of a streamed reply re-arms typing; within the interval that is one API call."""
    adapter = _make_adapter()

    for _ in range(20):
        await adapter._retrigger_typing("123", None)
    await _drain(adapter)

    assert adapter._bot.send_chat_action.await_count == 1


@pytest.mark.asyncio
async def test_retrigger_resumes_after_the_interval_elapses():
    adapter = _make_adapter()
    adapter._telegram_typing_retrigger_interval = 2.0

    await adapter._retrigger_typing("123", None)
    await _drain(adapter)

    # Simulate the interval having passed rather than sleeping through it.
    adapter._telegram_typing_retrigger_at["123"] -= 2.5
    await adapter._retrigger_typing("123", None)
    await _drain(adapter)

    assert adapter._bot.send_chat_action.await_count == 2


@pytest.mark.asyncio
async def test_throttle_is_per_chat():
    adapter = _make_adapter()

    await adapter._retrigger_typing("123", None)
    await adapter._retrigger_typing("456", None)
    await _drain(adapter, "123")
    await _drain(adapter, "456")

    assert adapter._bot.send_chat_action.await_count == 2


@pytest.mark.asyncio
async def test_in_flight_rearm_is_not_duplicated():
    """A slow round-trip must not accumulate one task per chunk."""
    adapter = _make_adapter()
    adapter._telegram_typing_retrigger_interval = 0.0  # throttle off: the in-flight guard alone
    released = asyncio.Event()

    async def blocking_action(**kwargs):
        await released.wait()

    adapter._bot.send_chat_action = AsyncMock(side_effect=blocking_action)

    for _ in range(10):
        await adapter._retrigger_typing("123", None)

    assert len(adapter._telegram_typing_retrigger_tasks) == 1
    released.set()
    await _drain(adapter)
    assert adapter._bot.send_chat_action.await_count == 1


@pytest.mark.asyncio
async def test_typing_indicator_disabled_suppresses_rearm():
    """`typing_indicator: false` only gated _keep_typing, so the re-arm still cost a call per send."""
    adapter = _make_adapter(typing_indicator=False)

    await adapter._retrigger_typing("123", None)
    await _drain(adapter)

    assert adapter._telegram_typing_retrigger_tasks == {}
    adapter._bot.send_chat_action.assert_not_called()


@pytest.mark.asyncio
async def test_final_reply_still_does_not_rearm():
    adapter = _make_adapter()

    await adapter._retrigger_typing("123", {"notify": True})

    assert adapter._telegram_typing_retrigger_tasks == {}
    adapter._bot.send_chat_action.assert_not_called()


@pytest.mark.asyncio
async def test_rearm_is_tracked_for_shutdown_cancellation():
    """Detached tasks must join _background_tasks or they outlive the adapter."""
    adapter = _make_adapter()
    released = asyncio.Event()

    async def blocking_action(**kwargs):
        await released.wait()

    adapter._bot.send_chat_action = AsyncMock(side_effect=blocking_action)

    await adapter._retrigger_typing("123", None)
    task = adapter._telegram_typing_retrigger_tasks["123"]
    assert task in adapter._background_tasks

    await adapter.cancel_background_tasks()
    assert task.cancelled() or task.done()
    assert adapter._telegram_typing_retrigger_tasks == {}


@pytest.mark.asyncio
async def test_rearm_failure_does_not_escape_to_the_send_path():
    adapter = _make_adapter()
    adapter._bot.send_chat_action = AsyncMock(side_effect=OSError("telegram network failure"))

    await adapter._retrigger_typing("123", None)
    await _drain(adapter)

    assert adapter._telegram_typing_retrigger_tasks == {}


@pytest.mark.asyncio
async def test_send_returns_without_waiting_on_typing():
    """End-to-end: an intermediate send completes even while sendChatAction is stalled."""
    adapter = _make_adapter()
    adapter._rich_messages_enabled = False
    adapter._bot.send_message = AsyncMock(return_value=type("Msg", (), {"message_id": 1})())
    released = asyncio.Event()

    async def blocking_action(**kwargs):
        await released.wait()

    adapter._bot.send_chat_action = AsyncMock(side_effect=blocking_action)

    result = await asyncio.wait_for(adapter.send("123", "chunk"), timeout=1.0)

    assert result.success is True
    released.set()
    await _drain(adapter)
