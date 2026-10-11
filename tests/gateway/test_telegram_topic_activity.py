"""Actual adapter lifecycle: topic presence must not depend on answer streaming."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

TOPIC = {"thread_id": "99", "telegram_dm_topic_reply_fallback": True,
         "telegram_reply_to_message_id": "42"}
OTHER = {**TOPIC, "thread_id": "987"}


def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake", extra={"rich_messages": False}))
    a._bot = SimpleNamespace(send_message_draft=AsyncMock(return_value=True),
                             send_chat_action=AsyncMock(return_value=True),
                             send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)))
    return a


@pytest.fixture
def clock(monkeypatch):
    from plugins.platforms.telegram import topic_activity
    c = SimpleNamespace(now=100.0)
    monkeypatch.setattr(topic_activity, "time", SimpleNamespace(monotonic=lambda: c.now))
    return c


@pytest.mark.asyncio
async def test_work_presence_without_stream_consumer_and_with_sibling_topic(clock):
    a = adapter()
    await a.send_typing("123", TOPIC)
    first = a._bot.send_message_draft.call_args.kwargs
    assert first["text"] == ""
    assert first["message_thread_id"] == 99
    assert first["draft_id"] != 0
    a._bot.send_message.assert_not_awaited()
    await a.send_typing("123", TOPIC)
    assert a._bot.send_message_draft.await_count == 1
    clock.now += 11
    await a.send_typing("123", TOPIC)
    assert a._bot.send_message_draft.call_args.kwargs == first
    a.pause_typing_for_chat("123", TOPIC)
    clock.now += 11
    await a.send_typing("123", TOPIC)
    assert a._bot.send_message_draft.await_count == 2
    await a.send_typing("123", OTHER)
    assert a._bot.send_message_draft.call_args.kwargs["message_thread_id"] == 987
    a.resume_typing_for_chat("123", TOPIC)
    await a.send_typing("123", TOPIC)
    assert a._bot.send_message_draft.await_count == 4


@pytest.mark.asyncio
async def test_activity_preserves_streamed_text_and_clears_on_final_send(clock):
    a = adapter()
    await a.send_typing("123", TOPIC)
    first = a._bot.send_message_draft.call_args.kwargs
    result = await a.send_draft("123", 77, "An answer", TOPIC)
    assert result.success
    streamed = a._bot.send_message_draft.call_args.kwargs
    assert streamed["draft_id"] == first["draft_id"]
    assert streamed["text"] == "An answer"
    clock.now += 11
    await a.send_typing("123", TOPIC)
    assert a._bot.send_message_draft.call_args.kwargs == streamed
    result = await a.send("123", "An answer", metadata={**TOPIC, "notify": True})
    assert result.success
    count = a._bot.send_message_draft.await_count
    assert count == 3  # Final send must not rearm activity.
    assert not a._topic_activity_state().previews
    clock.now += 11
    await a.send_typing("123", TOPIC)  # Next real work, no stale answer.
    assert a._bot.send_message_draft.call_args.kwargs["text"] == ""


@pytest.mark.asyncio
async def test_interim_send_rearms_and_stop_forgets_only_its_topic(clock):
    a = adapter()
    await a.send_typing("123", TOPIC)
    await a.send_draft("123", 77, "Progress", TOPIC)
    await a.send("123", "Progress", metadata=TOPIC)
    await asyncio.gather(*a._telegram_typing_retrigger_tasks.values())
    assert a._bot.send_message_draft.call_args.kwargs["text"] == ""
    await a.send_typing("123", OTHER)
    await a.stop_typing("123", metadata=TOPIC)
    assert ("123", "99") not in a._topic_activity_state().previews
    assert ("123", "987") in a._topic_activity_state().previews


@pytest.mark.asyncio
@pytest.mark.parametrize("chat,meta", [("-123", TOPIC), ("123", {}),
    ("123", {"thread_id": "7"}), ("123", {**TOPIC, "thread_id": "1"}),
    ("123", {**TOPIC, "thread_id": "oops"})])
async def test_never_sends_preview_outside_known_private_topic(chat, meta):
    a = adapter()
    await a.send_typing(chat, meta)
    a._bot.send_message_draft.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejection_is_nonfatal_and_backed_off(clock):
    a = adapter()
    a._bot.send_message_draft.return_value = False
    await a.send_typing("123", TOPIC)
    clock.now += 11
    await a.send_typing("123", TOPIC)
    assert a._bot.send_message_draft.await_count == 1
    assert a._bot.send_chat_action.await_count == 2
    result = await a.send("123", "Final answer", metadata={**TOPIC, "notify": True})
    assert result.success


@pytest.mark.asyncio
async def test_disable_and_older_sdk():
    a = adapter()
    a.config.extra["topic_activity_preview"] = False
    await a.send_typing("123", TOPIC)
    a._bot.send_message_draft.assert_not_awaited()
    a.config.extra["topic_activity_preview"] = True
    del a._bot.send_message_draft
    await a.send_typing("123", TOPIC)
    assert a._bot.send_chat_action.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_stream_cannot_be_erased_by_activity(clock):
    a = adapter()
    entered, release = asyncio.Event(), asyncio.Event()
    async def blocked(**kw):
        entered.set()
        await release.wait()
        return True
    a._bot.send_message_draft.side_effect = blocked
    typing = asyncio.create_task(a.send_typing("123", TOPIC))
    await asyncio.wait_for(entered.wait(), 1)
    stream = asyncio.create_task(a.send_draft("123", 77, "Answer", TOPIC))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(typing, stream)
    clock.now += 11
    await a.send_typing("123", TOPIC)
    assert a._bot.send_message_draft.call_args.kwargs["text"] == "Answer"
