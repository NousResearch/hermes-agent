"""One streamed reply never opens more messages than the delivering adapter's split cap (#152012).

The adapter caps a single split delivery (``MAX_SPLIT_MESSAGES``), but the stream consumer opens
messages on its own: it seals the live preview every ~4,000 chars and sends the heads of a reply
that overflows before its first preview. Those calls each fit the cap, so on their own a 415k
streamed reply posted 100+ messages. These drive the REAL consumer and Telegram adapter against a
fake Bot and count the messages the chat receives.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.waiting import async_wait_until


@pytest.fixture(autouse=True)
def _parse_mode(monkeypatch):
    monkeypatch.setattr(telegram_adapter, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))


class _Bot:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def send_message(self, text="", **_kw):
        self.calls.append(("send", text))
        return SimpleNamespace(message_id=1000 + len(self.calls))

    async def edit_message_text(self, text="", **_kw):
        self.calls.append(("edit", text))
        return True

    async def send_chat_action(self, **_kw):
        return True


def _telegram(*, warnings: bool = True) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._telegram_chat_outbound_slot_secs = 0.0
    adapter._bot = _Bot()
    if not warnings:
        adapter.warning_notifications_enabled = lambda *_a, **_k: False
    return adapter


def _body(chars: int) -> str:
    return "\n".join(" ".join(f"w{i}-{j}" for j in range(10)) for i in range(chars // 70))


async def _stream(adapter, text: str, *, step: int, cancel: bool = False) -> GatewayStreamConsumer:
    consumer = GatewayStreamConsumer(adapter, "4242", config=StreamConsumerConfig(
        cursor=" ▉", edit_interval=0.01, buffer_threshold=10**9))
    task = asyncio.create_task(consumer.run())
    for pos in range(0, len(text), step):
        consumer.on_delta(text[pos:pos + step])
        await asyncio.sleep(0.003)  # health: allow HX013 -- the pace of the stream IS the behaviour (seals between deltas)
    if cancel:  # the gateway's 5 s flush window ran out while the consumer was still draining
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    else:
        consumer.finish(text)
        await asyncio.wait_for(task, 60)
    return consumer


def _sends(adapter) -> list[str]:
    return [text for kind, text in adapter._bot.calls if kind == "send"]


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [2_000, 10**9], ids=["slow-growing", "overflows-before-first-preview"])
async def test_a_huge_streamed_reply_opens_at_most_the_cap_and_ends_with_one_notice(step):
    adapter = _telegram()
    text = _body(300_000)
    consumer = await _stream(adapter, text, step=step)

    sends = _sends(adapter)
    assert len(sends) <= adapter.MAX_SPLIT_MESSAGES
    assert sum("Response truncated" in s for s in sends) == 1
    assert "Response truncated" in sends[-1]
    # The gateway's normal final send must not repeat the reply.
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches(text) is True
    assert consumer.delivered_final_matches("another answer " + text) is False


@pytest.mark.asyncio
async def test_with_warnings_hidden_the_tail_is_dropped_without_a_notice():
    adapter = _telegram(warnings=False)
    text = _body(80_000)
    consumer = await _stream(adapter, text, step=2_000)

    sends = _sends(adapter)
    assert len(sends) <= adapter.MAX_SPLIT_MESSAGES
    assert not any("truncated" in s for s in sends)
    for part in re.findall(r"\\\((\d+)/(\d+)\\\)", "".join(sends)):  # an (N/M) marker never reads N > M
        assert int(part[0]) <= int(part[1])
    assert consumer.delivered_final_matches(text) is True


@pytest.mark.asyncio
async def test_a_stream_cancelled_mid_drain_stays_inside_the_cap():
    adapter = _telegram()
    text = _body(60_000)
    consumer = await _stream(adapter, text, step=2_000, cancel=True)

    sends = _sends(adapter)
    assert len(sends) <= adapter.MAX_SPLIT_MESSAGES
    assert "Response truncated" in sends[-1]
    assert consumer.delivered_final_matches(text) is True


@pytest.mark.asyncio
async def test_a_reply_under_the_cap_is_delivered_whole_without_a_notice():
    adapter = _telegram()
    text = _body(10_000)
    consumer = await _stream(adapter, text, step=500)

    shown = "".join(t for _k, t in adapter._bot.calls).replace("\\", "")  # MarkdownV2 escapes
    assert not any("truncated" in t for _k, t in adapter._bot.calls)
    assert all(word in shown for word in text.split()[-20:])
    assert consumer.delivered_final_matches(text) is True


class _UncappedAdapter(BasePlatformAdapter):
    """A platform with no split ceiling: the consumer keeps opening messages as before."""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self):
        from gateway.config import Platform
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent: list[str] = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        return SendResult(success=True, message_id=message_id)


@pytest.mark.asyncio
async def test_an_adapter_without_a_cap_keeps_every_message():
    adapter = _UncappedAdapter()
    text = _body(60_000)
    await _stream(adapter, text, step=2_000)

    assert len(adapter.sent) > TelegramAdapter.MAX_SPLIT_MESSAGES
    assert not any("truncated" in s for s in adapter.sent)


@pytest.mark.asyncio
async def test_a_cancelled_overflow_edit_refused_mid_split_does_not_repost_the_landed_head():
    """The gateway cancels a consumer still delivering a huge final; its overflow edit lands the
    first parts, then a continuation is flood-refused. The landed head is never sent again, and the
    reply still ends inside the cap with the notice (before: the gateway posted the whole capped
    reply again beside the head, 3 + 8 messages)."""
    adapter = _telegram()
    refused = {"left": 1}
    real_send = adapter._bot.send_message

    async def send_message(text="", **kw):
        if len(adapter._bot.calls) >= 3 and refused["left"]:
            refused["left"] -= 1
            raise type("RetryAfter", (Exception,), {"retry_after": 1})("Flood control exceeded. Retry in 1 seconds")
        return await real_send(text=text, **kw)

    adapter._bot.send_message = send_message
    text = _body(60_000)
    consumer = GatewayStreamConsumer(adapter, "4242", config=StreamConsumerConfig(
        cursor=" ▉", edit_interval=30.0, buffer_threshold=10**9))
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(text[:300])
    await async_wait_until(lambda: consumer.message_id, what="the first preview")
    consumer._accumulated = consumer._stream_ledger = text[:40_000]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    sends = _sends(adapter)
    assert len(sends) <= adapter.MAX_SPLIT_MESSAGES
    assert len(set(sends)) == len(sends)  # no message posted twice
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches(text) is True
