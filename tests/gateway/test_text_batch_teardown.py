"""Adapter teardown must not lose a text batch that is still in its quiet period.

The quiet period is operator-configured (up to the 600 s safety cap), so it can outlast the
gateway's teardown budget. ``cancel_background_tasks`` (called on shutdown and adapter replacement,
before ``disconnect``) dispatches every buffered batch at once and gives the turn it starts a bounded
window, so the gateway can still answer (during a shutdown drain: the "not accepting new work" reply).
A bare ``disconnect`` (the fatal-error path) cannot answer any more: it stops the timers so none fires
into the disconnected adapter, and warns about what it dropped.
"""

import asyncio
import logging
from typing import Any, Dict

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.weixin import WeixinAdapter
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _event(platform: Platform, text: str) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT,
                        source=SessionSource(platform=platform, chat_id="60123", chat_type="dm", user_id="60123"))


@pytest.mark.asyncio
@pytest.mark.parametrize("cls, platform", [(WhatsAppAdapter, Platform.WHATSAPP), (WeixinAdapter, Platform.WEIXIN)],
                         ids=["whatsapp", "weixin"])
async def test_teardown_dispatches_a_batch_still_in_its_quiet_period(cls, platform, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = cls(PlatformConfig(enabled=True, extra={"text_batch_delay_seconds": 600}))
    delivered = []

    async def record(event: MessageEvent) -> None:
        delivered.append(event.text)

    adapter.handle_message = record
    adapter._enqueue_text_event(_event(platform, "one"))
    adapter._enqueue_text_event(_event(platform, "two"))
    (timer,) = adapter._pending_text_batch_tasks.values()

    await asyncio.wait_for(adapter.cancel_background_tasks(), timeout=10)

    assert delivered == ["one\ntwo"]
    assert timer.done()
    assert adapter._pending_text_batches == {} and adapter._pending_text_batch_tasks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("cls, platform", [(WhatsAppAdapter, Platform.WHATSAPP), (WeixinAdapter, Platform.WEIXIN)],
                         ids=["whatsapp", "weixin"])
async def test_bare_disconnect_stops_the_timer_and_warns(cls, platform, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = cls(PlatformConfig(enabled=True, extra={"text_batch_delay_seconds": 600}))
    delivered = []

    async def record(event: MessageEvent) -> None:
        delivered.append(event.text)

    adapter.handle_message = record
    adapter._enqueue_text_event(_event(platform, "one"))
    (timer,) = adapter._pending_text_batch_tasks.values()

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(adapter.disconnect(), timeout=10)
    await asyncio.sleep(0)

    assert timer.done() and delivered == []
    assert adapter._pending_text_batches == {} and adapter._pending_text_batch_tasks == {}
    assert any("undelivered text batch" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["in-quiet-period", "mid-dispatch"])
async def test_teardown_keeps_its_deadline_and_fences_a_stuck_dispatch(stage, tmp_path, monkeypatch):
    """A handler that never returns cannot hold teardown past its deadline, nor deliver after disconnect."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"text_batch_delay_seconds": 600}))
    adapter._text_batch_flush_deadline_seconds = lambda: 0.2
    entered, release, delivered = asyncio.Event(), asyncio.Event(), []

    async def stuck(event: MessageEvent) -> None:
        entered.set()
        await release.wait()
        delivered.append(event.text)

    adapter.handle_message = stuck
    if stage == "mid-dispatch":
        adapter._text_batch_delay_for = lambda pending: 0
    adapter._enqueue_text_event(_event(Platform.WHATSAPP, "hello"))
    if stage == "mid-dispatch":
        await asyncio.wait_for(entered.wait(), timeout=5)  # the timer already popped the batch

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(adapter.cancel_background_tasks(), timeout=5)
    assert loop.time() - started < 2  # 0.2 s deadline; nothing else in flight
    await adapter.disconnect()
    release.set()
    await asyncio.sleep(0.05)

    assert entered.is_set()  # the batch was handed over ...
    assert delivered == []  # ... but the stuck hand-off never completes after disconnect


class _ReplyingAdapter(BasePlatformAdapter):
    """Real ``handle_message`` → background turn → ``send``; the gateway handler answers after a hop."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.WHATSAPP)
        self._text_batch_delay_seconds = 600.0
        self._text_batch_split_delay_seconds = 600.0
        self.sent: list = []

        async def gateway(event: MessageEvent) -> str:
            await asyncio.sleep(0.05)  # profile/session lookups before the drain reply
            return f"not accepting new work: {event.text}"

        self.set_message_handler(gateway)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, chat_id: str, content: str, *a: Any, **k: Any) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_the_gateway_answer_to_a_flushed_batch_is_sent_before_teardown_cancels():
    adapter = _ReplyingAdapter()
    adapter._enqueue_text_event(_event(Platform.WHATSAPP, "hello"))

    await asyncio.wait_for(adapter.cancel_background_tasks(), timeout=10)

    assert adapter.sent == ["not accepting new work: hello"]
