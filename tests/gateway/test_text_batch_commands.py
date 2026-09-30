"""A slash command is never held in, or merged into, a text batch.

With a long quiet period, ``hello`` followed by ``/stop`` used to become one batch ``hello\\n/stop``
(which does not parse as a command) or send ``/stop`` ahead of ``hello``. The intake now flushes the
session's pending text first and dispatches the command at once, as Telegram does with its command
handler. Driven through each adapter's real intake path.
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.weixin import WeixinAdapter
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


async def _whatsapp_intake(adapter: WhatsAppAdapter, texts: list) -> None:
    """One real ``_poll_messages`` pass over ``texts`` (bridge and event parsing stubbed)."""
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="60123@s.whatsapp.net", chat_type="dm", user_id="60123")
    adapter._build_message_event = AsyncMock(side_effect=[
        MessageEvent(text=t, message_type=MessageType.TEXT, source=source) for t in texts])
    adapter._report_bridge_exit = AsyncMock(return_value=False)
    adapter._send_read_receipt = AsyncMock()
    adapter._http_session = object()

    @contextlib.asynccontextmanager
    async def bridge_req(*_a, **_k):
        adapter._running = False  # a single poll
        yield SimpleNamespace(status=200, json=AsyncMock(return_value=[{"id": i} for i in range(len(texts))]))

    adapter._bridge_req = bridge_req
    adapter._running = True
    await adapter._poll_messages()


async def _weixin_intake(adapter: WeixinAdapter, texts: list) -> None:
    adapter._poll_session = object()
    for i, text in enumerate(texts):
        await adapter._process_message({"from_user_id": "wxid_user1", "message_id": f"m{i}",
                                        "item_list": [{"type": 1, "text_item": {"text": text}}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["whatsapp", "weixin"])
async def test_command_flushes_pending_text_then_dispatches_at_once(platform, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    extra = {"text_batch_delay_seconds": 40}
    if platform == "whatsapp":
        adapter, intake = WhatsAppAdapter(PlatformConfig(enabled=True, extra=extra)), _whatsapp_intake
    else:
        adapter = WeixinAdapter(PlatformConfig(enabled=True, token="t", extra={**extra, "account_id": "acct"}))
        intake = _weixin_intake
    delivered = []

    async def record(event: MessageEvent) -> None:
        delivered.append(event.text)

    adapter.handle_message = record

    await asyncio.wait_for(intake(adapter, ["hello", "/stop"]), timeout=10)

    assert delivered == ["hello", "/stop"]  # no 40 s wait, no merge, arrival order kept
    assert adapter._pending_text_batches == {}


@pytest.mark.asyncio
async def test_command_waits_for_a_batch_whose_timer_already_fired(tmp_path, monkeypatch):
    """The timer popped ``hello`` and scheduled its dispatch; ``/stop`` arriving now must still land second."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"text_batch_delay_seconds": 40}))
    seen = []

    async def record(event: MessageEvent) -> None:
        seen.append(event.text)

    adapter.handle_message = record
    adapter._text_batch_delay_for = lambda pending: 0  # let the timer fire at once
    popped = asyncio.Event()
    pop = adapter._pop_text_batch

    def pop_and_signal(key):
        event = pop(key)
        popped.set()  # wakes the test before the dispatch the timer schedules next gets to run
        return event

    adapter._pop_text_batch = pop_and_signal
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="60123@s.whatsapp.net", chat_type="dm", user_id="60123")
    await adapter._batch_text_or_dispatch(MessageEvent(text="hello", message_type=MessageType.TEXT, source=source))
    await popped.wait()  # awaited directly, so the test resumes before the dispatch the timer scheduled
    assert adapter._pending_text_batches == {}  # the command path cannot flush it: the timer holds it

    await adapter._batch_text_or_dispatch(MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source))
    await asyncio.sleep(0)

    assert seen == ["hello", "/stop"]


@pytest.mark.asyncio
async def test_command_is_not_held_forever_by_a_stuck_batch_hand_off(tmp_path, monkeypatch):
    """Order behind an in-flight batch is best-effort: /stop reaches the handler within the bound."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"text_batch_delay_seconds": 40}))
    adapter._text_batch_flush_deadline_seconds = lambda: 0.2
    adapter._text_batch_delay_for = lambda pending: 0
    seen, hello_entered, release = [], asyncio.Event(), asyncio.Event()

    async def handler(event: MessageEvent) -> None:
        seen.append(event.text)
        if event.text == "hello":
            hello_entered.set()
            await release.wait()  # never hands off

    adapter.handle_message = handler
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="60123@s.whatsapp.net", chat_type="dm", user_id="60123")
    await adapter._batch_text_or_dispatch(MessageEvent(text="hello", message_type=MessageType.TEXT, source=source))
    await asyncio.wait_for(hello_entered.wait(), timeout=5)

    try:
        await asyncio.wait_for(adapter._batch_text_or_dispatch(
            MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)), timeout=2)
        assert seen == ["hello", "/stop"] and not release.is_set()
    finally:
        release.set()
