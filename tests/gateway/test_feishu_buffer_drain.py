"""Regression coverage for Feishu inbound text-batch teardown."""

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.feishu import adapter as feishu_adapter_module
from plugins.platforms.feishu.adapter import FeishuAdapter


def _event(text: str, chat_id: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=feishu_adapter_module.Platform.FEISHU,
            chat_id=chat_id,
            chat_type="dm",
            user_id=f"user-{chat_id}",
        ),
        message_id=f"message-{chat_id}",
    )


def _adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig())
    adapter._text_batch_delay_seconds = 60
    adapter._text_batch_split_delay_seconds = 60
    adapter._persist_seen_message_ids = Mock()
    adapter._release_app_lock = AsyncMock()
    adapter._mark_disconnected = Mock()
    adapter._stop_webhook_server = AsyncMock()
    adapter._teardown_ws_thread = AsyncMock()
    adapter._shutdown_sdk_executor = Mock()
    return adapter


async def _cancel_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_disconnect_awaits_popped_dispatch_and_isolates_multiple_batch_failures():
    adapter = _adapter()
    popped_started = asyncio.Event()
    release_popped = asyncio.Event()
    popped_finished = asyncio.Event()
    failed_started = asyncio.Event()
    healthy_finished = asyncio.Event()
    seen: list[str] = []

    async def handle(event: MessageEvent) -> None:
        seen.append(event.text)
        if event.text == "popped":
            popped_started.set()
            await release_popped.wait()
            popped_finished.set()
        elif event.text == "fails":
            failed_started.set()
            raise RuntimeError("one batch failed")
        elif event.text == "healthy":
            healthy_finished.set()

    adapter._handle_message_with_guards = handle
    adapter._text_batch_delay_seconds = 0
    adapter._text_batch_split_delay_seconds = 0
    await adapter._dispatch_inbound_event(_event("popped", "chat-popped"))
    await asyncio.wait_for(popped_started.wait(), timeout=2.0)

    # The timer has crossed the pop boundary and is blocked in its shielded dispatch. Keep two
    # other keys pending so teardown must drain them independently and concurrently.
    adapter._text_batch_delay_seconds = 60
    adapter._text_batch_split_delay_seconds = 60
    await adapter._dispatch_inbound_event(_event("fails", "chat-fails"))
    await adapter._dispatch_inbound_event(_event("healthy", "chat-healthy"))

    disconnect = asyncio.create_task(adapter.disconnect())
    try:
        await asyncio.wait_for(
            asyncio.gather(failed_started.wait(), healthy_finished.wait()),
            timeout=2.0,
        )
        release_popped.set()
        await asyncio.wait_for(disconnect, timeout=2.0)
    finally:
        release_popped.set()
        if not disconnect.done():
            disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)
        await _cancel_tasks(list(adapter._pending_text_batch_tasks.values()))

    assert popped_finished.is_set()
    assert Counter(seen) == Counter({"popped": 1, "fails": 1, "healthy": 1})
    assert adapter._pending_text_batches == {}
    assert adapter._pending_text_batch_tasks == {}
    assert adapter._pending_text_batch_counts == {}
    adapter._stop_webhook_server.assert_awaited_once()
    adapter._teardown_ws_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_bounds_slow_drain_and_reconnect_has_no_stale_ingress(monkeypatch):
    adapter = _adapter()
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.05")
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()
    fresh_finished = asyncio.Event()
    handler_tasks: list[asyncio.Task] = []
    seen: list[str] = []

    async def handle(event: MessageEvent) -> None:
        task = asyncio.current_task()
        assert task is not None
        handler_tasks.append(task)
        seen.append(event.text)
        if event.text == "slow":
            slow_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                slow_cancelled.set()
                raise
        elif event.text == "fresh":
            fresh_finished.set()

    adapter._handle_message_with_guards = handle
    await adapter._dispatch_inbound_event(_event("slow", "chat-slow"))

    disconnect = asyncio.create_task(adapter.disconnect())
    try:
        await asyncio.wait_for(slow_started.wait(), timeout=2.0)
        # Ingress after teardown has quiesced must not enter the snapshot or survive its reset.
        await adapter._dispatch_inbound_event(_event("late", "chat-late"))
        await asyncio.wait_for(disconnect, timeout=2.0)
    finally:
        if not disconnect.done():
            disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)
        await _cancel_tasks(list(adapter._pending_text_batch_tasks.values()))

    assert slow_cancelled.is_set()
    assert seen == ["slow"]
    assert all(task.done() for task in handler_tasks)
    assert adapter._pending_text_batches == {}
    assert adapter._pending_text_batch_tasks == {}
    assert adapter._pending_text_batch_counts == {}
    adapter._stop_webhook_server.assert_awaited_once()
    adapter._teardown_ws_thread.assert_awaited_once()

    # A reconnect re-opens admission without reviving the late batch or any old timer task.
    adapter._app_id = "app"
    adapter._app_secret = "x"
    adapter._connection_mode = "webhook"
    adapter._verification_token = "token"
    adapter._connect_with_retry = AsyncMock()
    adapter._wire_plugin_handlers = Mock()
    monkeypatch.setattr(feishu_adapter_module, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(feishu_adapter_module, "acquire_scoped_lock", lambda *args, **kwargs: (True, None))

    assert await adapter.connect(is_reconnect=True)
    adapter._text_batch_delay_seconds = 0
    adapter._text_batch_split_delay_seconds = 0
    await adapter._dispatch_inbound_event(_event("fresh", "chat-fresh"))
    await asyncio.wait_for(fresh_finished.wait(), timeout=2.0)
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())

    assert seen == ["slow", "fresh"]
    assert adapter._pending_text_batches == {}
    assert adapter._pending_text_batch_tasks == {}
