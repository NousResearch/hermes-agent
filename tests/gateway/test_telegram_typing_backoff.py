"""Telegram typing indicator transient backoff tests."""

import asyncio
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)
from gateway.config import Platform, PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._test_typing_lease_id = adapter._begin_typing_lease("123", "test-session")
    return adapter


def _active_turn_metadata(adapter, **extra):
    return {"_hermes_typing_lease_id": adapter._test_typing_lease_id, **extra}


@pytest.mark.asyncio
async def test_late_typing_refresh_from_released_turn_is_blocked():
    adapter = _make_adapter()

    await adapter.send_typing(
        "123", metadata={"_hermes_typing_lease_id": "finished-turn"}
    )

    adapter._bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_typing_action_holds_finalization_fence_until_transport_returns():
    adapter = _make_adapter()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_action(**kwargs):
        entered.set()
        await release.wait()

    adapter._bot.send_chat_action = AsyncMock(side_effect=blocked_action)
    typing_task = asyncio.create_task(
        adapter.send_typing("123", metadata=_active_turn_metadata(adapter))
    )
    await entered.wait()
    fence_task = asyncio.create_task(
        adapter._fence_typing_lease_before_final("123", adapter._test_typing_lease_id)
    )
    await asyncio.sleep(0)
    assert not fence_task.done()

    release.set()
    await typing_task
    await fence_task


@pytest.mark.asyncio
async def test_final_notify_send_fences_typing_before_visible_message():
    adapter = _make_adapter()
    order = []

    async def record_fence(chat_id, lease_id):
        order.append("fence")

    async def record_send(**kwargs):
        order.append("send")
        return SimpleNamespace(message_id="final-1")

    adapter._fence_typing_lease_before_final = AsyncMock(side_effect=record_fence)
    adapter._bot.send_message = AsyncMock(side_effect=record_send)

    await adapter.send(
        "123", "final response", metadata=_active_turn_metadata(adapter, notify=True)
    )

    assert order[:2] == ["fence", "send"]


@pytest.mark.asyncio
async def test_final_edit_fences_typing_before_visible_edit():
    adapter = _make_adapter()
    order = []

    async def record_fence(chat_id, lease_id):
        order.append("fence")

    async def record_edit(**kwargs):
        order.append("edit")
        return SimpleNamespace(message_id="final-1")

    adapter._fence_typing_lease_before_final = AsyncMock(side_effect=record_fence)
    adapter._bot.edit_message_text = AsyncMock(side_effect=record_edit)

    await adapter.edit_message(
        "123", "1", "final response", finalize=True,
        metadata=_active_turn_metadata(adapter),
    )

    assert order[:2] == ["fence", "edit"]
    adapter._bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_intermediate_finalized_edit_preserves_lease_and_rearms_typing():
    """Formatting-final is not necessarily the final delivery for the turn."""
    adapter = _make_adapter()
    order = []

    async def record_fence(chat_id, lease_id):
        order.append("fence")

    async def record_edit(**kwargs):
        order.append("edit")
        return SimpleNamespace(message_id="1")

    async def record_typing(**kwargs):
        order.append("typing")

    adapter._fence_typing_lease_before_final = AsyncMock(side_effect=record_fence)
    adapter._bot.edit_message_text = AsyncMock(side_effect=record_edit)
    adapter._bot.send_chat_action = AsyncMock(side_effect=record_typing)
    metadata = _active_turn_metadata(adapter)

    result = await adapter.edit_message(
        "123",
        "1",
        "sealed intermediate segment",
        finalize=True,
        is_turn_final=False,
        metadata=metadata,
    )

    assert result.success
    assert order == ["edit", "typing"]
    assert adapter._typing_lease_allows("123", metadata)


@pytest.mark.asyncio
async def test_status_edit_is_intermediate_and_preserves_typing_lease():
    adapter = _make_adapter()
    adapter._status_message_ids[("123", "compression")] = "1"
    adapter._bot.edit_message_text = AsyncMock(
        return_value=SimpleNamespace(message_id="1")
    )
    metadata = _active_turn_metadata(adapter)

    result = await adapter.send_or_update_status(
        "123", "compression", "still working", metadata=metadata
    )

    assert result.success
    adapter._bot.send_chat_action.assert_awaited_once()
    assert adapter._typing_lease_allows("123", metadata)


@pytest.mark.asyncio
async def test_first_formatting_final_send_preserves_and_rearms_typing():
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(
        return_value=SimpleNamespace(message_id="1")
    )
    metadata = _active_turn_metadata(adapter)
    consumer = GatewayStreamConsumer(adapter, "123", metadata=metadata)

    delivered = await consumer._send_or_edit(
        "sealed intermediate segment",
        finalize=True,
        is_turn_final=False,
    )

    assert delivered
    adapter._bot.send_chat_action.assert_awaited_once()
    assert adapter._typing_lease_allows("123", metadata)


@pytest.mark.asyncio
async def test_fresh_replacement_for_intermediate_segment_preserves_typing():
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(
        return_value=SimpleNamespace(message_id="2")
    )
    adapter._bot.delete_message = AsyncMock()
    metadata = _active_turn_metadata(adapter)
    consumer = GatewayStreamConsumer(adapter, "123", metadata=metadata)
    consumer._message_id = "1"
    consumer._preview_message_ids = {"1"}

    delivered = await consumer._try_fresh_final(
        "sealed intermediate segment",
        is_turn_final=False,
    )

    assert delivered
    adapter._bot.send_chat_action.assert_awaited_once()
    assert adapter._typing_lease_allows("123", metadata)


@pytest.mark.asyncio
async def test_intermediate_edit_rearms_typing_with_active_lease():
    adapter = _make_adapter()
    order = []

    async def record_edit(**kwargs):
        order.append("edit")
        return SimpleNamespace(message_id="1")

    async def record_typing(**kwargs):
        order.append("typing")

    adapter._bot.edit_message_text = AsyncMock(side_effect=record_edit)
    adapter._bot.send_chat_action = AsyncMock(side_effect=record_typing)

    result = await adapter.edit_message(
        "123",
        "1",
        "progress update",
        finalize=False,
        metadata=_active_turn_metadata(adapter),
    )

    assert result.success
    assert order == ["edit", "typing"]


@pytest.mark.asyncio
async def test_failed_intermediate_edit_does_not_rearm_typing():
    adapter = _make_adapter()
    adapter._bot.edit_message_text = AsyncMock(side_effect=ValueError("edit failed"))

    result = await adapter.edit_message(
        "123",
        "1",
        "progress update",
        finalize=False,
        metadata=_active_turn_metadata(adapter),
    )

    assert not result.success
    adapter._bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnected_adapter_forwards_heartbeat_to_live_replacement():
    ingress = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    replacement = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    replacement._bot = AsyncMock()
    runner = SimpleNamespace(adapters={Platform.TELEGRAM: replacement})
    ingress.gateway_runner = runner
    replacement.gateway_runner = runner
    lease_id = ingress._begin_typing_lease("123", "test-session")
    metadata = {"_hermes_typing_lease_id": lease_id}
    ingress._bot = None

    await ingress.send_typing("123", metadata=metadata)

    replacement._bot.send_chat_action.assert_awaited_once()
    assert replacement._typing_lease_allows("123", metadata)


@pytest.mark.asyncio
async def test_cancelled_intermediate_rearm_drains_before_final_fence():
    adapter = _make_adapter()
    entered = asyncio.Event()
    release = asyncio.Event()

    adapter._bot.edit_message_text = AsyncMock(return_value=SimpleNamespace(message_id="1"))

    async def blocked_typing(**kwargs):
        entered.set()
        await release.wait()

    adapter._bot.send_chat_action = AsyncMock(side_effect=blocked_typing)
    edit_task = asyncio.create_task(
        adapter.edit_message(
            "123",
            "1",
            "progress update",
            finalize=False,
            metadata=_active_turn_metadata(adapter),
        )
    )
    await entered.wait()

    edit_task.cancel()
    fence_task = asyncio.create_task(
        adapter._fence_typing_lease_before_final("123", adapter._test_typing_lease_id)
    )
    await asyncio.sleep(0)

    assert not fence_task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await edit_task
    await fence_task


@pytest.mark.asyncio
async def test_typing_disabled_never_calls_telegram_action():
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="test-token", typing_indicator=False)
    )
    adapter._bot = AsyncMock()
    adapter._test_typing_lease_id = adapter._begin_typing_lease("123", "test-session")

    await adapter.send_typing("123", metadata=_active_turn_metadata(adapter))

    adapter._bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_typing_transient_failure_enters_cooldown(monkeypatch):
    adapter = _make_adapter()
    now = {"value": 1000.0}
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.get_running_loop", lambda: type("Loop", (), {"time": lambda self: now["value"]})())
    monkeypatch.setattr(adapter, "_telegram_typing_cooldown_seconds", 30.0, raising=False)

    async def fail_once(**kwargs):
        raise OSError("temporary telegram network failure")

    adapter._bot.send_chat_action = AsyncMock(side_effect=fail_once)

    await adapter.send_typing("123", metadata=_active_turn_metadata(adapter))
    await adapter.send_typing("123", metadata=_active_turn_metadata(adapter))

    assert adapter._bot.send_chat_action.await_count == 1
    assert adapter._telegram_typing_cooldown_until["123"] == pytest.approx(1030.0)

    now["value"] = 1031.0
    adapter._bot.send_chat_action = AsyncMock(return_value=None)
    await adapter.send_typing("123", metadata=_active_turn_metadata(adapter))

    assert adapter._bot.send_chat_action.await_count == 1
    assert "123" not in adapter._telegram_typing_cooldown_until


@pytest.mark.asyncio
async def test_typing_retry_after_timedelta_honors_server_delay(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.get_running_loop",
        lambda: type("Loop", (), {"time": lambda self: 1000.0})(),
    )

    class RetryAfterError(OSError):
        retry_after = timedelta(seconds=90)

    adapter._bot.send_chat_action = AsyncMock(
        side_effect=RetryAfterError("retry after 90 seconds")
    )

    await adapter.send_typing("123", metadata=_active_turn_metadata(adapter))

    assert adapter._telegram_typing_cooldown_until["123"] == pytest.approx(1090.0)


@pytest.mark.asyncio
async def test_typing_dm_topic_fallback_success_does_not_cool_down(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.get_running_loop", lambda: type("Loop", (), {"time": lambda self: 10.0})())

    calls = []

    async def send_chat_action(**kwargs):
        calls.append(kwargs)
        if "message_thread_id" in kwargs:
            raise RuntimeError("message thread not found")
        return None

    adapter._bot.send_chat_action = AsyncMock(side_effect=send_chat_action)

    await adapter.send_typing(
        "123",
        metadata=_active_turn_metadata(
            adapter,
            thread_id="99", telegram_dm_topic_reply_fallback=True
        ),
    )

    assert len(calls) == 2
    assert "123" not in adapter._telegram_typing_cooldown_until


@pytest.mark.asyncio
async def test_typing_bad_thread_failure_does_not_cool_down(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.get_running_loop", lambda: type("Loop", (), {"time": lambda self: 10.0})())

    async def bad_request(**kwargs):
        raise ValueError("message thread not found")

    adapter._bot.send_chat_action = AsyncMock(side_effect=bad_request)

    await adapter.send_typing("123", metadata=_active_turn_metadata(adapter, thread_id="99"))
    await adapter.send_typing("123", metadata=_active_turn_metadata(adapter, thread_id="99"))

    assert adapter._bot.send_chat_action.await_count == 2
    assert "123" not in adapter._telegram_typing_cooldown_until
