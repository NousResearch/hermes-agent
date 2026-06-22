"""TelegramAdapter send-path health gating after reconnect storms.

After sustained Bad Gateway / TimedOut reconnect cycles, the PTB httpx client
can enter a wedged state where ``bot.send_message()`` returns a valid Message
but nothing reaches the recipient.  ``_send_path_degraded`` short-circuits
``send()`` so cron's live-adapter branch falls through to standalone HTTP.
"""
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    return adapter


@pytest.mark.asyncio
async def test_send_succeeds_when_path_healthy():
    """Healthy adapter delivers normally; send_message is called."""
    adapter = _make_adapter()
    assert adapter._send_path_degraded is False

    result = await adapter.send("123", "hello")

    assert result.success is True
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_send_short_circuits_when_path_degraded():
    """Degraded adapter returns failure WITHOUT calling send_message,
    so cron's live-adapter branch falls through to standalone HTTP."""
    adapter = _make_adapter()
    adapter._send_path_degraded = True

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.error == "send_path_degraded"
    assert result.retryable is True
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_uses_adapter_backpressure_for_concurrent_calls():
    """Concurrent sends are gated before they enter python-telegram-bot.

    This protects the shared PTB/httpx pool from digest/status bursts that
    otherwise create many simultaneous send_message calls.
    """
    adapter = _make_adapter()
    adapter._send_semaphore = asyncio.Semaphore(1)
    adapter.send_typing = AsyncMock()
    active = 0
    max_active = 0

    async def fake_send_message(**_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return MagicMock(message_id=42)

    adapter._bot.send_message = AsyncMock(side_effect=fake_send_message)

    results = await asyncio.gather(
        adapter.send("123", "one"),
        adapter.send("123", "two"),
        adapter.send("123", "three"),
    )

    assert [r.success for r in results] == [True, True, True]
    assert max_active == 1
    assert adapter._bot.send_message.await_count == 3


@pytest.mark.asyncio
async def test_send_queue_timeout_returns_retryable_failure():
    """A saturated Telegram send queue fails fast instead of wedging Hermes."""
    adapter = _make_adapter()
    adapter._send_semaphore = asyncio.Semaphore(1)
    adapter._telegram_send_queue_timeout_seconds = 0.01
    adapter.send_typing = AsyncMock()
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_send_message(**_kwargs):
        first_started.set()
        await release_first.wait()
        return MagicMock(message_id=42)

    adapter._bot.send_message = AsyncMock(side_effect=fake_send_message)

    first = asyncio.create_task(adapter.send("123", "first"))
    await first_started.wait()
    second = await adapter.send("123", "second")
    release_first.set()
    first_result = await first

    assert first_result.success is True
    assert second.success is False
    assert second.retryable is True
    assert "telegram_send_queue_timeout" in second.error


@pytest.mark.asyncio
async def test_reconnect_storm_sets_and_heartbeat_clears_flag(monkeypatch):
    """_handle_polling_network_error sets the flag; a successful heartbeat
    probe in _verify_polling_after_reconnect clears it."""
    adapter = _make_adapter()
    adapter._app = MagicMock()
    adapter._app.updater = MagicMock()
    adapter._app.updater.running = True
    adapter._app.updater.stop = AsyncMock()
    adapter._app.updater.start_polling = AsyncMock()
    adapter._app.bot = MagicMock()
    adapter._app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._polling_error_callback_ref = AsyncMock()
    monkeypatch.setattr(
        "gateway.platforms.telegram.Update", MagicMock(ALL_TYPES=[])
    )

    await adapter._handle_polling_network_error(OSError("Bad Gateway"))
    assert adapter._send_path_degraded is True

    with patch("gateway.platforms.telegram.asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()
    assert adapter._send_path_degraded is False
