"""TelegramAdapter send-path health gating after reconnect storms.

After sustained Bad Gateway / TimedOut reconnect cycles, the PTB httpx client
can enter a wedged state where ``bot.send_message()`` returns a valid Message
but nothing reaches the recipient.  ``_send_path_degraded`` short-circuits
``send()`` so cron's live-adapter branch falls through to standalone HTTP.
"""
import sys
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

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


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
async def test_get_me_success_without_polling_progress_does_not_heal(monkeypatch):
    """A responsive general Bot API path is not proof that getUpdates works."""
    adapter = _make_adapter()
    adapter._app = MagicMock()
    adapter._app.updater = MagicMock()
    adapter._app.updater.running = True
    adapter._app.bot = MagicMock()
    adapter._app.bot.get_me = AsyncMock(return_value=MagicMock())

    generation, progress = adapter._begin_polling_generation()
    recovery = MagicMock()
    monkeypatch.setattr(adapter, "_schedule_polling_recovery", recovery)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter._POLLING_PROGRESS_TIMEOUT", 0,
        raising=False,
    )
    await adapter._verify_polling_after_reconnect(generation, progress)

    adapter._app.bot.get_me.assert_awaited_once()
    recovery.assert_called_once()
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_successful_reconnect_waits_for_get_updates_progress(monkeypatch):
    """start_polling() return alone cannot heal; matching progress can."""
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
        "plugins.platforms.telegram.adapter.Update", MagicMock(ALL_TYPES=[])
    )
    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(OSError("Bad Gateway"))

    verifier = adapter._polling_progress_verifier_task
    assert adapter._send_path_degraded is True
    assert adapter._polling_network_error_count == 1
    blocked = await adapter.send("123", "hello")
    assert blocked.success is False

    adapter._record_polling_progress(adapter._polling_generation)
    await verifier
    assert adapter._send_path_degraded is False
    assert adapter._polling_network_error_count == 0
    result = await adapter.send("123", "hello")
    assert result.success is True
