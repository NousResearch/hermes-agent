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
async def test_send_reports_retryable_failure_when_standalone_fallback_fails(monkeypatch):
    """Degraded adapter never uses PTB; failed fallback remains retryable."""
    adapter = _make_adapter()
    adapter._send_path_degraded = True
    standalone = AsyncMock(return_value={"error": "fallback failed"})
    monkeypatch.setattr("tools.send_message_tool._send_telegram", standalone)

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.error == "send_path_degraded: standalone fallback failed: fallback failed"
    assert result.retryable is True
    standalone.assert_awaited_once()
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_degraded_send_path_uses_standalone_fallback(monkeypatch):
    """Gateway replies must still deliver while the PTB send path is gated."""
    adapter = _make_adapter()
    adapter.config.token = "tok"
    adapter._send_path_degraded = True
    standalone = AsyncMock(
        return_value={
            "success": True,
            "platform": "telegram",
            "chat_id": "123",
            "message_id": "99",
        }
    )
    monkeypatch.setattr("tools.send_message_tool._send_telegram", standalone)

    result = await adapter.send("123", "hello")

    assert result.success is True
    assert result.message_id == "99"
    standalone.assert_awaited_once_with(
        "tok",
        "123",
        "hello",
        thread_id=None,
        disable_link_previews=False,
    )
    adapter._bot.send_message.assert_not_awaited()


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
