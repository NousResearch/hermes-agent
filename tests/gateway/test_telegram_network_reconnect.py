"""
Tests for Telegram polling network error recovery.

Specifically tests the fix for #3173 — when start_polling() fails after a
network error, the adapter must self-reschedule the next reconnect attempt
rather than silently leaving polling dead.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


@pytest.mark.asyncio
async def test_connect_polling_continues_after_transient_delete_webhook_timeout(monkeypatch):
    """Transient delete_webhook timeouts should not abort Telegram polling startup."""
    adapter = _make_adapter()

    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder

    mock_bot = MagicMock()
    timed_out = type("TimedOut", (Exception,), {})
    mock_bot.delete_webhook = AsyncMock(side_effect=[timed_out("Timed out"), timed_out("Timed out"), timed_out("Timed out")])
    mock_bot.set_my_commands = AsyncMock()

    mock_updater = MagicMock()
    mock_updater.start_polling = AsyncMock()
    mock_updater.running = False

    mock_app = MagicMock()
    mock_app.bot = mock_bot
    mock_app.updater = mock_updater
    mock_app.initialize = AsyncMock()
    mock_app.start = AsyncMock()
    mock_app.add_handler = MagicMock()

    builder.build.return_value = mock_app

    monkeypatch.setattr("gateway.platforms.telegram.Application", MagicMock(builder=MagicMock(return_value=builder)))
    monkeypatch.setattr("gateway.platforms.telegram.HTTPXRequest", MagicMock())
    monkeypatch.setattr("gateway.platforms.telegram.Update", MagicMock(ALL_TYPES=[]))
    monkeypatch.setattr("gateway.platforms.telegram.TelegramMessageHandler", MagicMock())
    monkeypatch.setattr("gateway.platforms.telegram.CallbackQueryHandler", MagicMock())
    monkeypatch.setattr("gateway.platforms.telegram.filters", MagicMock(TEXT=1, COMMAND=2, LOCATION=4, VENUE=8, PHOTO=16, VIDEO=32, AUDIO=64, VOICE=128, Document=MagicMock(ALL=256), Sticker=MagicMock(ALL=512)))
    monkeypatch.setattr("gateway.platforms.telegram.resolve_proxy_url", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("gateway.platforms.telegram.logger", MagicMock())
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter, "_mark_connected", lambda: None)
    monkeypatch.setattr(adapter, "_setup_dm_topics", AsyncMock())

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("asyncio.sleep", _sleep)

    assert await adapter.connect() is True
    assert mock_bot.delete_webhook.await_count == 3
    mock_updater.start_polling.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconnect_self_schedules_on_start_polling_failure():
    """
    When start_polling() raises during a network error retry, the adapter must
    schedule a new _handle_polling_network_error task — otherwise polling stays
    dead with no further error callbacks to trigger recovery.

    Regression test for #3173: gateway becomes unresponsive after Telegram 502.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock(side_effect=Exception("Timed out"))

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # A retry task must have been added to _background_tasks
    pending = [t for t in adapter._background_tasks if not t.done()]
    assert len(pending) >= 1, (
        "Expected at least one self-rescheduled retry task in _background_tasks "
        f"after start_polling failure, got {len(pending)}"
    )

    # Clean up — cancel the pending retry so it doesn't run after the test
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_reconnect_does_not_self_schedule_when_fatal_error_set():
    """
    When a fatal error is already set, the failed reconnect should NOT create
    another retry task — the gateway is already shutting down this adapter.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1
    adapter._set_fatal_error("telegram_network_error", "already fatal", retryable=True)

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock(side_effect=Exception("Timed out"))

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    initial_count = len(adapter._background_tasks)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Timed out"))

    assert len(adapter._background_tasks) == initial_count, (
        "Should not schedule a retry when a fatal error is already set"
    )


@pytest.mark.asyncio
async def test_reconnect_success_resets_error_count():
    """
    When start_polling() succeeds, _polling_network_error_count should reset to 0.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 3

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()  # succeeds

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    assert adapter._polling_network_error_count == 0


@pytest.mark.asyncio
async def test_reconnect_triggers_fatal_after_max_retries():
    """
    After MAX_NETWORK_RETRIES attempts, the adapter should set a fatal error
    rather than retrying forever.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 10  # MAX_NETWORK_RETRIES

    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    mock_app = MagicMock()
    adapter._app = mock_app

    await adapter._handle_polling_network_error(Exception("still failing"))

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "telegram_network_error"
    fatal_handler.assert_called_once()
