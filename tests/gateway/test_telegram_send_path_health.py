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
async def test_business_send_uses_bot_identity_by_default():
    """Business routing metadata must not make agent output human-authored."""
    adapter = _make_adapter()

    result = await adapter.send(
        "8533179145",
        "OK",
        metadata={"business_connection_id": "biz-123"},
    )

    assert result.success is True
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert "business_connection_id" not in kwargs


@pytest.mark.asyncio
async def test_business_send_as_account_requires_explicit_opt_in():
    """Trusted callers may opt in explicitly; default remains fail-closed."""
    adapter = _make_adapter()

    result = await adapter.send(
        "8533179145",
        "OK",
        metadata={
            "business_connection_id": "biz-123",
            "telegram_business_send_as_account": True,
        },
    )

    assert result.success is True
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs["business_connection_id"] == "biz-123"


@pytest.mark.asyncio
async def test_reconnect_storm_sets_and_heartbeat_clears_flag(monkeypatch):
    """_handle_polling_network_error sets the flag while reconnecting; if the
    reconnect attempt itself raises (polling not yet healthy), the flag stays
    True until a later successful heartbeat probe in
    _verify_polling_after_reconnect clears it."""
    adapter = _make_adapter()
    adapter._app = MagicMock()
    adapter._app.updater = MagicMock()
    adapter._app.updater.running = True
    adapter._app.updater.stop = AsyncMock()
    # First start_polling attempt fails — the reconnect handler must leave the
    # flag set (path still unhealthy) and not clear it prematurely.
    adapter._app.updater.start_polling = AsyncMock(side_effect=OSError("still down"))
    adapter._app.bot = MagicMock()
    adapter._app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._polling_error_callback_ref = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Update", MagicMock(ALL_TYPES=[])
    )
    # Suppress the self-rescheduled retry so the test doesn't recurse.
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.ensure_future", MagicMock()
    )

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(OSError("Bad Gateway"))
    # start_polling failed → path still degraded.
    assert adapter._send_path_degraded is True

    # Now the deferred probe runs against a recovered (running) updater and
    # a responsive bot — it clears the flag.
    adapter._app.updater.running = True
    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()
    assert adapter._send_path_degraded is False


@pytest.mark.asyncio
async def test_successful_reconnect_clears_flag_without_probe(monkeypatch):
    """Regression for #35205: a successful start_polling() clears
    _send_path_degraded immediately, so outbound sends are not blocked for
    the full HEARTBEAT_PROBE_DELAY window (and never get stuck True if the
    deferred probe is never scheduled / never runs)."""
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
    # Don't let the deferred probe run — prove the clear happens in the
    # reconnect handler itself, not in _verify_polling_after_reconnect.
    monkeypatch.setattr(
        adapter, "_verify_polling_after_reconnect", AsyncMock()
    )

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(OSError("Bad Gateway"))

    assert adapter._send_path_degraded is False
    assert adapter._polling_network_error_count == 0
    # And send() works again right away.
    result = await adapter.send("123", "hello")
    assert result.success is True
