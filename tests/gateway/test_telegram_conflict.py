import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_connect_rejects_same_host_token_lock(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (False, {"pid": 4242}),
    )

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram_token_lock"
    assert adapter.has_fatal_error is True
    assert "already using this Telegram bot token" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_polling_conflict_retries_before_fatal(monkeypatch):
    """A single 409 should trigger a retry, not an immediate fatal error."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()

    assert ok is True
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)
    assert callable(captured["error_callback"])

    conflict = type("Conflict", (Exception,), {})

    # First conflict: should retry, NOT be fatal
    captured["error_callback"](conflict("Conflict: terminated by other getUpdates request"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Give the scheduled task a chance to run
    for _ in range(10):
        await asyncio.sleep(0)

    assert adapter.has_fatal_error is False, "First conflict should not be fatal"
    assert adapter._polling_conflict_count == 0, "Count should reset after successful retry"


@pytest.mark.asyncio
async def test_polling_conflict_becomes_fatal_after_retries(monkeypatch):
    """After exhausting retries, the conflict should become fatal."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    # Make start_polling fail on retries to exhaust retries
    call_count = {"n": 0}

    async def failing_start_polling(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call (initial connect) succeeds
            captured["error_callback"] = kwargs["error_callback"]
        else:
            # Retry calls fail
            raise Exception("Connection refused")

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=failing_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock(), delete_webhook=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()
    assert ok is True

    conflict = type("Conflict", (Exception,), {})

    # Directly call _handle_polling_conflict to avoid event-loop scheduling
    # complexity.  Each call simulates one 409 from Telegram.
    for i in range(4):
        await adapter._handle_polling_conflict(
            conflict("Conflict: terminated by other getUpdates request")
        )

    # After 3 failed retries (count 1-3 each enter the retry branch but
    # start_polling raises), the 4th conflict pushes count to 4 which
    # exceeds MAX_CONFLICT_RETRIES (3), entering the fatal branch.
    assert adapter.fatal_error_code == "telegram_polling_conflict", (
        f"Expected fatal after 4 conflicts, got code={adapter.fatal_error_code}, "
        f"count={adapter._polling_conflict_count}"
    )
    assert adapter.has_fatal_error is True
    fatal_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_marks_retryable_fatal_error_for_startup_network_failure(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    builder = MagicMock()
    builder.token.return_value = builder
    app = SimpleNamespace(
        bot=SimpleNamespace(delete_webhook=AsyncMock(), set_my_commands=AsyncMock()),
        updater=SimpleNamespace(),
        add_handler=MagicMock(),
        initialize=AsyncMock(side_effect=RuntimeError("Temporary failure in name resolution")),
        start=AsyncMock(),
    )
    builder.build.return_value = app
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram_connect_error"
    assert adapter.fatal_error_retryable is True
    assert "Temporary failure in name resolution" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_connect_clears_webhook_before_polling(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    updater = SimpleNamespace(
        start_polling=AsyncMock(),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(
        delete_webhook=AsyncMock(),
        set_my_commands=AsyncMock(),
    )
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "gateway.platforms.telegram.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )

    ok = await adapter.connect()

    assert ok is True
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)


@pytest.mark.asyncio
async def test_disconnect_skips_inactive_updater_and_app(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    updater = SimpleNamespace(running=False, stop=AsyncMock())
    app = SimpleNamespace(
        updater=updater,
        running=False,
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    adapter._app = app

    warning = MagicMock()
    monkeypatch.setattr("gateway.platforms.telegram.logger.warning", warning)

    await adapter.disconnect()

    updater.stop.assert_not_awaited()
    app.stop.assert_not_awaited()
    app.shutdown.assert_awaited_once()
    warning.assert_not_called()
