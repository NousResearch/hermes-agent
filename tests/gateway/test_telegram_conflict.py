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

    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


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
async def test_polling_conflict_stops_polling_and_notifies_handler(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))
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
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock())
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

    ok = await adapter.connect()

    assert ok is True
    assert callable(captured["error_callback"])

    conflict = type("Conflict", (Exception,), {})
    captured["error_callback"](conflict("Conflict: terminated by other getUpdates request; make sure that only one bot instance is running"))

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert adapter.fatal_error_code == "telegram_polling_conflict"
    assert adapter.has_fatal_error is True
    updater.stop.assert_awaited()
    fatal_handler.assert_awaited_once()
