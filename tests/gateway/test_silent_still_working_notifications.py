"""Tests for silent gateway progress notifications on Telegram."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig


def _run(coro):
    return asyncio.run(coro)


def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
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
from hermes_cli.config import DEFAULT_CONFIG  # noqa: E402


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    msg = MagicMock()
    msg.message_id = 42
    adapter._bot.send_message = AsyncMock(return_value=msg)
    return adapter


def test_default_config_makes_gateway_notifications_silent():
    assert DEFAULT_CONFIG["agent"]["gateway_notify_silent"] is True


def test_telegram_send_maps_silent_metadata_to_disable_notification():
    adapter = _make_adapter()

    result = _run(adapter.send("12345", "Still working", metadata={"silent": True}))

    assert result.success is True
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs["disable_notification"] is True


def test_telegram_send_maps_disable_notification_metadata():
    adapter = _make_adapter()

    result = _run(
        adapter.send(
            "12345",
            "Still working",
            metadata={"disable_notification": "true"},
        )
    )

    assert result.success is True
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs["disable_notification"] is True


def test_telegram_send_defaults_to_audible_notifications():
    adapter = _make_adapter()

    result = _run(adapter.send("12345", "Normal reply"))

    assert result.success is True
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs["disable_notification"] is False
