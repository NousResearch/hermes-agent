"""Tests for Telegram copy-to-clipboard inline buttons."""

import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
    try:
        import telegram  # noqa: F401
        if hasattr(telegram, "__file__"):
            return
    except ImportError:
        pass

    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules[name] = mod
    sys.modules["telegram.error"] = mod.error


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter, _extract_copy_buttons


@pytest.fixture(autouse=True)
def _unload_adapter_after_test():
    """Let later tests that inject fake telegram modules import a fresh adapter."""
    yield
    sys.modules.pop("plugins.platforms.telegram.adapter", None)
    pkg = sys.modules.get("plugins.platforms.telegram")
    if pkg is not None and getattr(pkg, "adapter", None) is telegram_adapter:
        try:
            delattr(pkg, "adapter")
        except AttributeError:
            pass


class FakeCopyTextButton:
    def __init__(self, text: str):
        self.text = text


class FakeInlineKeyboardButton:
    def __init__(self, text: str, **kwargs):
        self.text = text
        self.kwargs = kwargs


class FakeInlineKeyboardMarkup:
    def __init__(self, rows):
        self.rows = rows


def _make_adapter(monkeypatch):
    monkeypatch.setattr(telegram_adapter, "CopyTextButton", FakeCopyTextButton)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", FakeInlineKeyboardButton)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", FakeInlineKeyboardMarkup)

    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def test_extract_copy_buttons_strips_valid_markers_only():
    content = """VNC 주소
COPY_BUTTON: Mac mini IP | 100.96.33.123
body remains
COPY_BUTTON: bad | 
"""

    stripped, buttons = _extract_copy_buttons(content)

    assert buttons == [("Mac mini IP", "100.96.33.123")]
    assert "COPY_BUTTON: Mac mini IP" not in stripped
    assert "body remains" in stripped
    assert "COPY_BUTTON: bad |" in stripped


@pytest.mark.asyncio
async def test_send_attaches_copy_text_keyboard_and_strips_markers(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    mock_msg = MagicMock()
    mock_msg.message_id = 123
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock(return_value=mock_msg)

    result = await adapter.send(
        chat_id="12345",
        content=(
            "VNC 주소\n"
            "COPY_BUTTON: Mac mini IP | 100.96.33.123\n"
            "COPY_BUTTON: MacBook VNC | 100.99.47.75:5900"
        ),
    )

    assert result.success is True
    kwargs = bot.send_message.call_args.kwargs
    assert "COPY_BUTTON:" not in kwargs["text"]
    assert "VNC 주소" in kwargs["text"]

    markup = kwargs["reply_markup"]
    assert isinstance(markup, FakeInlineKeyboardMarkup)
    assert len(markup.rows) == 2
    first = markup.rows[0][0]
    second = markup.rows[1][0]
    assert first.text == "Mac mini IP"
    assert first.kwargs["copy_text"].text == "100.96.33.123"
    assert second.text == "MacBook VNC"
    assert second.kwargs["copy_text"].text == "100.99.47.75:5900"


@pytest.mark.asyncio
async def test_copy_button_only_message_gets_safe_fallback_text(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    mock_msg = MagicMock()
    mock_msg.message_id = 124
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock(return_value=mock_msg)

    result = await adapter.send(
        chat_id="12345",
        content="COPY_BUTTON: MacBook IP | 100.99.47.75",
    )

    assert result.success is True
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["text"] == "Copy buttons"
    assert kwargs["reply_markup"].rows[0][0].kwargs["copy_text"].text == "100.99.47.75"


@pytest.mark.asyncio
async def test_send_leaves_markers_visible_when_copy_text_button_unavailable(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(telegram_adapter, "CopyTextButton", None)
    mock_msg = MagicMock()
    mock_msg.message_id = 125
    bot = cast(Any, adapter._bot)
    bot.send_message = AsyncMock(return_value=mock_msg)

    result = await adapter.send(
        chat_id="12345",
        content="VNC\nCOPY_BUTTON: MacBook IP | 100.99.47.75",
    )

    assert result.success is True
    kwargs = bot.send_message.call_args.kwargs
    assert "COPY\\_BUTTON: MacBook IP \\| 100\\.99\\.47\\.75" in kwargs["text"]
    assert "reply_markup" not in kwargs


@pytest.mark.asyncio
async def test_finalize_edit_attaches_copy_text_keyboard_and_strips_markers(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    bot = cast(Any, adapter._bot)
    bot.edit_message_text = AsyncMock(return_value=None)

    result = await adapter.edit_message(
        chat_id="12345",
        message_id="777",
        content=(
            "VNC 주소\n"
            "COPY_BUTTON: Mac mini IP | 100.96.33.123\n"
            "COPY_BUTTON: MacBook VNC | 100.99.47.75:5900"
        ),
        finalize=True,
    )

    assert result.success is True
    kwargs = bot.edit_message_text.call_args.kwargs
    assert kwargs["message_id"] == 777
    assert "COPY_BUTTON:" not in kwargs["text"]
    assert "VNC 주소" in kwargs["text"]
    assert kwargs["reply_markup"].rows[0][0].kwargs["copy_text"].text == "100.96.33.123"
    assert kwargs["reply_markup"].rows[1][0].kwargs["copy_text"].text == "100.99.47.75:5900"
