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
    content = (
        "VNC 주소\n"
        "COPY_BUTTON: Mac mini IP | 100.96.33.123\n"
        "body remains\n"
        "COPY_BUTTON: bad | \n"
    )

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


@pytest.mark.asyncio
async def test_finalize_overflow_attaches_copy_keyboard_to_last_continuation(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    cast(Any, adapter).MAX_MESSAGE_LENGTH = 80
    bot = cast(Any, adapter._bot)
    bot.edit_message_text = AsyncMock(return_value=None)

    next_message_id = iter(range(778, 900))

    async def _send_message(**kwargs):
        message = MagicMock()
        message.message_id = next(next_message_id)
        return message

    bot.send_message = AsyncMock(side_effect=_send_message)
    content = "word " * 80 + "\nCOPY_BUTTON: Copy endpoint | https://example.test/path"

    result = await adapter.edit_message(
        chat_id="12345",
        message_id="777",
        content=content,
        finalize=True,
    )

    assert result.success is True
    assert bot.send_message.call_count > 0
    assert "reply_markup" not in bot.edit_message_text.call_args.kwargs
    for call in bot.send_message.call_args_list[:-1]:
        assert "reply_markup" not in call.kwargs
    final_kwargs = bot.send_message.call_args_list[-1].kwargs
    assert final_kwargs["reply_markup"].rows[0][0].kwargs["copy_text"].text == "https://example.test/path"
    delivered_text = bot.edit_message_text.call_args.kwargs["text"] + "".join(
        call.kwargs["text"] for call in bot.send_message.call_args_list
    )
    assert "COPY_BUTTON:" not in delivered_text


@pytest.mark.asyncio
async def test_finalize_edit_flood_retry_preserves_copy_keyboard(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    bot = cast(Any, adapter._bot)

    class RetryAfter(Exception):
        retry_after = 0

    bot.edit_message_text = AsyncMock(
        side_effect=[Exception("Markdown parse failed"), RetryAfter("retry after"), None]
    )

    result = await adapter.edit_message(
        chat_id="12345",
        message_id="777",
        content="Endpoint\nCOPY_BUTTON: Copy endpoint | https://example.test/path",
        finalize=True,
    )

    assert result.success is True
    retry_kwargs = bot.edit_message_text.call_args_list[-1].kwargs
    assert "COPY_BUTTON:" not in retry_kwargs["text"]
    assert retry_kwargs["reply_markup"].rows[0][0].kwargs["copy_text"].text == "https://example.test/path"


@pytest.mark.asyncio
async def test_finalize_reactive_overflow_single_chunk_preserves_copy_keyboard(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    cast(Any, adapter).MAX_MESSAGE_LENGTH = 80
    adapter.format_message = MagicMock(return_value="x" * 100)
    bot = cast(Any, adapter._bot)
    bot.edit_message_text = AsyncMock(
        side_effect=[Exception("message too long"), Exception("message too long"), None]
    )

    result = await adapter.edit_message(
        chat_id="12345",
        message_id="777",
        content="Endpoint\nCOPY_BUTTON: Copy endpoint | https://example.test/path",
        finalize=True,
    )

    assert result.success is True
    final_kwargs = bot.edit_message_text.call_args_list[-1].kwargs
    assert final_kwargs["reply_markup"].rows[0][0].kwargs["copy_text"].text == "https://example.test/path"
