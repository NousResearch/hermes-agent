"""Standalone _send_telegram honors explicit payload_type (TKT-0033, Option 1).

The legacy standalone path auto-detected HTML via a content regex and picked
ParseMode.HTML when matched. That sniff is removed: the helper now resolves
parse mode exclusively from the declared ``payload_type`` contract, reusing
``TelegramAdapter._resolve_send_format`` so there is one source of truth.

- ``payload_type="text/html"`` → raw content + ParseMode.HTML
- ``payload_type="text/markdown"`` (default) → MarkdownV2 conversion + ParseMode.MARKDOWN_V2
- Content containing ``<b>`` with no declared payload_type must NOT be
  sniffed into HTML; it goes MarkdownV2 (and would be caught upstream by the
  Task 4 validator if it truly was HTML).
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# python-telegram-bot is an optional dep — skip when absent.
pytest.importorskip("telegram", reason="python-telegram-bot not installed")


def _install_telegram_mock(monkeypatch, bot):
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    telegram_mod = SimpleNamespace(Bot=lambda token: bot, constants=constants_mod)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _make_bot():
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_photo = AsyncMock()
    bot.send_video = AsyncMock()
    bot.send_voice = AsyncMock()
    bot.send_audio = AsyncMock()
    bot.send_document = AsyncMock()
    return bot


class TestSendTelegramPayloadTypeContract:
    """_send_telegram must resolve parse mode from payload_type, not content."""

    def test_html_payload_type_uses_html_parse_mode(self, monkeypatch):
        """Declared text/html → raw content + ParseMode.HTML, no sniffing."""
        from tools.send_message_tool import _send_telegram

        bot = _make_bot()
        _install_telegram_mock(monkeypatch, bot)

        raw = "<b>Report</b><ul><li>one</li></ul>"
        result = asyncio.run(
            _send_telegram("tok", "123", raw, payload_type="text/html")
        )

        assert result["success"] is True
        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "HTML"
        assert kwargs["text"] == raw  # raw, not MarkdownV2-escaped

    def test_default_payload_type_uses_markdownv2(self, monkeypatch):
        """No payload_type declared → MarkdownV2 path, even if content has <b>."""
        from tools.send_message_tool import _send_telegram

        bot = _make_bot()
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram("tok", "123", "<b>Hello</b> world")
        )

        assert result["success"] is True
        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "MarkdownV2"
        # The content must have been through format_message (MarkdownV2 escaped)
        assert kwargs["text"] != "<b>Hello</b> world"

    def test_explicit_markdown_payload_type_uses_markdownv2(self, monkeypatch):
        """Declared text/markdown → MarkdownV2 path, even if content has <b>."""
        from tools.send_message_tool import _send_telegram

        bot = _make_bot()
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram("tok", "123", "<b>Hello</b> world", payload_type="text/markdown")
        )

        assert result["success"] is True
        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "MarkdownV2"

    def test_unknown_payload_type_falls_back_to_markdownv2(self, monkeypatch):
        """Unknown payload_type (e.g. application/json) → MarkdownV2 fallback."""
        from tools.send_message_tool import _send_telegram

        bot = _make_bot()
        _install_telegram_mock(monkeypatch, bot)

        result = asyncio.run(
            _send_telegram("tok", "123", "data", payload_type="application/json")
        )

        assert result["success"] is True
        bot.send_message.assert_awaited_once()
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == "MarkdownV2"
