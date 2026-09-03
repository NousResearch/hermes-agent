"""Tests for RTL rich messages (Bot API 10.1 InputRichMessage.is_rtl).

When ``platforms.telegram.extra.rich_rtl`` is enabled, rich-message payloads
whose content is predominantly Hebrew/Arabic (>30% of alphabetic characters)
carry ``is_rtl: True`` so Telegram renders them right-to-left. English or
mixed-LTR content stays untouched, and the flag is off by default.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from plugins.platforms.telegram.adapter import TelegramAdapter, _is_rtl_text

ARABIC_CONTENT = "## تقرير السوق\n\n| السهم | الحالة |\n|---|---|\n| أرامكو | صاعد |"
HEBREW_CONTENT = "## דוח השוק\n\n| מניה | מצב |\n|---|---|\n| טבע | עולה |"
ENGLISH_CONTENT = "## Market report\n\n| Ticker | Status |\n|---|---|\n| SPY | up |"


def _make_adapter(extra=None):
    """Build a TelegramAdapter with a mock bot wired for the rich path."""
    from gateway.config import PlatformConfig

    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"rich_messages": True, **(extra or {})},
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    bot.send_message_draft = AsyncMock(return_value=True)
    bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=1))
    bot.delete_message = AsyncMock(return_value=True)
    adapter._bot = bot
    return adapter


class TestIsRtlText:
    def test_arabic_dominant_true(self):
        assert _is_rtl_text("هذا تقرير السوق اليومي: SPY ارتفع 1.2%") is True

    def test_hebrew_dominant_true(self):
        assert _is_rtl_text("דוח השוק: SPY עלה היום") is True

    def test_english_false(self):
        assert _is_rtl_text("Market report: SPY rose 1.2% today") is False

    def test_empty_false(self):
        assert _is_rtl_text("") is False

    def test_no_letters_false(self):
        assert _is_rtl_text("12345 !@# 67890") is False

    def test_minority_rtl_false(self):
        # Two Arabic words buried in English prose stay LTR.
        text = "The term انت appears once and قيمة twice in this English text"
        assert _is_rtl_text(text) is False


class TestRichRtlPayload:
    def test_rtl_extra_off_by_default(self):
        adapter = _make_adapter()
        payload = adapter._rich_message_payload(ARABIC_CONTENT)
        assert "is_rtl" not in payload

    def test_rtl_extra_on_arabic_sets_flag(self):
        adapter = _make_adapter(extra={"rich_rtl": True})
        payload = adapter._rich_message_payload(ARABIC_CONTENT)
        assert payload["is_rtl"] is True
        assert "markdown" in payload

    def test_rtl_extra_on_hebrew_sets_flag(self):
        adapter = _make_adapter(extra={"rich_rtl": True})
        payload = adapter._rich_message_payload(HEBREW_CONTENT)
        assert payload["is_rtl"] is True

    def test_rtl_extra_on_english_no_flag(self):
        adapter = _make_adapter(extra={"rich_rtl": True})
        payload = adapter._rich_message_payload(ENGLISH_CONTENT)
        assert "is_rtl" not in payload

    def test_rtl_extra_string_true_coerced(self):
        adapter = _make_adapter(extra={"rich_rtl": "true"})
        payload = adapter._rich_message_payload(ARABIC_CONTENT)
        assert payload["is_rtl"] is True

    def test_forwarded_in_send_payload(self):
        adapter = _make_adapter(extra={"rich_rtl": True})
        import asyncio

        asyncio.run(adapter._try_send_rich("123", ARABIC_CONTENT, None, None))
        _, kwargs = adapter._bot.do_api_request.call_args
        assert kwargs["api_kwargs"]["rich_message"]["is_rtl"] is True

    def test_not_forwarded_when_english(self):
        adapter = _make_adapter(extra={"rich_rtl": True})
        import asyncio

        asyncio.run(adapter._try_send_rich("123", ENGLISH_CONTENT, None, None))
        _, kwargs = adapter._bot.do_api_request.call_args
        assert "is_rtl" not in kwargs["api_kwargs"]["rich_message"]

    def test_not_forwarded_when_disabled(self):
        adapter = _make_adapter()
        import asyncio

        asyncio.run(adapter._try_send_rich("123", ARABIC_CONTENT, None, None))
        _, kwargs = adapter._bot.do_api_request.call_args
        assert "is_rtl" not in kwargs["api_kwargs"]["rich_message"]
