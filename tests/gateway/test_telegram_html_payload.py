"""Live-lane HTML formatting: Bot API parse_mode=HTML for pre-rendered HTML payloads."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.constants import ParseMode

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _looks_like_html_payload,
    _strip_html_for_plain,
)


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=77))
    return adapter


class TestDetect:
    def test_fleet_alert_html(self):
        assert _looks_like_html_payload("🛡 <b>Fleet Policy</b> · <i>08.09</i>")

    def test_blockquote_html(self):
        assert _looks_like_html_payload("<blockquote>• текст</blockquote>")

    def test_plain_markdown_not_html(self):
        assert not _looks_like_html_payload("**bold** and _italic_ and `code`")

    def test_html_shown_inside_code_span_is_not_payload(self):
        assert not _looks_like_html_payload("Пример: `<b>не форматирует</b>` — просто текст")

    def test_html_shown_inside_fence_is_not_payload(self):
        assert not _looks_like_html_payload("```html\n<b>escaped sample</b>\n```")

    def test_random_angle_bracket_not_html(self):
        assert not _looks_like_html_payload("a < b and c > d, <3")

    def test_strip_html_for_plain(self):
        assert _strip_html_for_plain("<b>4</b> ждут &amp; решения").strip() == "4 ждут & решения"


class TestSendLane:
    @pytest.mark.asyncio
    async def test_html_payload_sent_with_parse_mode_html(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        text = "🛡 <b>Fleet Policy</b> · <i>08.09 14:30</i>\n🔴 Одобрения: <b>4</b> ждут решения"
        result = await adapter.send("1256122537", text)
        assert result.success is True
        kwargs = adapter._bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == ParseMode.HTML
        assert kwargs["text"] == text  # untouched — no MarkdownV2 escaping

    @pytest.mark.asyncio
    async def test_html_payload_skips_rich_markdown_path(self):
        adapter = _make_adapter()
        # rich enabled: HTML must NOT go through sendRichMessage (markdown renderer)
        adapter._bot.send_rich_message = AsyncMock(return_value=MagicMock(message_id=78))
        text = "<b>bold</b> payload"
        result = await adapter.send("1256122537", text)
        assert result.success is True
        kwargs = adapter._bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == ParseMode.HTML

    @pytest.mark.asyncio
    async def test_html_parse_error_falls_back_to_plain(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        err = Exception("Can't parse entities: unexpected end tag")
        adapter._bot.send_message = AsyncMock(side_effect=[err, MagicMock(message_id=79)])
        result = await adapter.send("1256122537", "<b>unclosed payload")
        assert result.success is True
        second = adapter._bot.send_message.await_args_list[1].kwargs
        assert second["parse_mode"] is None
        assert "<b>" not in second["text"]

    @pytest.mark.asyncio
    async def test_markdown_reply_still_markdownv2(self):
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        result = await adapter.send("1256122537", "**bold** reply")
        assert result.success is True
        kwargs = adapter._bot.send_message.await_args.kwargs
        assert kwargs["parse_mode"] == ParseMode.MARKDOWN_V2

    @pytest.mark.asyncio
    async def test_oversized_html_splits_and_sends_all_chunks(self):
        """A >4096 HTML payload must reach Telegram as multiple parse_mode=HTML chunks
        (never one over-long send that Telegram rejects)."""
        adapter = _make_adapter()
        adapter._rich_send_disabled = True
        body = "\n".join(f"<b>Строка {i}</b> · данные" for i in range(400))
        text = f"<blockquote>{body}</blockquote>"
        result = await adapter.send("1256122537", text)
        assert result.success is True
        calls = adapter._bot.send_message.await_args_list
        assert len(calls) > 1
        assert all(c.kwargs["parse_mode"] == ParseMode.HTML for c in calls)


class TestChunkHtml:
    def test_short_text_single_chunk(self):
        from plugins.platforms.telegram.adapter import _chunk_html_payload, utf16_len
        assert _chunk_html_payload("<b>ok</b>", 4096, utf16_len) == ["<b>ok</b>"]

    def test_line_boundaries_never_mid_tag(self):
        from plugins.platforms.telegram.adapter import _chunk_html_payload, utf16_len, _strip_html_for_plain
        body = "\n".join(f"<b>Строка {i}</b> · данные" for i in range(400))
        text = f"<blockquote>{body}</blockquote>"
        chunks = _chunk_html_payload(text, 4096, utf16_len)
        assert len(chunks) > 1
        for ch in chunks:
            assert utf16_len(ch) <= 4096
            assert _balanced(ch)  # every chunk is a standalone well-formed doc
        # visible text preserved (boundary newlines between messages may be dropped)
        joined = _strip_html_for_plain("".join(chunks)).replace("\n", "")
        assert joined == _strip_html_for_plain(text).replace("\n", "")

    def test_single_huge_line_splits_and_preserves_visible_text(self):
        from plugins.platforms.telegram.adapter import _chunk_html_payload, utf16_len, _strip_html_for_plain
        text = ("<b>x</b> " * 900).strip()  # one line, >4096 utf-16
        chunks = _chunk_html_payload(text, 4096, utf16_len)
        assert len(chunks) > 1
        assert all(utf16_len(c) <= 4096 for c in chunks)
        assert all(_balanced(c) for c in chunks)
        assert _strip_html_for_plain("".join(chunks)) == _strip_html_for_plain(text)


def _balanced(html_text: str) -> bool:
    """True when every opened tag is closed (ignoring void tags) — a well-formed Bot API HTML doc."""
    import re
    stack = []
    for m in re.finditer(r'<(/?)([a-zA-Z][a-zA-Z0-9-]*)([^>]*)>', html_text):
        closing, name = m.group(1), m.group(2).lower()
        if name in ("br",):
            continue
        if closing:
            if not stack or stack[-1] != name:
                return False
            stack.pop()
        else:
            stack.append(name)
    return not stack
