"""Tests for Telegram MarkdownV2 formatting in gateway/platforms/telegram.py.

Covers: _escape_mdv2 (pure function), format_message (markdown-to-MarkdownV2
conversion pipeline), and edge cases that could produce invalid MarkdownV2
or corrupt user-visible content.
"""

import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Mock the telegram package if it's not installed
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import (  # noqa: E402
    TelegramAdapter,
    _escape_mdv2,
    _strip_mdv2,
    _wrap_markdown_tables,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    return TelegramAdapter(config)


# =========================================================================
# _escape_mdv2
# =========================================================================


class TestEscapeMdv2:
    def test_escapes_all_special_characters(self):
        special = r'_*[]()~`>#+-=|{}.!\ '
        escaped = _escape_mdv2(special)
        # Every special char should be preceded by backslash
        for ch in r'_*[]()~`>#+-=|{}.!\  ':
            if ch == ' ':
                continue
            assert f'\\{ch}' in escaped

    def test_empty_string(self):
        assert _escape_mdv2("") == ""

    def test_no_special_characters(self):
        assert _escape_mdv2("hello world 123") == "hello world 123"

    def test_backslash_escaped(self):
        assert _escape_mdv2("a\\b") == "a\\\\b"

    def test_dot_escaped(self):
        assert _escape_mdv2("v2.0") == "v2\\.0"

    def test_exclamation_escaped(self):
        assert _escape_mdv2("wow!") == "wow\\!"

    def test_mixed_text_and_specials(self):
        result = _escape_mdv2("Hello (world)!")
        assert result == "Hello \\(world\\)\\!"


# =========================================================================
# format_message - basic conversions
# =========================================================================


class TestFormatMessageBasic:
    def test_empty_string(self, adapter):
        assert adapter.format_message("") == ""

    def test_none_input(self, adapter):
        # content is falsy, returned as-is
        assert adapter.format_message(None) is None

    def test_plain_text_specials_escaped(self, adapter):
        result = adapter.format_message("Price is $5.00!")
        assert "\\." in result
        assert "\\!" in result

    def test_plain_text_no_markdown(self, adapter):
        result = adapter.format_message("Hello world")
        assert result == "Hello world"


# =========================================================================
# format_message - code blocks
# =========================================================================


class TestFormatMessageCodeBlocks:
    def test_fenced_code_block_preserved(self, adapter):
        text = "Before\n```python\nprint('hello')\n```\nAfter"
        result = adapter.format_message(text)
        # Code block contents must NOT be escaped
        assert "```python\nprint('hello')\n```" in result
        # But "After" should have no escaping needed (plain text)
        assert "After" in result

    def test_inline_code_preserved(self, adapter):
        text = "Use `my_var` here"
        result = adapter.format_message(text)
        # Inline code content must NOT be escaped
        assert "`my_var`" in result
        # The surrounding text's underscore-free content should be fine
        assert "Use" in result

    def test_code_block_special_chars_not_escaped(self, adapter):
        text = "```\nif (x > 0) { return !x; }\n```"
        result = adapter.format_message(text)
        # Inside code block, > and ! and { should NOT be escaped
        assert "if (x > 0) { return !x; }" in result

    def test_inline_code_special_chars_not_escaped(self, adapter):
        text = "Run `rm -rf ./*` carefully"
        result = adapter.format_message(text)
        assert "`rm -rf ./*`" in result

    def test_multiple_code_blocks(self, adapter):
        text = "```\nblock1\n```\ntext\n```\nblock2\n```"
        result = adapter.format_message(text)
        assert "block1" in result
        assert "block2" in result
        # "text" between blocks should be present
        assert "text" in result

    def test_inline_code_backslashes_escaped(self, adapter):
        r"""Backslashes in inline code must be escaped for MarkdownV2."""
        text = r"Check `C:\ProgramData\VMware\` path"
        result = adapter.format_message(text)
        assert r"`C:\\ProgramData\\VMware\\`" in result

    def test_fenced_code_block_backslashes_escaped(self, adapter):
        r"""Backslashes in fenced code blocks must be escaped for MarkdownV2."""
        text = "```\npath = r'C:\\Users\\test'\n```"
        result = adapter.format_message(text)
        assert r"C:\\Users\\test" in result

    def test_fenced_code_block_backticks_escaped(self, adapter):
        r"""Backticks inside fenced code blocks must be escaped for MarkdownV2."""
        text = "```\necho `hostname`\n```"
        result = adapter.format_message(text)
        assert r"echo \`hostname\`" in result

    def test_inline_code_no_double_escape(self, adapter):
        r"""Already-escaped backslashes should not be quadruple-escaped."""
        text = r"Use `\\server\share`"
        result = adapter.format_message(text)
        # \\ in input → \\\\ in output (each \ escaped once)
        assert r"`\\\\server\\share`" in result


# =========================================================================
# format_message - bold and italic
# =========================================================================


class TestFormatMessageBoldItalic:
    def test_bold_converted(self, adapter):
        result = adapter.format_message("This is **bold** text")
        # MarkdownV2 bold uses single *
        assert "*bold*" in result
        # Original ** should be gone
        assert "**" not in result

    def test_italic_converted(self, adapter):
        result = adapter.format_message("This is *italic* text")
        # MarkdownV2 italic uses _
        assert "_italic_" in result

    def test_bold_with_special_chars(self, adapter):
        result = adapter.format_message("**hello.world!**")
        # Content inside bold should be escaped
        assert "*hello\\.world\\!*" in result

    def test_italic_with_special_chars(self, adapter):
        result = adapter.format_message("*hello.world*")
        assert "_hello\\.world_" in result

    def test_bold_and_italic_in_same_line(self, adapter):
        result = adapter.format_message("**bold** and *italic*")
        assert "*bold*" in result
        assert "_italic_" in result


# =========================================================================
# format_message - headers
# =========================================================================


class TestFormatMessageHeaders:
    def test_h1_converted_to_bold(self, adapter):
        result = adapter.format_message("# Title")
        # Header becomes bold in MarkdownV2
        assert "*Title*" in result
        # Hash should be removed
        assert "#" not in result

    def test_h2_converted(self, adapter):
        result = adapter.format_message("## Subtitle")
        assert "*Subtitle*" in result

    def test_header_with_inner_bold_stripped(self, adapter):
        # Headers strip redundant **...** inside
        result = adapter.format_message("## **Important**")
        # Should be *Important* not ***Important***
        assert "*Important*" in result
        count = result.count("*")
        # Should have exactly 2 asterisks (open + close)
        assert count == 2

    def test_header_with_special_chars(self, adapter):
        result = adapter.format_message("# Hello (World)!")
        assert "\\(" in result
        assert "\\)" in result
        assert "\\!" in result

    def test_multiline_headers(self, adapter):
        text = "# First\nSome text\n## Second"
        result = adapter.format_message(text)
        assert "*First*" in result
        assert "*Second*" in result
        assert "Some text" in result


# =========================================================================
# format_message - links
# =========================================================================


class TestFormatMessageLinks:
    def test_markdown_link_converted(self, adapter):
        result = adapter.format_message("[Click here](https://example.com)")
        assert "[Click here](https://example.com)" in result

    def test_link_display_text_escaped(self, adapter):
        result = adapter.format_message("[Hello!](https://example.com)")
        # The ! in display text should be escaped
        assert "Hello\\!" in result

    def test_link_url_parentheses_escaped(self, adapter):
        result = adapter.format_message("[link](https://example.com/path_(1))")
        # The ) in URL should be escaped
        assert "\\)" in result

    def test_link_with_surrounding_text(self, adapter):
        result = adapter.format_message("Visit [Google](https://google.com) today.")
        assert "[Google](https://google.com)" in result
        assert "today\\." in result


# =========================================================================
# format_message - BUG: italic regex spans newlines
# =========================================================================


class TestItalicNewlineBug:
    r"""Italic regex ``\*([^*]+)\*`` matched across newlines, corrupting content.

    This affects bullet lists using * markers and any text where * appears
    at the end of one line and start of another.
    """

    def test_bullet_list_not_corrupted(self, adapter):
        """Bullet list items using * must NOT be merged into italic."""
        text = "* Item one\n* Item two\n* Item three"
        result = adapter.format_message(text)
        # Each item should appear in the output (not eaten by italic conversion)
        assert "Item one" in result
        assert "Item two" in result
        assert "Item three" in result
        # Should NOT contain _ (italic markers) wrapping list items
        assert "_" not in result or "Item" not in result.split("_")[1] if "_" in result else True

    def test_asterisk_list_items_preserved(self, adapter):
        """Each * list item should remain as a separate line, not become italic."""
        text = "* Alpha\n* Beta"
        result = adapter.format_message(text)
        # Both items must be present in output
        assert "Alpha" in result
        assert "Beta" in result
        # The text between first * and second * must NOT become italic
        lines = result.split("\n")
        assert len(lines) >= 2

    def test_italic_does_not_span_lines(self, adapter):
        """*text on\nmultiple lines* should NOT become italic."""
        text = "Start *across\nlines* end"
        result = adapter.format_message(text)
        # Should NOT have underscore italic markers wrapping cross-line text
        # If this fails, the italic regex is matching across newlines
        assert "_across\nlines_" not in result

    def test_single_line_italic_still_works(self, adapter):
        """Normal single-line italic must still convert correctly."""
        text = "This is *italic* text"
        result = adapter.format_message(text)
        assert "_italic_" in result


# =========================================================================
# format_message - strikethrough
# =========================================================================


class TestFormatMessageStrikethrough:
    def test_strikethrough_converted(self, adapter):
        result = adapter.format_message("This is ~~deleted~~ text")
        assert "~deleted~" in result
        assert "~~" not in result

    def test_strikethrough_with_special_chars(self, adapter):
        result = adapter.format_message("~~hello.world!~~")
        assert "~hello\\.world\\!~" in result

    def test_strikethrough_in_code_not_converted(self, adapter):
        result = adapter.format_message("`~~not struck~~`")
        assert "`~~not struck~~`" in result

    def test_strikethrough_with_bold(self, adapter):
        result = adapter.format_message("**bold** and ~~struck~~")
        assert "*bold*" in result
        assert "~struck~" in result


# =========================================================================
# format_message - spoiler
# =========================================================================


class TestFormatMessageSpoiler:
    def test_spoiler_converted(self, adapter):
        result = adapter.format_message("This is ||hidden|| text")
        assert "||hidden||" in result

    def test_spoiler_with_special_chars(self, adapter):
        result = adapter.format_message("||hello.world!||")
        assert "||hello\\.world\\!||" in result

    def test_spoiler_in_code_not_converted(self, adapter):
        result = adapter.format_message("`||not spoiler||`")
        assert "`||not spoiler||`" in result

    def test_spoiler_pipes_not_escaped(self, adapter):
        """The || delimiters must not be escaped as \\|\\|."""
        result = adapter.format_message("||secret||")
        assert "\\|\\|" not in result
        assert "||secret||" in result


# =========================================================================
# format_message - blockquote
# =========================================================================


class TestFormatMessageBlockquote:
    def test_blockquote_converted(self, adapter):
        result = adapter.format_message("> This is a quote")
        assert "> This is a quote" in result
        # > must NOT be escaped
        assert "\\>" not in result

    def test_blockquote_with_special_chars(self, adapter):
        result = adapter.format_message("> Hello (world)!")
        assert "> Hello \\(world\\)\\!" in result
        assert "\\>" not in result

    def test_blockquote_multiline(self, adapter):
        text = "> Line one\n> Line two"
        result = adapter.format_message(text)
        assert "> Line one" in result
        assert "> Line two" in result
        assert "\\>" not in result

    def test_blockquote_in_code_not_converted(self, adapter):
        result = adapter.format_message("```\n> not a quote\n```")
        assert "> not a quote" in result

    def test_nested_blockquote(self, adapter):
        result = adapter.format_message(">> Nested quote")
        assert ">> Nested quote" in result
        assert "\\>" not in result

    def test_gt_in_middle_of_line_still_escaped(self, adapter):
        """Only > at line start is a blockquote; mid-line > should be escaped."""
        result = adapter.format_message("5 > 3")
        assert "\\>" in result

    def test_expandable_blockquote(self, adapter):
        """Expandable blockquote prefix **> and trailing || must NOT be escaped."""
        result = adapter.format_message("**> Hidden content||")
        assert "**>" in result
        assert "||" in result
        assert "\\*" not in result  # asterisks in prefix must not be escaped
        assert "\\>" not in result  # > in prefix must not be escaped

    def test_single_asterisk_gt_not_blockquote(self, adapter):
        """Single asterisk before > should not be treated as blockquote prefix."""
        result = adapter.format_message("*> not a quote")
        assert "\\*" in result
        assert "\\>" in result

    def test_regular_blockquote_with_pipes_escaped(self, adapter):
        """Regular blockquote ending with || should escape the pipes."""
        result = adapter.format_message("> not expandable||")
        assert "> not expandable" in result
        assert "\\|" in result
        assert "\\>" not in result


# =========================================================================
# format_message - mixed/complex
# =========================================================================


class TestFormatMessageComplex:
    def test_code_block_with_bold_outside(self, adapter):
        text = "**Note:**\n```\ncode here\n```"
        result = adapter.format_message(text)
        assert "*Note:*" in result or "*Note\\:*" in result
        assert "```\ncode here\n```" in result

    def test_bold_inside_code_not_converted(self, adapter):
        """Bold markers inside code blocks should not be converted."""
        text = "```\n**not bold**\n```"
        result = adapter.format_message(text)
        assert "**not bold**" in result

    def test_link_inside_code_not_converted(self, adapter):
        text = "`[not a link](url)`"
        result = adapter.format_message(text)
        assert "`[not a link](url)`" in result

    def test_header_after_code_block(self, adapter):
        text = "```\ncode\n```\n## Title"
        result = adapter.format_message(text)
        assert "*Title*" in result
        assert "```\ncode\n```" in result

    def test_multiple_bold_segments(self, adapter):
        result = adapter.format_message("**a** and **b** and **c**")
        assert result.count("*") >= 6  # 3 bold pairs = 6 asterisks

    def test_special_chars_in_plain_text(self, adapter):
        result = adapter.format_message("Price: $5.00 (50% off!)")
        assert "\\." in result
        assert "\\(" in result
        assert "\\)" in result
        assert "\\!" in result

    def test_empty_bold(self, adapter):
        """**** (empty bold) should not crash."""
        result = adapter.format_message("****")
        assert result is not None

    def test_empty_code_block(self, adapter):
        result = adapter.format_message("```\n```")
        assert "```" in result

    def test_placeholder_collision(self, adapter):
        """Many formatting elements should not cause placeholder collisions."""
        text = (
            "# Header\n"
            "**bold1** *italic1* `code1`\n"
            "**bold2** *italic2* `code2`\n"
            "```\nblock\n```\n"
            "[link](https://url.com)"
        )
        result = adapter.format_message(text)
        # No placeholder tokens should leak into output
        assert "\x00" not in result
        # All elements should be present
        assert "Header" in result
        assert "block" in result
        assert "url.com" in result


# =========================================================================
# _strip_mdv2 — plaintext fallback
# =========================================================================


class TestStripMdv2:
    def test_removes_escape_backslashes(self):
        assert _strip_mdv2(r"hello\.world\!") == "hello.world!"

    def test_removes_bold_markers(self):
        assert _strip_mdv2("*bold text*") == "bold text"

    def test_removes_italic_markers(self):
        assert _strip_mdv2("_italic text_") == "italic text"

    def test_removes_both_bold_and_italic(self):
        result = _strip_mdv2("*bold* and _italic_")
        assert result == "bold and italic"

    def test_preserves_snake_case(self):
        assert _strip_mdv2("my_variable_name") == "my_variable_name"

    def test_preserves_multi_underscore_identifier(self):
        assert _strip_mdv2("some_func_call here") == "some_func_call here"

    def test_plain_text_unchanged(self):
        assert _strip_mdv2("plain text") == "plain text"

    def test_empty_string(self):
        assert _strip_mdv2("") == ""

    def test_removes_strikethrough_markers(self):
        assert _strip_mdv2("~struck text~") == "struck text"

    def test_removes_spoiler_markers(self):
        assert _strip_mdv2("||hidden text||") == "hidden text"


# =========================================================================
# Markdown table auto-wrap
# =========================================================================


class TestWrapMarkdownTables:
    """_wrap_markdown_tables rewrites GFM pipe tables into Telegram-friendly
    row groups instead of leaving noisy pipe syntax in the final message."""

    def test_basic_table_rewritten_as_row_groups(self):
        text = (
            "Scores:\n\n"
            "| Player | Score |\n"
            "|--------|-------|\n"
            "| Alice  | 150   |\n"
            "| Bob    | 120   |\n"
            "\nEnd."
        )
        out = _wrap_markdown_tables(text)
        assert "**Alice**" in out
        assert "• Player: Alice" in out
        assert "• Score: 150" in out
        assert "**Bob**" in out
        assert "• Score: 120" in out
        # Surrounding prose is preserved
        assert out.startswith("Scores:")
        assert out.endswith("End.")

    def test_bare_pipe_table_rewritten(self):
        """Tables without outer pipes (GFM allows this) are still detected."""
        text = "head1 | head2\n--- | ---\na | b\nc | d"
        out = _wrap_markdown_tables(text)
        assert out.startswith("**a**")
        assert "• head1: a" in out
        assert "• head2: b" in out
        assert "**c**" in out

    def test_alignment_separators(self):
        """Separator rows with :--- / ---: / :---: alignment markers match."""
        text = (
            "| Name | Age | City |\n"
            "|:-----|----:|:----:|\n"
            "| Ada  |  30 | NYC  |"
        )
        out = _wrap_markdown_tables(text)
        assert "**Ada**" in out
        assert "• Age: 30" in out
        assert "• City: NYC" in out

    def test_two_consecutive_tables_rewritten_separately(self):
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "\n"
            "| X | Y |\n"
            "|---|---|\n"
            "| 9 | 8 |"
        )
        out = _wrap_markdown_tables(text)
        assert out.count("**1**") == 1
        assert out.count("**9**") == 1
        assert "• A: 1" in out
        assert "• X: 9" in out

    def test_plain_text_with_pipes_not_wrapped(self):
        """A bare pipe in prose must NOT trigger wrapping."""
        text = "Use the | pipe operator to chain commands."
        assert _wrap_markdown_tables(text) == text

    def test_horizontal_rule_not_wrapped(self):
        """A lone '---' horizontal rule must not be mistaken for a separator."""
        text = "Section A\n\n---\n\nSection B"
        assert _wrap_markdown_tables(text) == text

    def test_existing_code_block_with_pipes_left_alone(self):
        """A table already inside a fenced code block must not be re-wrapped."""
        text = (
            "```\n"
            "| a | b |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "```"
        )
        assert _wrap_markdown_tables(text) == text

    def test_no_pipe_character_short_circuits(self):
        text = "Plain **bold** text with no table."
        assert _wrap_markdown_tables(text) == text

    def test_no_dash_short_circuits(self):
        text = "a | b\nc | d"  # has pipes but no '-' separator row
        assert _wrap_markdown_tables(text) == text

    def test_single_column_separator_not_matched(self):
        """Single-column tables (rare) are not detected — we require at
        least one internal pipe in the separator row to avoid false
        positives on formatting rules."""
        text = "| a |\n| - |\n| b |"
        assert _wrap_markdown_tables(text) == text


class TestFormatMessageTables:
    """End-to-end: pipe tables become readable Telegram-native text instead
    of escaped pipe syntax or fenced code blocks."""

    def test_table_rendered_as_bullets(self, adapter):
        text = (
            "Data:\n\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| A    | B    |\n"
        )
        out = adapter.format_message(text)
        assert "*A*" in out
        assert "• Col1: A" in out
        assert "• Col2: B" in out
        assert "```" not in out
        assert "\\|" not in out

    def test_text_after_table_still_formatted(self, adapter):
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "\n"
            "Nice **work** team!"
        )
        out = adapter.format_message(text)
        # MarkdownV2 bold conversion still happens outside the table
        assert "*work*" in out
        # Exclamation outside fence is escaped
        assert "\\!" in out
        assert "*1*" in out
        assert "• A: 1" in out

    def test_multiple_tables_in_single_message(self, adapter):
        text = (
            "First:\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "\n"
            "Second:\n"
            "| X | Y |\n"
            "|---|---|\n"
            "| 9 | 8 |\n"
        )
        out = adapter.format_message(text)
        assert out.count("*1*") == 1
        assert out.count("*9*") == 1
        assert "• X: 9" in out


@pytest.mark.asyncio
async def test_send_escapes_chunk_indicator_for_markdownv2(adapter):
    adapter.MAX_MESSAGE_LENGTH = 80
    adapter._bot = MagicMock()

    sent_texts = []

    async def _fake_send_message(**kwargs):
        sent_texts.append(kwargs["text"])
        msg = MagicMock()
        msg.message_id = len(sent_texts)
        return msg

    adapter._bot.send_message = AsyncMock(side_effect=_fake_send_message)

    content = ("**bold** chunk content " * 12).strip()
    result = await adapter.send("123", content)

    assert result.success is True
    assert len(sent_texts) > 1
    assert re.search(r" \\\([0-9]+/[0-9]+\\\)$", sent_texts[0])
    assert re.search(r" \\\([0-9]+/[0-9]+\\\)$", sent_texts[-1])


# =========================================================================
# edit_message — streaming Markdown safety
# =========================================================================


class TestEditMessageStreamingSafety:
    @pytest.mark.asyncio
    async def test_non_final_edit_uses_plain_text_without_markdown(self):
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
        adapter._bot = MagicMock()
        adapter._bot.edit_message_text = AsyncMock()

        result = await adapter.edit_message("123", "456", "partial **bold", finalize=False)

        assert result.success is True
        adapter._bot.edit_message_text.assert_awaited_once_with(
            chat_id=123,
            message_id=456,
            text="partial **bold",
        )

    @pytest.mark.asyncio
    async def test_final_edit_uses_markdownv2_with_plain_fallback(self):
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
        adapter._bot = MagicMock()
        adapter._bot.edit_message_text = AsyncMock(side_effect=[Exception("bad markdown"), None])

        result = await adapter.edit_message("123", "456", "final **bold**", finalize=True)

        assert result.success is True
        first_call = adapter._bot.edit_message_text.await_args_list[0].kwargs
        second_call = adapter._bot.edit_message_text.await_args_list[1].kwargs
        assert "parse_mode" in first_call
        assert first_call["text"] == "final *bold*"
        assert second_call == {
            "chat_id": 123,
            "message_id": 456,
            "text": "final **bold**",
        }

# =========================================================================
# Telegram guest mention gating
# =========================================================================


def _guest_test_adapter(*, guest_mode=True, require_mention=True, allowed_chats=None):
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "guest_mode": guest_mode,
            "require_mention": require_mention,
            "allowed_chats": allowed_chats or ["-100200"],
        },
    )
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._mention_patterns = adapter._compile_mention_patterns()
    return adapter


def _guest_group_message(text, *, chat_id=-100201, entities=None, reply_to_bot=False):
    reply_to_message = SimpleNamespace(from_user=SimpleNamespace(id=999)) if reply_to_bot else None
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=entities or [],
        caption_entities=[],
        message_thread_id=None,
        chat=SimpleNamespace(id=chat_id, type="group"),
        from_user=SimpleNamespace(id=111),
        reply_to_message=reply_to_message,
    )


def _guest_mention_entity(text, mention="@hermes_bot"):
    return SimpleNamespace(type="mention", offset=text.index(mention), length=len(mention))


class TestTelegramGuestMentionGating:
    def test_guest_mode_allows_explicit_mention_outside_allowed_chats(self):
        adapter = _guest_test_adapter(guest_mode=True, allowed_chats=["-100200"])
        text = "please help @hermes_bot"
        message = _guest_group_message(
            text,
            chat_id=-100201,
            entities=[_guest_mention_entity(text)],
        )

        assert adapter._should_process_message(message) is True

    def test_guest_mode_does_not_allow_reply_outside_allowed_chats(self):
        adapter = _guest_test_adapter(guest_mode=True, allowed_chats=["-100200"])
        message = _guest_group_message("replying without mention", chat_id=-100201, reply_to_bot=True)

        assert adapter._should_process_message(message) is False

    def test_guest_mode_disabled_keeps_allowed_chats_as_hard_gate_for_mentions(self):
        adapter = _guest_test_adapter(guest_mode=False, allowed_chats=["-100200"])
        text = "please help @hermes_bot"
        message = _guest_group_message(
            text,
            chat_id=-100201,
            entities=[_guest_mention_entity(text)],
        )

        assert adapter._should_process_message(message) is False

    def test_guest_mode_allows_bot_command_entity_outside_allowed_chats(self):
        """``/cmd@botname`` is a ``bot_command`` entity, not ``mention``."""
        adapter = _guest_test_adapter(guest_mode=True, allowed_chats=["-100200"])
        text = "/status@hermes_bot"
        message = _guest_group_message(
            text,
            chat_id=-100201,
            entities=[SimpleNamespace(type="bot_command", offset=0, length=len(text))],
        )

        assert adapter._should_process_message(message) is True

    def test_guest_mode_allows_text_mention_entity_outside_allowed_chats(self):
        """MessageEntity(type=text_mention) tags a user by ID — recognised as mention."""
        adapter = _guest_test_adapter(guest_mode=True, allowed_chats=["-100200"])
        message = _guest_group_message(
            "hey there",
            chat_id=-100201,
            entities=[SimpleNamespace(type="text_mention", offset=0, length=3, user=SimpleNamespace(id=999))],
        )

        assert adapter._should_process_message(message) is True

    def test_guest_mode_allows_mention_in_caption_outside_allowed_chats(self):
        """Media caption @mention should bypass allowed_chats via guest_mode."""
        adapter = _guest_test_adapter(guest_mode=True, allowed_chats=["-100200"])
        text = "look @hermes_bot"
        message = _guest_group_message(
            text="",
            chat_id=-100201,
            entities=[],
        )
        message.caption = text
        message.caption_entities = [_guest_mention_entity(text)]

        assert adapter._should_process_message(message) is True
