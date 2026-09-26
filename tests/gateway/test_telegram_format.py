"""Tests for Telegram MarkdownV2 formatting in gateway/platforms/telegram.py.

Covers: _escape_mdv2 (pure function), format_message (markdown-to-MarkdownV2
conversion pipeline), and edge cases that could produce invalid MarkdownV2
or corrupt user-visible content.
"""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Mock the telegram package if it's not installed
# ---------------------------------------------------------------------------
from plugins.platforms.telegram.adapter import (  # noqa: E402
    TelegramAdapter,
    _escape_mdv2,
    _strip_mdv2,
    _wrap_markdown_tables,
)
from gateway.platforms.helpers import normalize_latex_math_symbols, strip_markdown


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






# =========================================================================
# format_message - basic conversions
# =========================================================================


class TestFormatMessageBasic:


    def test_plain_text_specials_escaped(self, adapter):
        result = adapter.format_message("Price is $5.00!")
        assert "\\." in result
        assert "\\!" in result


# =========================================================================
# format_message - code blocks
# =========================================================================


class TestFormatMessageCodeBlocks:


    def test_code_block_special_chars_not_escaped(self, adapter):
        text = "```\nif (x > 0) { return !x; }\n```"
        result = adapter.format_message(text)
        # Inside code block, > and ! and { should NOT be escaped
        assert "if (x > 0) { return !x; }" in result

    def test_inline_code_special_chars_not_escaped(self, adapter):
        text = "Run `rm -rf ./*` carefully"
        result = adapter.format_message(text)
        assert "`rm -rf ./*`" in result


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


@pytest.mark.asyncio
async def test_final_send_does_not_retrigger_typing(adapter):
    """The final reply (metadata['notify']) must NOT re-arm Telegram's typing
    timer. The gateway has already torn down the refresh loop by then, so a
    re-trigger here would leave the '...typing' bubble lingering after the
    answer (Telegram has no stop-typing API). See #48678."""
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    adapter._bot.send_chat_action = AsyncMock()
    adapter._rich_messages_enabled = False

    result = await adapter.send("12345", "All done.", metadata={"notify": True})

    assert result.success is True
    adapter._bot.send_chat_action.assert_not_called()


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


    def test_reload_mcp_summary_escapes_dynamic_server_names(self, adapter):
        content = (
            "🔄 **MCP Servers Reloaded**\n"
            "♻️ Reconnected: agent_one, tool[beta]\n"
            "➕ Added: alpha*prod\n"
            "🔧 3 tool(s) available from 2 server(s)"
        )
        result = adapter.format_message(content)
        assert "*MCP Servers Reloaded*" in result
        assert "agent\\_one" in result
        assert "tool\\[beta\\]" in result
        assert "alpha\\*prod" in result


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


    def test_header_with_inner_bold_stripped(self, adapter):
        # Headers strip redundant **...** inside
        result = adapter.format_message("## **Important**")
        # Should be *Important* not ***Important***
        assert "*Important*" in result
        count = result.count("*")
        # Should have exactly 2 asterisks (open + close)
        assert count == 2


# =========================================================================
# format_message - links
# =========================================================================


class TestFormatMessageLinks:

    def test_link_display_text_escaped(self, adapter):
        result = adapter.format_message("[Hello!](https://example.com)")
        # The ! in display text should be escaped
        assert "Hello\\!" in result

    def test_link_url_parentheses_escaped(self, adapter):
        result = adapter.format_message("[link](https://example.com/path_(1))")
        # The ) in URL should be escaped
        assert "\\)" in result


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


# =========================================================================
# format_message - strikethrough
# =========================================================================


class TestFormatMessageStrikethrough:
    def test_strikethrough_converted(self, adapter):
        result = adapter.format_message("This is ~~deleted~~ text")
        assert "~deleted~" in result
        assert "~~" not in result


# =========================================================================
# format_message - spoiler
# =========================================================================


class TestFormatMessageSpoiler:


    def test_spoiler_pipes_not_escaped(self, adapter):
        """The || delimiters must not be escaped as \\|\\|."""
        result = adapter.format_message("||secret||")
        assert "\\|\\|" not in result
        assert "||secret||" in result


# =========================================================================
# format_message - blockquote
# =========================================================================


class TestFormatMessageBlockquote:


    def test_blockquote_multiline(self, adapter):
        text = "> Line one\n> Line two"
        result = adapter.format_message(text)
        assert "> Line one" in result
        assert "> Line two" in result
        assert "\\>" not in result


    def test_gt_in_middle_of_line_still_escaped(self, adapter):
        """Only > at line start is a blockquote; mid-line > should be escaped."""
        result = adapter.format_message("5 > 3")
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


    def test_empty_bold(self, adapter):
        """**** (empty bold) should not crash."""
        result = adapter.format_message("****")
        assert result is not None


# =========================================================================
# _strip_mdv2 — plaintext fallback
# =========================================================================


class TestStripMdv2:
    def test_removes_escape_backslashes(self):
        assert _strip_mdv2(r"hello\.world\!") == "hello.world!"


    def test_removes_both_bold_and_italic(self):
        result = _strip_mdv2("*bold* and _italic_")
        assert result == "bold and italic"

    def test_preserves_snake_case(self):
        assert _strip_mdv2("my_variable_name") == "my_variable_name"




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
        # The heading IS the Player cell — don't repeat it as a bullet.
        assert "• Player: Alice" not in out
        assert "• Score: 150" in out
        assert "**Bob**" in out
        assert "• Score: 120" in out
        # Heading and its bullet sit on consecutive lines (no blank between).
        assert "**Alice**\n• Score: 150" in out
        # Separate row groups ARE separated by a blank line.
        assert "• Score: 150\n\n**Bob**" in out
        # Surrounding prose is preserved
        assert out.startswith("Scores:")
        assert out.endswith("End.")



    def test_no_pipe_character_short_circuits(self):
        text = "Plain **bold** text with no table."
        assert _wrap_markdown_tables(text) == text


    def test_row_group_uses_single_newlines_within_group(self):
        """Regression: each bullet within a row-group must be separated by
        a single newline, not a blank line.  Telegram renders blank lines
        as paragraph breaks, which previously left every bullet floating in
        its own paragraph and made multi-column tables unreadable.

        Mirrors the exact pattern that produced the screenshot bug report:
        a five-column comparison table with no row-label column.
        """
        text = (
            "| Play | Capital | Build | $/day | Risk |\n"
            "|---|---|---|---|---|\n"
            "| A. Copy Hands (HK/SZ) | $5-10k | 2 wk | $30-70 | Low |\n"
            "| B. NO-sweeper        | $50-100k | 3 wk | $300-1000 | Med |"
        )
        out = _wrap_markdown_tables(text)

        # No bullet sits inside its own paragraph: the substring "\n\n• "
        # would mean a blank line precedes a bullet, which is the bug.
        assert "\n\n• " not in out

        # The two row-groups DO have a paragraph break between them.
        groups = [g for g in out.split("\n\n") if g.strip()]
        assert len(groups) == 2
        # Heading + 4 bullets per group means each group is exactly 5 lines.
        for group in groups:
            line_count = group.count("\n") + 1
            assert line_count == 5, (
                "Each row-group should be 5 lines (heading + 4 bullets), "
                f"got {line_count}:\n{group}"
            )


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
        # Heading 'A' duplicates the Col1 value — skip that bullet.
        assert "• Col1: A" not in out
        assert "• Col2: B" in out
        assert "```" not in out
        assert "\\|" not in out


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
    async def test_message_too_long_splits_into_continuations_not_silent_truncation(self):
        """When edit_message_text exceeds Telegram's 4096 UTF-16 limit on
        finalize, the adapter must split the content across the existing
        message + new continuation messages so the user gets the full reply.
        Previously the adapter best-effort truncated the content with '…' and
        returned success=True, dropping everything past the truncation
        boundary (#19537)."""
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
        adapter._bot = MagicMock()
        adapter._bot.edit_message_text = AsyncMock()
        # Continuation sends return monotonically increasing message ids.
        _next_id = [1000]
        async def _fake_send(**kwargs):
            _next_id[0] += 1
            return SimpleNamespace(message_id=_next_id[0])
        adapter._bot.send_message = AsyncMock(side_effect=_fake_send)

        # 6000-char content well over the 4096 UTF-16 limit.
        oversized = "x" * 6000
        result = await adapter.edit_message("123", "456", oversized, finalize=True)

        # Adapter reports success with continuations populated.
        assert result.success is True
        assert result.error is None
        assert len(result.continuation_message_ids) >= 1, (
            "expected at least one continuation message"
        )
        # The reported message_id is the LAST visible message (the final
        # continuation), so subsequent edits target the most recent.
        assert result.message_id == result.continuation_message_ids[-1]
        # Original message_id (456) was edited with chunk 1.
        first_edit = adapter._bot.edit_message_text.call_args
        assert first_edit.kwargs["message_id"] == 456
        # Continuations were sent threaded as replies for visual grouping.
        assert adapter._bot.send_message.await_count == len(result.continuation_message_ids)


    @pytest.mark.asyncio
    async def test_mid_stream_overflow_truncates_instead_of_splitting(self):
        """During streaming (finalize=False), oversized content must be
        truncated to fit one message rather than split into continuations.
        Splitting mid-stream creates new message IDs that the stream consumer
        then edits with the full accumulated text, causing an infinite
        duplication loop (#48648)."""
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
        adapter._bot = MagicMock()
        adapter._bot.edit_message_text = AsyncMock()
        _next_id = [1000]

        async def _fake_send(**kwargs):
            _next_id[0] += 1
            return SimpleNamespace(message_id=_next_id[0])

        adapter._bot.send_message = AsyncMock(side_effect=_fake_send)

        oversized = "x" * 6000
        result = await adapter.edit_message("123", "456", oversized, finalize=False)

        # Must NOT create continuation messages during streaming.
        assert result.success is True
        assert adapter._bot.send_message.await_count == 0, (
            "mid-stream overflow must not send continuation messages"
        )
        # message_id stays the original — no shift to a new ID.
        assert result.message_id == "456"
        # The edit must contain truncated content (≤ MAX_MESSAGE_LENGTH).
        edited_text = adapter._bot.edit_message_text.call_args.kwargs["text"]
        assert len(edited_text) <= adapter.MAX_MESSAGE_LENGTH

    @pytest.mark.asyncio
    async def test_saturated_preview_dedups_repeat_oversized_edits(self):
        """Once a mid-stream preview saturates at the truncation cap, further
        oversized edits truncate to the SAME text — re-sending them is a
        visual no-op that burns Telegram's flood budget (~1 edit/0.8s for the
        rest of a long stream ⇒ flood control with 200s+ penalties, hanging
        final delivery). The adapter must skip identical saturated previews
        without an API call."""
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
        adapter._bot = MagicMock()
        adapter._bot.edit_message_text = AsyncMock()
        adapter._bot.send_message = AsyncMock()

        # First oversized edit: delivers the truncated preview (1 API call).
        r1 = await adapter.edit_message("123", "456", "x" * 6000, finalize=False)
        assert r1.success is True
        assert adapter._bot.edit_message_text.await_count == 1

        # Stream keeps growing within the same chunk count: previews truncate
        # identically — no API calls. (7000 and 8000 chars both truncate to
        # the same "…(1/2)" preview; 9000 crosses into "(1/3)" — a real change
        # that SHOULD be delivered, at most one edit per ~4096 chars of growth
        # instead of one per 0.8s tick.)
        for grow in (7000, 8000):
            r = await adapter.edit_message("123", "456", "x" * grow, finalize=False)
            assert r.success is True
            assert r.message_id == "456"
        assert adapter._bot.edit_message_text.await_count == 1, (
            "identical saturated previews must not be re-sent"
        )
        # Chunk-count boundary: marker changes (1/2 → 1/3) — one real edit.
        await adapter.edit_message("123", "456", "x" * 9000, finalize=False)
        assert adapter._bot.edit_message_text.await_count == 2
        # ...and saturates again at the new marker.
        await adapter.edit_message("123", "456", "x" * 9100, finalize=False)
        assert adapter._bot.edit_message_text.await_count == 2

        # A DIFFERENT oversized prefix (content changed within the first 4096)
        # must still go through.
        await adapter.edit_message("123", "456", "y" * 9100, finalize=False)
        assert adapter._bot.edit_message_text.await_count == 3


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
    # PR db50af910 added a TELEGRAM_ALLOWED_USERS allowlist gate to
    # _should_process_message. These tests aren't exercising the auth
    # gate — they're exercising the guest-mode mention/allowed_chats
    # logic that runs after — so stub the user authz to always allow.
    adapter._is_callback_user_authorized = lambda *_a, **_kw: True
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


# =========================================================================
# LaTeX Math Normalization Tests (BDD 4.1.1 - 4.1.3)
# =========================================================================


class TestLatexNormalization:
    """Covers LaTeX math syntax normalization into clean Unicode symbols."""

    def test_latex_inequalities_converted_to_unicode(self, adapter):
        inp = "Возраст пакета $\\ge 180$ дней, скачивания $\\ge 50$k"
        out = adapter.format_message(inp)
        assert "≥ 180" in out
        assert "≥ 50k" in out
        assert "\\ge" not in out
        assert "$" not in out

    def test_latex_macros_standalone(self, adapter):
        assert "x ≤ 10" in adapter.format_message("x \\le 10")
        assert "t ≈ 5s" in adapter.format_message("t \\approx 5s")
        assert "a ≠ b" in adapter.format_message("a \\neq b")
        assert "± 5%" in adapter.format_message("\\pm 5%")
        assert "10 × 20" in adapter.format_message("10 \\times 20")
        assert "a → b" in adapter.format_message("a \\to b")
        assert "a ← b" in adapter.format_message("a \\leftarrow b")
        assert "∞" in adapter.format_message("\\infty")
        assert "…" in adapter.format_message("\\dots")
        assert "·" in adapter.format_message("\\cdot")

    def test_latex_inside_fenced_code_block_preserved(self, adapter):
        text = "```python\npattern = r\"\\ge\\d+\"\n```"
        assert normalize_latex_math_symbols(text) == text
        out = adapter.format_message(text)
        assert "≥" not in out
        assert "\\\\ge" in out

    def test_latex_inside_inline_code_preserved(self, adapter):
        text = "Run `\\ge 180` in bash"
        assert normalize_latex_math_symbols(text) == text
        out = adapter.format_message(text)
        assert "≥" not in out
        assert "\\\\ge" in out

    def test_math_delimiters_stripped(self, adapter):
        assert normalize_latex_math_symbols("$\\ge 180$") == "≥ 180"
        assert normalize_latex_math_symbols("$$\\approx 100$$") == "≈ 100"
        assert normalize_latex_math_symbols("$x = 10$") == "x = 10"
        assert normalize_latex_math_symbols("$N$") == "N"
        assert "≥ 180" in adapter.format_message("$\\ge 180$")
        assert "≈ 100" in adapter.format_message("$$\\approx 100$$")
        assert "x \\= 10" in adapter.format_message("$x = 10$")
        assert "N" in adapter.format_message("$N$")

    def test_greek_letters_converted(self, adapter):
        text = "\\alpha, \\beta, \\gamma, \\Delta, \\pi, \\sigma, \\mu, \\theta, \\lambda, \\omega, \\Omega"
        out = adapter.format_message(text)
        assert "α, β, γ, Δ, π, σ, μ, θ, λ, ω, Ω" in out

    def test_degree_converted(self, adapter):
        assert "25°" in adapter.format_message("25\\degree")
        assert "25°" in adapter.format_message("25^\\circ")
        assert "25°" in adapter.format_message("25^{\\circ}")

    def test_currency_dollars_preserved(self, adapter):
        out = adapter.format_message("Price is $5.00!")
        assert "$5" in out
        out2 = adapter.format_message("From $5 to $10 per unit")
        assert "$5" in out2 and "$10" in out2


# =========================================================================
# Cascading Fallback Cleanliness Tests (BDD 4.5.1)
# =========================================================================


class TestCascadingFallbackCleanliness:
    """Covers Tier 3 plain text fallback cleanliness (zero escape backslashes)."""

    def test_fallback_plain_text_has_no_escape_backslashes(self):
        escaped = r"This is \*bold\* and \_italic\_ with snake\_case and v2\.0 and wow\!"
        plain = _strip_mdv2(escaped)
        assert "\\" not in plain
        assert plain == "This is bold and italic with snake_case and v2.0 and wow!"

    def test_fallback_preserves_readable_structure(self):
        text = "# Header\n\n* Item 1\n* Item 2\n\n[Link](https://example.com)\n\n```python\nprint('hello')\n```"
        plain = strip_markdown(text)
        assert "\\" not in plain
        assert "#" not in plain
        assert "Item 1" in plain
        assert "Item 2" in plain
        assert "Link" in plain
        assert "https://example.com" not in plain


# =========================================================================
# Special Characters and Complex Formatting (BDD 4.2.2, 4.3.2, 4.4.1)
# =========================================================================


class TestSpecialCharactersAndTables:
    """Covers special character escaping, angle brackets, and table formatting."""

    def test_snake_case_outside_code_escaped(self, adapter):
        out = adapter.format_message("Check file_name_v2_final.py and user_account_id")
        assert "file\\_name\\_v2\\_final\\.py" in out
        assert "user\\_account\\_id" in out

    def test_angle_brackets_outside_code(self, adapter):
        out = adapter.format_message("if x < 10 and y > 20:")
        assert "x < 10" in out or "x \\< 10" in out
        assert "y \\> 20" in out

    def test_table_with_latex_math(self, adapter):
        table = (
            "| Metric | Threshold | Status |\n"
            "|---|---|---|\n"
            "| Delay | $\\le 100$ms | OK |\n"
            "| Count | $\\ge 50$ | OK |\n"
        )
        out = adapter.format_message(table)
        assert "≤ 100ms" in out
        assert "≥ 50" in out
        assert "\\le" not in out
        assert "\\ge" not in out
        assert "$" not in out


class TestBddScenarios20260925:
    """Canonical test suite for the 6 SpDD BDD Scenarios (task_20260925_telegram_formatting_fix)."""

    def test_scenario_1_empty_backticks_and_multiline_boundary(self, adapter):
        inp = (
            "3. **Reasoning скрыт (`show_reasoning: false`):** Поскольку блок мыслей `` вырезается, а модель 14B...\n"
            "- Параметр 1: `val`\n"
            "- Параметр 2: `val2`\n"
        )
        out = adapter.format_message(inp)
        assert "\\`\\`" in out or "``" not in out
        assert "`show_reasoning: false`" in out
        assert "`val`" in out
        assert "`val2`" in out
        assert "•" in out

    def test_scenario_2_zero_bullet_eating_strip_markdown(self):
        inp = (
            "* Скомпилирован байткод всех пропатченных модулей\n"
            "* Запущен локальный E2E-тест конвертации\n"
            "* Перезапущен сервис hermes-gateway-andrey.service"
        )
        out = strip_markdown(inp)
        expected = inp.strip()
        assert out == expected
        assert out.count("*") == 3

    def test_scenario_3_code_symbols_protection_strip_markdown(self):
        inp = "Маркеры `*`, `-`, `+` в начале строк переводятся в нативные символы `•` и `◦`"
        out = strip_markdown(inp)
        assert out == "Маркеры *, -, + в начале строк переводятся в нативные символы • и ◦"

    @pytest.mark.asyncio
    async def test_scenario_4_fallback_crash_free_plain_text(self, adapter):
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(
            side_effect=[
                Exception("400 Bad Request: Can't parse entities in message text"),
                SimpleNamespace(message_id=428774),
            ]
        )
        send_kwargs = {"chat_id": 12345678}
        raw = "**Important:** Check `*`, `-`, `+` list markers."
        formatted = adapter.format_message(raw)
        res = await adapter._send_chunk_markdown_or_plain(formatted, send_kwargs, original_raw_chunk=raw)
        assert res.message_id == 428774
        assert adapter._bot.send_message.call_count == 2
        second_call = adapter._bot.send_message.call_args_list[1]
        plain_text = second_call.kwargs.get("text")
        assert "\\" not in plain_text
        assert "Important: Check *, -, + list markers." in plain_text
        assert second_call.kwargs.get("parse_mode") is None

    def test_scenario_5_latex_math_to_unicode(self):
        inp = r"Значение $x \ge 180$, погрешность $\approx 5\%$, дробь $\frac{a}{b}$, угол $90^\circ$, предел $\alpha \to \infty$."
        out = normalize_latex_math_symbols(inp)
        assert "x ≥ 180" in out
        assert "≈ 5%" in out
        assert "a/b" in out
        assert "90°" in out
        assert "α → ∞" in out
        assert r"\ge" not in out
        assert r"\approx" not in out
        assert r"\frac" not in out
        assert r"\alpha" not in out
        assert r"\to" not in out
        assert r"\infty" not in out

    @pytest.mark.asyncio
    async def test_scenario_6_streaming_incomplete_markdown_edit(self, adapter):
        adapter._bot = MagicMock()
        adapter._bot.edit_message_text = AsyncMock(
            side_effect=[
                Exception("400 Bad Request: Can't parse entities: can't find end of code entity"),
                None,
            ]
        )
        partial = "Here is the code:\n```python\ndef run():\n    return 42"
        res = await adapter.edit_message("12345678", "100", partial, finalize=True)
        assert res.success is True
        assert adapter._bot.edit_message_text.call_count == 2



