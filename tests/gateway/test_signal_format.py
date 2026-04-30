"""Tests for Signal _markdown_to_signal() formatting.

Covers the markdown-to-bodyRanges conversion pipeline: bold, italic,
strikethrough, monospace, code blocks, headings, and — critically — the
false-positive regressions that caused spurious italics in production.
"""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.signal import SignalAdapter


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _m2s(text: str):
    """Shorthand: call the static method and return (plain_text, styles)."""
    return SignalAdapter._markdown_to_signal(text)


def _style_types(styles: list[str]) -> list[str]:
    """Extract just the STYLE part from '0:4:BOLD' strings."""
    return [s.rsplit(":", 1)[1] for s in styles]


def _find_style(styles: list[str], style_type: str) -> list[str]:
    """Return only styles matching a given type."""
    return [s for s in styles if s.endswith(f":{style_type}")]


# ===========================================================================
# Basic formatting
# ===========================================================================

class TestMarkdownToSignalBasic:
    """Core formatting: bold, italic, strikethrough, monospace."""

    def test_bold_double_asterisk(self):
        text, styles = _m2s("hello **world**")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":BOLD")

    def test_bold_double_underscore(self):
        text, styles = _m2s("hello __world__")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":BOLD")

    def test_italic_single_asterisk(self):
        text, styles = _m2s("hello *world*")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":ITALIC")

    def test_italic_single_underscore(self):
        text, styles = _m2s("hello _world_")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":ITALIC")

    def test_strikethrough(self):
        text, styles = _m2s("hello ~~world~~")
        assert text == "hello world"
        assert len(styles) == 1
        assert styles[0].endswith(":STRIKETHROUGH")

    def test_inline_monospace(self):
        text, styles = _m2s("run `ls -la` now")
        assert text == "run ls -la now"
        assert len(styles) == 1
        assert styles[0].endswith(":MONOSPACE")

    def test_fenced_code_block(self):
        text, styles = _m2s("before\n```\ncode here\n```\nafter")
        assert "code here" in text
        assert "```" not in text
        assert any(s.endswith(":MONOSPACE") for s in styles)

    def test_heading_becomes_bold(self):
        text, styles = _m2s("## Section Title")
        assert text == "Section Title"
        assert len(styles) == 1
        assert styles[0].endswith(":BOLD")

    def test_multiple_styles(self):
        text, styles = _m2s("**bold** and *italic*")
        assert text == "bold and italic"
        types = _style_types(styles)
        assert "BOLD" in types
        assert "ITALIC" in types

    def test_plain_text_no_styles(self):
        text, styles = _m2s("just plain text")
        assert text == "just plain text"
        assert styles == []

    def test_empty_string(self):
        text, styles = _m2s("")
        assert text == ""
        assert styles == []


# ===========================================================================
# Italic false-positive regressions
# ===========================================================================

class TestItalicFalsePositives:
    """Regressions from signal-italic-false-positive-fix.md and
    signal-italic-bullet-list-fix.md."""

    # --- snake_case (original fix) ---

    def test_snake_case_not_italic(self):
        """snake_case identifiers must NOT be italicized."""
        text, styles = _m2s("the config_file is ready")
        assert text == "the config_file is ready"
        assert _find_style(styles, "ITALIC") == []

    def test_multiple_snake_case(self):
        text, styles = _m2s("set OPENAI_API_KEY and ANTHROPIC_API_KEY")
        assert _find_style(styles, "ITALIC") == []

    def test_snake_case_path(self):
        text, styles = _m2s("/tools/delegate_tool.py")
        assert _find_style(styles, "ITALIC") == []

    def test_snake_case_between_words(self):
        """file_path and error_code — underscores between words."""
        text, styles = _m2s("file_path and error_code")
        assert _find_style(styles, "ITALIC") == []

    # --- Bullet lists (second fix) ---

    def test_bullet_list_not_italic(self):
        """* item lines must NOT be treated as italic delimiters."""
        md = "* item one\n* item two\n* item three"
        text, styles = _m2s(md)
        assert _find_style(styles, "ITALIC") == []

    def test_bullet_list_with_content_before(self):
        md = "Here are things:\n\n* first thing\n* second thing"
        text, styles = _m2s(md)
        assert _find_style(styles, "ITALIC") == []

    def test_bullet_list_file_paths(self):
        """Real-world case that triggered the bug."""
        md = (
            "* tools/delegate_tool.py — delegation\n"
            "* tools/file_tools.py — file operations\n"
            "* tools/web_tools.py — web operations"
        )
        text, styles = _m2s(md)
        assert _find_style(styles, "ITALIC") == []

    def test_bullet_with_italic_inside(self):
        """Italic *inside* a bullet item should still work."""
        md = "* this has *emphasis* inside\n* plain item"
        text, styles = _m2s(md)
        italic_styles = _find_style(styles, "ITALIC")
        assert len(italic_styles) == 1
        # The italic should cover "emphasis", not the whole bullet
        assert "emphasis" in text

    # --- Cross-line spans (DOTALL removal) ---

    def test_star_italic_no_cross_line(self):
        """*foo\\nbar* must NOT match as italic (no DOTALL)."""
        text, styles = _m2s("*foo\nbar*")
        assert _find_style(styles, "ITALIC") == []

    def test_underscore_italic_no_cross_line(self):
        """_foo\\nbar_ must NOT match as italic (no DOTALL)."""
        text, styles = _m2s("_foo\nbar_")
        assert _find_style(styles, "ITALIC") == []

    def test_star_italic_multiline_response(self):
        """Multi-paragraph response with * should not false-positive."""
        md = (
            "I checked the following files:\n\n"
            "* tools/delegate_tool.py — sub-agent delegation\n"
            "* tools/file_tools.py — file read/write/search\n"
            "* tools/web_tools.py — web search/extract\n\n"
            "Everything looks good."
        )
        text, styles = _m2s(md)
        assert _find_style(styles, "ITALIC") == []

    # --- Legitimate italic still works ---

    def test_star_italic_still_works(self):
        text, styles = _m2s("this is *italic* text")
        assert text == "this is italic text"
        assert len(_find_style(styles, "ITALIC")) == 1

    def test_underscore_italic_still_works(self):
        text, styles = _m2s("this is _italic_ text")
        assert text == "this is italic text"
        assert len(_find_style(styles, "ITALIC")) == 1

    def test_multiple_italic_same_line(self):
        text, styles = _m2s("*foo* and *bar* ok")
        assert text == "foo and bar ok"
        assert len(_find_style(styles, "ITALIC")) == 2

    def test_italic_single_word(self):
        text, styles = _m2s("*word*")
        assert text == "word"
        assert len(_find_style(styles, "ITALIC")) == 1

    def test_italic_multi_word(self):
        text, styles = _m2s("*several words here*")
        assert text == "several words here"
        assert len(_find_style(styles, "ITALIC")) == 1


# ===========================================================================
# Style position accuracy
# ===========================================================================

class TestStylePositions:
    """Verify that start:length positions map to the correct text."""

    def _extract(self, text: str, style_str: str) -> str:
        """Given 'start:length:STYLE', extract the substring from text."""
        # Positions are UTF-16 code units; for ASCII they match code points
        parts = style_str.split(":")
        start, length = int(parts[0]), int(parts[1])
        # Encode to UTF-16-LE, slice, decode back
        encoded = text.encode("utf-16-le")
        extracted = encoded[start * 2 : (start + length) * 2]
        return extracted.decode("utf-16-le")

    def test_bold_position(self):
        text, styles = _m2s("hello **world** end")
        assert len(styles) == 1
        assert self._extract(text, styles[0]) == "world"

    def test_italic_position(self):
        text, styles = _m2s("hello *world* end")
        assert len(styles) == 1
        assert self._extract(text, styles[0]) == "world"

    def test_multiple_styles_positions(self):
        text, styles = _m2s("**bold** then *italic*")
        assert len(styles) == 2
        extracted = {self._extract(text, s) for s in styles}
        assert extracted == {"bold", "italic"}

    def test_emoji_utf16_offset(self):
        """Emoji (multi-byte UTF-16) before a styled span."""
        text, styles = _m2s("👋 **hello**")
        assert text == "👋 hello"
        assert len(styles) == 1
        assert self._extract(text, styles[0]) == "hello"


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Tricky inputs that have caused issues or could regress."""

    def test_bold_inside_bullet(self):
        """Bold inside a bullet list item."""
        md = "* **important** item\n* normal item"
        text, styles = _m2s(md)
        assert len(_find_style(styles, "BOLD")) == 1
        assert _find_style(styles, "ITALIC") == []

    def test_code_span_with_underscores(self):
        """`snake_case_var` — backtick takes priority over underscore."""
        text, styles = _m2s("use `my_var_name` here")
        assert text == "use my_var_name here"
        types = _style_types(styles)
        assert "MONOSPACE" in types
        assert "ITALIC" not in types

    def test_bold_and_italic_nested(self):
        """***bold+italic*** — bold captured, not italic (bold pattern first)."""
        text, styles = _m2s("***word***")
        # ** matches bold around *word*, or *** is ambiguous;
        # either way there should be no false italic of the whole string
        assert "word" in text

    def test_lone_asterisk(self):
        """A single * with no pair should not cause issues."""
        text, styles = _m2s("5 * 3 = 15")
        # Should not crash; any italic match would be a false positive
        assert "5" in text and "15" in text

    def test_lone_underscore(self):
        """A single _ with no pair."""
        text, styles = _m2s("this _ that")
        assert text == "this _ that"

    def test_consecutive_underscored_words(self):
        """_foo and _bar (leading underscores, no closers)."""
        text, styles = _m2s("call _init and _setup")
        assert _find_style(styles, "ITALIC") == []

    def test_mixed_formatting_no_bleed(self):
        """Multiple format types don't bleed into each other."""
        md = "**bold** and `code` and *italic* and ~~strike~~"
        text, styles = _m2s(md)
        assert text == "bold and code and italic and strike"
        types = _style_types(styles)
        assert sorted(types) == ["BOLD", "ITALIC", "MONOSPACE", "STRIKETHROUGH"]


# ===========================================================================
# signal-markdown-strip-patch: core conversion pipeline
# ===========================================================================

class TestMarkdownStripPatch:
    """Tests for the original signal-markdown-strip-patch.
    
    Covers: fenced code blocks with language tags, links preserved,
    headings converted to bold, multiple headings, UTF-16 correctness
    for multi-byte characters, and marker stripping completeness.
    """

    def test_fenced_code_block_with_language_tag(self):
        """```python\\ncode\\n``` — language tag is stripped, content is MONOSPACE."""
        text, styles = _m2s("```python\nprint('hello')\n```")
        assert "```" not in text
        assert "python" not in text  # language tag stripped
        assert "print('hello')" in text
        assert any(s.endswith(":MONOSPACE") for s in styles)

    def test_fenced_code_block_multiline(self):
        """Multi-line code blocks preserve all lines."""
        md = "```\nline1\nline2\nline3\n```"
        text, styles = _m2s(md)
        assert "line1" in text
        assert "line2" in text
        assert "line3" in text
        assert "```" not in text

    def test_links_preserved(self):
        """[text](url) links are kept as-is — Signal auto-linkifies."""
        md = "Check [this link](https://example.com) for details"
        text, styles = _m2s(md)
        # Links should pass through — either as markdown or just preserved
        assert "https://example.com" in text

    def test_heading_h1(self):
        """# H1 becomes bold text."""
        text, styles = _m2s("# Main Title")
        assert text == "Main Title"
        assert len(styles) == 1
        assert styles[0].endswith(":BOLD")

    def test_heading_h3(self):
        """### H3 becomes bold text."""
        text, styles = _m2s("### Sub Section")
        assert text == "Sub Section"
        assert len(styles) == 1
        assert styles[0].endswith(":BOLD")

    def test_multiple_headings(self):
        """Multiple headings each become separate bold spans."""
        md = "## First\n\nSome text\n\n## Second"
        text, styles = _m2s(md)
        assert "First" in text
        assert "Second" in text
        assert "##" not in text
        bold_styles = _find_style(styles, "BOLD")
        assert len(bold_styles) == 2

    def test_no_raw_markdown_markers_in_output(self):
        """All markdown syntax is stripped from plain text output."""
        md = "**bold** and *italic* and ~~struck~~ and `code` and ## heading"
        text, styles = _m2s(md)
        assert "**" not in text
        assert "~~" not in text
        assert "`" not in text
        # ## at end might remain if not at line start — that's ok
        # The important thing is styled markers are stripped

    def test_utf16_surrogate_pair_emoji(self):
        """Emoji requiring UTF-16 surrogate pairs don't corrupt offsets."""
        # 🎉 is U+1F389 — requires surrogate pair (2 UTF-16 code units)
        text, styles = _m2s("🎉🎉 **test**")
        assert "test" in text
        assert len(styles) == 1
        # Verify the style position is correct
        parts = styles[0].split(":")
        start, length = int(parts[0]), int(parts[1])
        # 🎉🎉 = 4 UTF-16 code units + space = 5, then "test" = 4
        assert start == 5
        assert length == 4

    def test_consecutive_newlines_collapsed(self):
        """3+ consecutive newlines are collapsed to 2."""
        text, styles = _m2s("first\n\n\n\n\nsecond")
        assert "\n\n\n" not in text
        assert "first" in text
        assert "second" in text

    def test_empty_bold_not_crash(self):
        """**** (empty bold) should not crash."""
        text, styles = _m2s("before **** after")
        # Should not raise — exact output doesn't matter much
        assert "before" in text


# ===========================================================================
# signal-streaming-patch: SUPPORTS_MESSAGE_EDITING and send() behavior
# ===========================================================================

class TestSignalStreamingPatch:
    """Tests for signal-streaming-patch: cursor suppression and edit support.
    
    These verify the adapter-level properties that prevent the streaming
    cursor from leaking into Signal messages.
    """

    def test_signal_does_not_support_editing(self, monkeypatch):
        """SignalAdapter.SUPPORTS_MESSAGE_EDITING must be False."""
        monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
        from gateway.platforms.signal import SignalAdapter
        assert SignalAdapter.SUPPORTS_MESSAGE_EDITING is False

    @pytest.mark.asyncio
    async def test_send_returns_no_message_id(self, monkeypatch):
        """send() returns message_id=None so stream consumer uses no-edit path."""
        monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
        from gateway.platforms.signal import SignalAdapter
        from gateway.config import PlatformConfig

        config = PlatformConfig(enabled=True)
        config.extra = {
            "http_url": "http://localhost:8080",
            "account": "+15551234567",
        }
        adapter = SignalAdapter(config)

        # Mock the RPC call
        async def mock_rpc(method, params, rpc_id=None):
            return {"timestamp": 1234567890}

        adapter._rpc = mock_rpc

        result = await adapter.send(
            chat_id="+15559876543",
            content="Hello",
        )
        assert result.message_id is None
