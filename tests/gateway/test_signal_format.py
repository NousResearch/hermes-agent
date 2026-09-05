"""Tests for Signal markdown_to_signal() formatting.

Covers the markdown-to-bodyRanges conversion pipeline: bold, italic,
strikethrough, monospace, code blocks, headings, and — critically — the
false-positive regressions that caused spurious italics in production.
"""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.signal import SignalAdapter
from gateway.platforms.signal_format import markdown_to_signal


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _m2s(text: str):
    """Shorthand: return (plain_text, styles)."""
    return markdown_to_signal(text)


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


    def test_italic_single_asterisk(self):
        text, styles = _m2s("hello *world*")
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


    # --- Bullet lists (second fix) ---


    def test_hyphen_bullet_list_uses_signal_safe_bullets(self):
        """Signal does not render Markdown list markers; normalize them."""
        md = "- item one\n- item two"
        text, styles = _m2s(md)
        assert text == "• item one\n• item two"
        assert styles == []


    def test_bullet_list_file_paths(self):
        """Real-world case that triggered the bug."""
        md = (
            "* tools/delegate_tool.py — delegation\n"
            "* tools/file_tools.py — file operations\n"
            "* tools/web_tools.py — web operations"
        )
        text, styles = _m2s(md)
        assert _find_style(styles, "ITALIC") == []


    # --- Cross-line spans (DOTALL removal) ---


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


    def test_multiple_italic_same_line(self):
        text, styles = _m2s("*foo* and *bar* ok")
        assert text == "foo and bar ok"
        assert len(_find_style(styles, "ITALIC")) == 2

    def test_italic_single_word(self):
        text, styles = _m2s("*word*")
        assert text == "word"
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


    def test_lone_asterisk(self):
        """A single * with no pair should not cause issues."""
        text, styles = _m2s("5 * 3 = 15")
        # Should not crash; any italic match would be a false positive
        assert "5" in text and "15" in text


# ===========================================================================
# signal-markdown-strip-patch: core conversion pipeline
# ===========================================================================

class TestMarkdownStripPatch:
    """Tests for the original signal-markdown-strip-patch.
    
    Covers: fenced code blocks with language tags, links preserved,
    headings converted to bold, multiple headings, UTF-16 correctness
    for multi-byte characters, and marker stripping completeness.
    """


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


    def test_multiple_headings(self):
        """Multiple headings each become separate bold spans."""
        md = "## First\n\nSome text\n\n## Second"
        text, styles = _m2s(md)
        assert "First" in text
        assert "Second" in text
        assert "##" not in text
        bold_styles = _find_style(styles, "BOLD")
        assert len(bold_styles) == 2

        # ## at end might remain if not at line start — that's ok
        # The important thing is styled markers are stripped


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


# ---------------------------------------------------------------------------
# Lone surrogates (#55143)
# ---------------------------------------------------------------------------

class TestLoneSurrogates:
    """Lone UTF-16 surrogates (U+D800-U+DFFF) reach ``markdown_to_signal`` from the
    ``send_message`` tool and from the gateway status path, neither of which passes
    through the ``final_response`` surrogate chokepoints. ``_utf16_len`` cannot encode
    them, so the whole Signal send aborted and the reply never reached the user."""

    def test_lone_surrogate_with_formatting_does_not_raise(self):
        """The regression: without the fix this call raises UnicodeEncodeError
        inside ``_utf16_len``, before any assertion runs."""
        text, styles = _m2s("**bold \ud800 text** and `code \udfff here`")
        assert "\ud800" not in text and "\udfff" not in text
        assert _style_types(styles) == ["BOLD", "MONOSPACE"]

    def test_lone_surrogate_is_replaced_not_dropped(self):
        """U+FFFD is one UTF-16 code unit, so offsets after it stay correct."""
        text, styles = _m2s("\ud800**b**")
        assert text == "�b"
        assert styles == ["1:1:BOLD"]

    def test_bare_lone_surrogate_without_formatting(self):
        text, styles = _m2s("\udfff")
        assert text == "�"
        assert styles == []

    @pytest.mark.parametrize("raw,expected_text", [
        ("**bold**", "bold"),
        ("*it*", "it"),
        ("~~s~~", "s"),
        ("`c`", "c"),
        ("# H1", "H1"),
        ("", ""),
    ])
    def test_surrogate_free_input_is_unchanged(self, raw, expected_text):
        """Control: sanitizing must not alter ordinary text."""
        assert _m2s(raw)[0] == expected_text

    def test_astral_pair_still_counts_as_two_utf16_units(self):
        """A valid surrogate PAIR is not a lone surrogate and must survive intact,
        with its two-code-unit width still reflected in the offsets."""
        text, styles = _m2s("\U0001F389 **b**")
        assert text == "\U0001F389 b"
        assert styles == ["3:1:BOLD"]
