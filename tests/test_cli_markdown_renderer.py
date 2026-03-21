"""Tests for the streaming markdown renderer."""

import pytest
from unittest.mock import patch, MagicMock
from hermes_cli.markdown_renderer import StreamingMarkdownRenderer, _hex_to_ansi_fg


# ── ANSI constants for assertions ──────────────────────────────────────
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_UNDERLINE = "\033[4m"
_STRIKE = "\033[9m"
_RST = "\033[0m"


class TestHexToAnsiFg:
    """Test hex-to-ANSI color conversion."""

    def test_standard_hex(self):
        assert _hex_to_ansi_fg("#FF0000") == "\033[38;2;255;0;0m"

    def test_lowercase_hex(self):
        assert _hex_to_ansi_fg("#4dd0e1") == "\033[38;2;77;208;225m"

    def test_no_hash(self):
        assert _hex_to_ansi_fg("FFBF00") == "\033[38;2;255;191;0m"

    def test_invalid_length_returns_empty(self):
        assert _hex_to_ansi_fg("#FFF") == ""
        assert _hex_to_ansi_fg("") == ""


class TestSkinIntegration:
    """Test that the renderer pulls colors from the skin engine."""

    def test_uses_skin_colors(self):
        mock_skin = MagicMock()
        mock_skin.get_color.side_effect = lambda key, fallback="": {
            "banner_accent": "#FF0000",
            "ui_label": "#00FF00",
            "banner_dim": "#0000FF",
        }.get(key, fallback)

        with patch.dict("sys.modules", {"hermes_cli.skin_engine": MagicMock(get_active_skin=lambda: mock_skin)}):
            r = StreamingMarkdownRenderer()
            result = r.render_line("# Title")
            assert "\033[38;2;255;0;0m" in result
            result = r.render_line("Use `code` here")
            assert "\033[38;2;0;255;0m" in result

    def test_fallback_on_import_error(self):
        """Renderer works even if skin engine is unavailable."""
        with patch.dict("sys.modules", {"hermes_cli.skin_engine": None}):
            r = StreamingMarkdownRenderer()
            result = r.render_line("# Title")
            assert _BOLD in result
            assert "Title" in result


class TestInlineFormatting:
    """Test inline markdown formatting."""

    def setup_method(self):
        self.r = StreamingMarkdownRenderer()

    def test_bold(self):
        result = self.r._render_inline("This is **bold** text")
        assert _BOLD in result
        assert "bold" in result
        assert "**" not in result

    def test_bold_underscore(self):
        result = self.r._render_inline("This is __bold__ text")
        assert _BOLD in result
        assert "__" not in result

    def test_italic(self):
        result = self.r._render_inline("This is *italic* text")
        assert _ITALIC in result
        assert "italic" in result

    def test_bold_italic(self):
        result = self.r._render_inline("This is ***bold italic*** text")
        assert _BOLD in result
        assert _ITALIC in result
        assert "***" not in result

    def test_strikethrough(self):
        result = self.r._render_inline("This is ~~struck~~ text")
        assert _STRIKE in result
        assert "struck" in result
        assert "~~" not in result

    def test_inline_code(self):
        result = self.r._render_inline("Use `pip install` here")
        assert "pip install" in result
        assert "`" not in result

    def test_link(self):
        result = self.r._render_inline("Visit [example](https://example.com)")
        assert _UNDERLINE in result
        assert "example" in result
        assert "](https://" not in result

    def test_plain_text_unchanged(self):
        result = self.r._render_inline("No formatting here")
        assert result == "No formatting here"

    def test_mixed_formatting(self):
        result = self.r._render_inline("**bold** and `code`")
        assert _BOLD in result


class TestStreamingMarkdownRenderer:
    """Test the stateful streaming renderer."""

    def setup_method(self):
        self.r = StreamingMarkdownRenderer()

    def test_header_h1(self):
        result = self.r.render_line("# Title")
        assert _BOLD in result
        assert "Title" in result
        assert "#" not in result

    def test_header_h2(self):
        result = self.r.render_line("## Subtitle")
        assert "Subtitle" in result
        assert "##" not in result

    def test_header_h3(self):
        result = self.r.render_line("### Section")
        assert "Section" in result
        assert "###" not in result

    def test_header_h4(self):
        result = self.r.render_line("#### Deep")
        assert "Deep" in result
        assert "####" not in result

    def test_header_h5(self):
        result = self.r.render_line("##### Very Deep")
        assert "Very Deep" in result
        assert "#####" not in result

    def test_header_h6(self):
        result = self.r.render_line("###### Deepest")
        assert "Deepest" in result
        assert "######" not in result

    def test_code_block_toggle(self):
        open_line = self.r.render_line("```python")
        assert "╭" in open_line
        assert "python" in open_line

        code_line = self.r.render_line("x = 42")
        # Pygments may split "x = 42" with color codes around "42"
        assert "x" in code_line
        assert "42" in code_line

        close_line = self.r.render_line("```")
        assert "╯" in close_line

    def test_code_block_state_reset(self):
        self.r.render_line("```")
        assert self.r._in_code_block is True
        self.r.render_line("```")
        assert self.r._in_code_block is False

    def test_unordered_list(self):
        result = self.r.render_line("- Item one")
        assert "•" in result
        assert "Item one" in result

    def test_unordered_list_nested(self):
        result = self.r.render_line("  - Nested item")
        assert "•" in result
        assert "Nested" in result

    def test_ordered_list(self):
        result = self.r.render_line("1. First")
        assert "1." in result
        assert "First" in result

    def test_blockquote(self):
        result = self.r.render_line("> Quote text")
        assert "│" in result
        assert "Quote text" in result

    def test_nested_blockquote(self):
        result = self.r.render_line("> > Nested quote")
        # Should have two bar segments
        assert result.count("│") == 2
        assert "Nested quote" in result

    def test_triple_nested_blockquote(self):
        result = self.r.render_line("> > > Deep")
        assert result.count("│") == 3

    def test_horizontal_rule(self):
        result = self.r.render_line("---")
        assert "─" in result

    def test_table_rows_buffered(self):
        """All table rows are buffered (return None) until a non-table line."""
        assert self.r.render_line("| Foo | Bar |") is None
        assert self.r.render_line("|-----|-----|") is None
        assert self.r.render_line("| A | B |") is None

    def test_table_rendered_on_flush(self):
        """Table renders with aligned columns when flushed by non-table line."""
        self.r.render_line("| Name | Status |")
        self.r.render_line("|------|--------|")
        self.r.render_line("| Long Name Here | OK |")
        result = self.r.render_line("")  # triggers flush
        assert "─" in result
        assert "┼" in result
        assert "Name" in result
        assert "Long Name Here" in result

    def test_table_column_alignment(self):
        """Column widths are calculated from ALL rows, not just header."""
        self.r.render_line("| A | B |")
        self.r.render_line("|---|---|")
        self.r.render_line("| Wide Content | X |")
        result = self.r.flush()
        # Header "A" should be padded to match "Wide Content"
        assert "A   " in result or "A " in result

    def test_table_inline_formatting(self):
        self.r.render_line("| Col1 | Col2 |")
        self.r.render_line("|------|------|")
        self.r.render_line("| **bold** | `code` |")
        result = self.r.flush()
        assert _BOLD in result

    def test_table_flush_unbuffered(self):
        """Flush returns None when no buffered content."""
        assert self.r.flush() is None

    def test_table_flush_no_separator(self):
        """Buffered pipe rows with no separator are flushed as plain text."""
        self.r.render_line("| Not a table |")
        result = self.r.flush()
        assert result is not None
        assert "Not a table" in result

    def test_plain_line(self):
        result = self.r.render_line("Just a normal line")
        assert result == "Just a normal line"

    def test_empty_line(self):
        result = self.r.render_line("")
        assert result == ""

    def test_code_block_no_lang(self):
        open_line = self.r.render_line("```")
        assert "╭" in open_line
        assert self.r._in_code_block is True

    def test_inline_in_list(self):
        result = self.r.render_line("- Use **bold** in list")
        assert "•" in result
        assert _BOLD in result
