"""Tests for Discord markdown table -> code block conversion.

Discord does not support markdown tables. format_message() must convert
pipe-delimited tables into monospace code-block tables so they render
correctly in Discord.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, danger=3, green=1, blurple=2, red=3
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="test"))


# ============================================================================
# Core table conversion
# ============================================================================


class TestTableToCodeBlock:
    """Markdown tables should be converted to code-block ASCII tables."""

    def test_simple_table_converted_to_code_block(self):
        adapter = _make_adapter()
        input_md = (
            "| Column A | Column B |\n"
            "|----------|----------|\n"
            "| val 1    | val 2    |\n"
        )
        result = adapter.format_message(input_md)
        assert "```" in result
        assert "|" in result
        assert "Column A" in result
        assert "val 1" in result

    def test_table_inside_code_block_not_modified(self):
        """Tables inside fenced code blocks must be left alone."""
        adapter = _make_adapter()
        input_md = (
            "```\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n"
            "```\n"
        )
        result = adapter.format_message(input_md)
        assert result.count("```") == 2
        assert result == input_md

    def test_table_inside_inline_code_not_modified(self):
        """Tables inside inline code must be left alone."""
        adapter = _make_adapter()
        input_md = "Use `| a | b |` for something"
        result = adapter.format_message(input_md)
        assert result == input_md

    def test_multiple_tables_each_wrapped(self):
        """Multiple tables should each be wrapped in their own code block."""
        adapter = _make_adapter()
        input_md = (
            "| T1 | V1 |\n|----|----|\n| a  | b  |\n\n"
            "Some text between.\n\n"
            "| T2 | V2 |\n|----|----|\n| c  | d  |\n"
        )
        result = adapter.format_message(input_md)
        code_block_count = result.count("```")
        assert code_block_count == 4  # 2 open + 2 close

    def test_non_table_pipe_lines_untouched(self):
        """Lines with pipes but not table syntax should not be converted."""
        adapter = _make_adapter()
        input_md = "Use git log --oneline | head -5 to check commits"
        result = adapter.format_message(input_md)
        assert "```" not in result
        assert result == input_md

    def test_text_around_table_preserved(self):
        """Text before and after a table should be preserved."""
        adapter = _make_adapter()
        input_md = (
            "Here is a summary:\n\n"
            "| Key | Value |\n"
            "|-----|-------|\n"
            "| foo | bar   |\n\n"
            "That's it."
        )
        result = adapter.format_message(input_md)
        assert "Here is a summary:" in result
        assert "That's it." in result
        assert "```" in result

    def test_table_with_alignment_markers_stripped(self):
        """Alignment markers like :---: should be removed from rendered table."""
        adapter = _make_adapter()
        input_md = (
            "| Left | Center | Right |\n"
            "|:-----|:------:|------:|\n"
            "| a    | b      | c     |\n"
        )
        result = adapter.format_message(input_md)
        assert "```" in result
        assert "Left" in result
        assert "Center" in result
        assert "Right" in result

    def test_empty_table_header_still_converted(self):
        """Even minimal tables should be wrapped."""
        adapter = _make_adapter()
        input_md = "| A |\n|---|\n| 1 |\n"
        result = adapter.format_message(input_md)
        assert "```" in result
        assert "A" in result

    def test_table_with_bold_content(self):
        """Bold markers inside table cells should be preserved."""
        adapter = _make_adapter()
        input_md = (
            "| Item | Status |\n"
            "|------|--------|\n"
            "| **Test** | **PASS** |\n"
        )
        result = adapter.format_message(input_md)
        assert "```" in result
        assert "Test" in result
        assert "PASS" in result

    def test_table_with_inline_code_cells(self):
        """Inline code inside table cells should be preserved."""
        adapter = _make_adapter()
        input_md = (
            "| File | Command |\n"
            "|------|---------|\n"
            "| `foo.py` | `pytest -v` |\n"
        )
        result = adapter.format_message(input_md)
        assert "foo.py" in result
        assert "pytest -v" in result

    def test_table_with_links_in_cells(self):
        """Markdown links inside table cells should be preserved."""
        adapter = _make_adapter()
        input_md = (
            "| Name | URL |\n"
            "|------|-----|\n"
            "| [GitHub](https://github.com) | [Docs](https://docs.example.com) |\n"
        )
        result = adapter.format_message(input_md)
        assert "GitHub" in result
        assert "https://github.com" in result


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge cases for format_message."""

    def test_no_tables_passthrough(self):
        adapter = _make_adapter()
        input_md = "**Bold** and *italic* and `code`"
        result = adapter.format_message(input_md)
        assert "```" not in result
        assert result == input_md

    def test_empty_string(self):
        adapter = _make_adapter()
        assert adapter.format_message("") == ""

    def test_code_block_with_table_like_content_preserved(self):
        """Code blocks that happen to contain table-like syntax."""
        adapter = _make_adapter()
        input_md = (
            "```python\n"
            "data = {'|': 'pipe'}\n"
            "| col1 | col2 |\n"
            "```\n"
        )
        result = adapter.format_message(input_md)
        # Should not add extra code block wrappers inside existing block
        assert result.count("```") == 2

    def test_single_pipe_line_not_table(self):
        """A single line with pipes but no separator is not a table."""
        adapter = _make_adapter()
        input_md = "| just a pipe | not a table |"
        result = adapter.format_message(input_md)
        assert "```" not in result
        assert result == input_md

    def test_separator_only_not_table(self):
        """A separator row alone (no header) should not be converted."""
        adapter = _make_adapter()
        input_md = "|---|----|"
        result = adapter.format_message(input_md)
        assert "```" not in result
        assert result == input_md

    def test_bold_outside_table_preserved(self):
        adapter = _make_adapter()
        input_md = "**This is bold** and | not | a | table |"
        result = adapter.format_message(input_md)
        assert "**This is bold**" in result
        assert "```" not in result

    def test_very_wide_table_still_wrapped(self):
        """A table with many columns (>6) should still be wrapped."""
        adapter = _make_adapter()
        cols = 8
        header = "| " + " | ".join(f"Col{i}" for i in range(1, cols+1)) + " |"
        sep = "|" + "|".join(["---"] * cols) + "|"
        row = "| " + " | ".join(f"v{i}" for i in range(1, cols+1)) + " |"
        input_md = f"{header}\n{sep}\n{row}\n"
        result = adapter.format_message(input_md)
        assert "```" in result
        assert "Col1" in result
        assert "Col8" in result
