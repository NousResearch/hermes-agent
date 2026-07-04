"""Test expand_details_blocks() HTML collapsible expansion.

These tests live under test_discord_format for now because
expand_details_blocks was added for the Discord adapter; the helper
lives in gateway.platforms.helpers and other plain-text platforms
could reuse it.
"""

import types
import sys
import pytest


def _make_discord_adapter():
    """Construct a DiscordAdapter with discord.py stubbed out."""
    fake_discord = types.ModuleType("discord")
    fake_discord.Intents = type("Intents", (), {"default": classmethod(lambda cls: cls())})
    fake_discord.Message = object
    fake_ext = types.ModuleType("discord.ext")
    fake_commands = types.ModuleType("discord.ext.commands")
    fake_ext.commands = fake_commands
    fake_discord.ext = fake_ext
    sys.modules.setdefault("discord", fake_discord)
    sys.modules.setdefault("discord.ext", fake_ext)
    sys.modules.setdefault("discord.ext.commands", fake_commands)

    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = object.__new__(DiscordAdapter)
    return adapter


class TestExpandDetailsBlocks:

    def test_single_line_details_with_summary(self):
        adapter = _make_discord_adapter()
        text = "<details><summary>Notes</summary>Some hidden text here.</details>"
        out = adapter.format_message(text)
        assert "📎 Notes" in out
        assert "Some hidden text here." in out
        assert "<details>" not in out
        assert "</details>" not in out
        assert "<summary>" not in out
        assert "</summary>" not in out

    def test_multiline_details_block(self):
        adapter = _make_discord_adapter()
        text = "<details><summary>Detailed Analysis</summary>\n\nLine 1\nLine 2\nLine 3\n</details>"
        out = adapter.format_message(text)
        assert "📎 Detailed Analysis" in out
        assert "Line 1" in out
        assert "Line 2" in out
        assert "Line 3" in out

    def test_plain_text_unchanged(self):
        adapter = _make_discord_adapter()
        text = "Hello world, no details here."
        assert adapter.format_message(text) == text

    def test_code_block_unchanged(self):
        adapter = _make_discord_adapter()
        text = "```\n<details><summary>fake</summary>should not expand</details>\n```"
        out = adapter.format_message(text)
        assert "<details>" in out
        assert "</details>" in out

    def test_empty_string(self):
        adapter = _make_discord_adapter()
        assert adapter.format_message("") == ""

    def test_table_and_details_together(self):
        adapter = _make_discord_adapter()
        text = (
            "Results:\n\n"
            "| Name | Score |\n"
            "|------|-------|\n"
            "| Alice | 95   |\n"
            "| Bob   | 80   |\n"
            "\n"
            "<details><summary>Note</summary>All scores are final.</details>"
        )
        out = adapter.format_message(text)
        # Table converted to bullets
        assert "**Alice**" in out
        assert "• Score: 95" in out
        # Details expanded
        assert "📎 Note" in out
        assert "All scores are final." in out
        assert "<details>" not in out
