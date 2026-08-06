"""Discord format_message: tables converted to bullet groups."""

import types
import sys


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


class TestDiscordFormatMessage:

    def test_table_converted_to_bullets(self):
        adapter = _make_discord_adapter()
        text = (
            "Results:\n\n"
            "| Name | Score |\n"
            "|------|-------|\n"
            "| Alice | 95   |\n"
            "| Bob   | 80   |\n"
            "\nDone."
        )
        out = adapter.format_message(text)
        assert "**Alice**" in out
        assert "• Score: 95" in out
        assert "**Bob**" in out
        assert "• Score: 80" in out
        assert out.startswith("Results:")
        assert out.rstrip().endswith("Done.")
        assert "|---" not in out

    def test_plain_text_unchanged(self):
        adapter = _make_discord_adapter()
        text = "Hello world, no tables here."
        assert adapter.format_message(text) == text

    def test_code_block_table_unchanged(self):
        adapter = _make_discord_adapter()
        text = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
        assert adapter.format_message(text) == text

    def test_empty_string(self):
        adapter = _make_discord_adapter()
        assert adapter.format_message("") == ""
