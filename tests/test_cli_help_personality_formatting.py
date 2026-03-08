import os
import re
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_cli():
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.personalities = {
        "zebra": "A personality with extra spacing\nfor tests.",
        "alpha": "Short and direct.",
    }
    cli_obj.system_prompt = ""
    cli_obj.agent = None
    return cli_obj


class TestCLIHelpFormatting:
    def test_show_help_groups_and_aligns_commands(self, capsys):
        cli_obj = _make_cli()

        with patch("cli._skill_commands", {
            "/gif-search": {"description": "Search for GIFs across providers"},
        }), patch("cli._cprint", lambda text: print(text)):
            cli_obj.show_help()

        output = _strip_ansi(capsys.readouterr().out)

        assert "Session" in output
        assert "Configuration" in output
        assert "Tools & Platform" in output
        assert "Context & Automation" in output
        assert re.search(r"/new\s+- Start a new session with a fresh conversation", output)
        assert re.search(r"/personality\s+- List or switch predefined personalities", output)
        assert "Skill Commands (1 installed)" in output
        assert "Use a skill command directly to run it." in output
        assert re.search(r"/gif-search\s+- Search for GIFs across providers", output)

    def test_show_help_truncates_long_skill_descriptions(self, capsys):
        cli_obj = _make_cli()
        long_description = " ".join(["Search across providers with extra detail"] * 6)

        with patch("cli._skill_commands", {
            "/gif-search": {"description": long_description},
        }), patch("cli._cprint", lambda text: print(text)), \
             patch("cli.shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            cli_obj.show_help()

        output = _strip_ansi(capsys.readouterr().out)
        assert "..." in output
        assert long_description not in output

    def test_show_help_uses_more_space_on_wider_terminals(self, capsys):
        cli_obj = _make_cli()
        description = "Search and download GIFs from Tenor using curl. No dependencies beyond curl and jq."

        with patch("cli._skill_commands", {
            "/gif-search": {"description": description},
        }), patch("cli._cprint", lambda text: print(text)), \
             patch("cli.shutil.get_terminal_size", return_value=os.terminal_size((140, 24))):
            cli_obj.show_help()

        output = _strip_ansi(capsys.readouterr().out)
        assert description in output


class TestCLIPersonalityFormatting:
    def test_personality_list_is_sorted_and_separated(self, capsys):
        cli_obj = _make_cli()

        with patch("cli.shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            cli_obj._handle_personality_command("/personality")

        output = capsys.readouterr().out

        assert "Personalities" in output
        assert "  alpha\n    Short and direct." in output
        assert "  zebra\n    A personality with extra spacing for tests." in output
        assert output.index("  alpha") < output.index("  zebra")
        assert "-" * 42 in output
        assert "Usage: /personality <name>" in output

    def test_personality_preview_expands_on_wider_terminals(self, capsys):
        cli_obj = _make_cli()
        cli_obj.personalities = {
            "builder": "very long prompt " * 10,
        }

        with patch("cli.shutil.get_terminal_size", return_value=os.terminal_size((140, 24))):
            cli_obj._handle_personality_command("/personality")

        output = capsys.readouterr().out
        assert "very long prompt very long prompt very long prompt" in output
