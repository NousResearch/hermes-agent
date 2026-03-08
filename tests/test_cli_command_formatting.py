import os
import re
import sys
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_cli():
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-opus-4.6"
    cli_obj.base_url = "https://openrouter.ai/api/v1"
    cli_obj.api_key = "sk-test-1234"
    cli_obj.provider = "openrouter"
    cli_obj.requested_provider = None
    cli_obj._explicit_api_key = None
    cli_obj._explicit_base_url = None
    cli_obj.max_turns = 90
    cli_obj.enabled_toolsets = ["file", "web"]
    cli_obj.verbose = False
    cli_obj.session_start = datetime(2026, 3, 8, 20, 0, 0)
    cli_obj.personalities = {}
    cli_obj.system_prompt = ""
    cli_obj.agent = None
    return cli_obj


class TestCLIToolsFormatting:
    def test_show_tools_uses_rich_panel_and_table(self, capsys):
        cli_obj = _make_cli()
        tools = [
            {"function": {"name": "read_file", "description": "Read file contents from disk. Supports text files."}},
            {"function": {"name": "search_files", "description": "Search files by pattern across the repository. Supports regex."}},
            {"function": {"name": "web_search", "description": "Search the web for relevant information and sources."}},
        ]
        toolsets = {"read_file": "file", "search_files": "file", "web_search": "web"}

        with patch("cli.get_tool_definitions", return_value=tools), \
             patch("cli.get_toolset_for_tool", side_effect=lambda name: toolsets[name]), \
             patch("cli._cprint", lambda text: print(text)), \
             patch("cli.shutil.get_terminal_size", return_value=os.terminal_size((100, 24))):
            cli_obj.show_tools()

        output = _strip_ansi(capsys.readouterr().out)
        assert "Available Tools" in output
        assert "Toolset" in output
        assert "Tool" in output
        assert "Description" in output
        assert "file" in output
        assert "web" in output
        assert "read_file" in output
        assert "web_search" in output
        assert "Total: 3 tools" in output


class TestCLIConfigFormatting:
    def test_show_config_uses_panel_sections(self, capsys):
        cli_obj = _make_cli()

        with patch("cli._cprint", lambda text: print(text)):
            cli_obj.show_config()

        output = _strip_ansi(capsys.readouterr().out)
        assert "Configuration" in output
        assert "Model" in output
        assert "Terminal" in output
        assert "Agent" in output
        assert "Session" in output
        assert "anthropic/claude-opus-4.6" in output
        assert "file, web" in output


class TestCLIProviderFormatting:
    def test_provider_command_uses_rich_panel_and_table(self, capsys):
        from cli import HermesCLI

        cli_obj = _make_cli()
        providers = [
            {"id": "openrouter", "label": "OpenRouter", "authenticated": True, "aliases": ["or"]},
            {"id": "anthropic", "label": "Anthropic", "authenticated": False, "aliases": []},
        ]

        with patch("hermes_cli.models.list_available_providers", return_value=providers), \
             patch("cli._cprint", lambda text: print(text)):
            HermesCLI.process_command(cli_obj, "/provider")

        output = _strip_ansi(capsys.readouterr().out)
        assert "Providers" in output
        assert "Current provider:" in output
        assert "OpenRouter (openrouter)" in output
        assert "Status" in output
        assert "Provider" in output
        assert "Label" in output
        assert "[✓]" in output
        assert "[✗]" in output
        assert "openrouter" in output
        assert "OpenRouter (also: or) ← active" in output
        assert "anthropic" in output
        assert "Switch: /model provider:model-name" in output
        assert "Setup:  hermes setup" in output
