"""Tests for CLI help rendering."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_show_help_includes_plugin_commands(monkeypatch):
    from cli import HermesCLI
    import cli as cli_mod

    printed_lines: list[str] = []

    class _FakeChatConsole:
        def print(self, text):
            printed_lines.append(str(text))

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "gpt-5"
    cli_obj.agent = None
    cli_obj.console = MagicMock()

    monkeypatch.setattr(cli_mod, "_cprint", lambda msg: printed_lines.append(str(msg)))
    monkeypatch.setattr(cli_mod, "ChatConsole", _FakeChatConsole)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_commands",
        lambda: {"design-sync": {"description": "Sync design context"}},
    )

    cli_obj.show_help()

    assert any("Plugin Commands" in line for line in printed_lines)
    assert any("/design-sync" in line for line in printed_lines)
