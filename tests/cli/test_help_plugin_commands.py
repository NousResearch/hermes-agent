"""Regression test for issue #55609.

``/help`` (``HermesCLI.show_help``) must list plugin-registered slash commands
so they are discoverable, mirroring the existing Skill Commands section. Before
the fix, plugin commands worked at dispatch time but were never rendered by
``show_help``, leaving users with no way to discover commands like
``/ponytail``.
"""
from unittest.mock import MagicMock, patch

import cli
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = None
    cli_obj._pending_input = MagicMock()
    return cli_obj


def _render_help(plugin_commands):
    """Render show_help with a fake plugin-command registry; return the text."""
    cli_obj = _make_cli()
    printed: list[str] = []

    class _FakeConsole:
        def print(self, *args, **kwargs):
            printed.append(" ".join(str(a) for a in args))

    with patch("hermes_cli.plugins.get_plugin_commands", return_value=plugin_commands), \
         patch.object(cli, "_ensure_skill_commands", return_value={}), \
         patch.object(cli, "get_skill_bundles", return_value={}), \
         patch.object(HermesCLI, "_command_available", return_value=True), \
         patch.object(cli, "ChatConsole", _FakeConsole), \
         patch.object(cli, "_cprint", side_effect=lambda t: printed.append(str(t))):
        cli_obj.show_help()

    return "\n".join(printed)


def test_show_help_lists_plugin_commands():
    plugin_commands = {
        "ponytail": {"description": "Run ponytail review", "handler": object(), "plugin": "ponytail"},
        "ponytail-audit": {"description": "Audit with ponytail", "handler": object(), "plugin": "ponytail"},
    }
    out = _render_help(plugin_commands)

    assert "Plugin Commands" in out
    assert "/ponytail" in out
    assert "/ponytail-audit" in out
    assert "Run ponytail review" in out


def test_show_help_no_plugin_section_when_empty():
    out = _render_help({})
    assert "Plugin Commands" not in out
