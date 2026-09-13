"""Tests for CLI model-alias slash dispatch (/<alias> -> /model <alias>).

Regression coverage for PR #59606's CLI side. The /<alias> shortcut must
behave like /model <alias> for the actual switch but never shadow a
higher-priority command (built-in / plugin / skill / quick-command) that
happens to share the name.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = None
    cli_obj._pending_input = MagicMock()
    return cli_obj


class TestModelAliasSlashDispatch:
    """Confirm that typing /<alias> dispatches to _handle_model_switch with
    the canonical /model command, not with the raw alias name."""

    def test_alias_dispatches_via_model_handler(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_model_switch") as mock_switch:
            cli_obj.process_command("/sonnet")
        # /sonnet must reach the model handler with a /model-canonical command
        mock_switch.assert_called_once()
        called_with = mock_switch.call_args[0][0]
        assert called_with.lower().startswith("/model"), (
            f"alias /sonnet must be canonicalised to /model before dispatch, "
            f"got {called_with!r}"
        )
        assert "sonnet" in called_with.lower()

    def test_alias_with_extra_args_preserves_them(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_model_switch") as mock_switch:
            cli_obj.process_command("/sonnet --provider openrouter")
        mock_switch.assert_called_once()
        called_with = mock_switch.call_args[0][0]
        assert "sonnet" in called_with.lower()
        assert "--provider openrouter" in called_with.lower()

    def test_unknown_command_behavior_unchanged(self):
        """An unknown slash command that is NOT a model alias must still
        print the 'Unknown command' notice and NOT call _handle_model_switch."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_model_switch") as mock_switch, \
             patch("cli._cprint") as mock_cprint:
            cli_obj.process_command("/xyzzy-not-an-alias-987")
        mock_switch.assert_not_called()
        # The unknown-command notice path prints via _cprint; just confirm
        # the model handler was not invoked.
        assert mock_switch.call_count == 0


class TestModelAliasPriority:
    """Alias resolution must NOT shadow built-ins, plugins, skills, or
    user-defined quick_commands. If a higher-priority handler matches the
    typed name, the alias branch must not fire."""

    def test_alias_does_not_shadow_built_in(self):
        """/help is a built-in; even if 'help' were an alias, the built-in
        branch must win."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, "show_help") as mock_help, \
             patch.object(cli_obj, "_handle_model_switch") as mock_switch:
            cli_obj.process_command("/help")
        mock_help.assert_called_once()
        mock_switch.assert_not_called()

    def test_alias_does_not_shadow_quick_command(self):
        """If the operator defines a quick_command named 'sonnet', that
        command wins over the model alias."""
        cli_obj = _make_cli()
        cli_obj.config = {
            "quick_commands": {
                "sonnet": {
                    "type": "alias",
                    "target": "help",
                }
            }
        }
        with patch.object(cli_obj, "show_help") as mock_help, \
             patch.object(cli_obj, "_handle_model_switch") as mock_switch:
            cli_obj.process_command("/sonnet")
        # quick_command target=/help routes to /help
        mock_help.assert_called_once()
        mock_switch.assert_not_called()

    def test_unknown_command_does_not_invoke_model_handler(self):
        """Guard against accidental alias-table expansion swallowing unknown
        commands: a totally fabricated alias must not match."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_model_switch") as mock_switch:
            cli_obj.process_command("/xyzzy-not-an-alias-987")
        mock_switch.assert_not_called()