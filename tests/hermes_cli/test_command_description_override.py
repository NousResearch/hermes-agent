"""Tests for command description overrides via config.yaml (issue #13107)."""

import pytest
from unittest.mock import patch


class TestResolveDescriptionOverrides:
    """Tests for _resolve_description_overrides()."""

    def test_returns_empty_dict_when_no_config(self):
        from hermes_cli.commands import _resolve_description_overrides
        with patch("hermes_cli.config.read_raw_config", side_effect=Exception("no config")):
            result = _resolve_description_overrides()
            assert result == {}

    def test_returns_overrides_from_config(self):
        from hermes_cli.commands import _resolve_description_overrides
        with patch("hermes_cli.config.read_raw_config", return_value={
            "commands": {"descriptions": {"new": "开始新会话", "help": "帮助"}}
        }):
            result = _resolve_description_overrides()
            assert result == {"new": "开始新会话", "help": "帮助"}

    def test_returns_empty_when_descriptions_is_none(self):
        from hermes_cli.commands import _resolve_description_overrides
        with patch("hermes_cli.config.read_raw_config", return_value={
            "commands": {"descriptions": None}
        }):
            result = _resolve_description_overrides()
            assert result == {}

    def test_returns_empty_when_commands_missing(self):
        from hermes_cli.commands import _resolve_description_overrides
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            result = _resolve_description_overrides()
            assert result == {}

    def test_coerces_non_string_values(self):
        from hermes_cli.commands import _resolve_description_overrides
        with patch("hermes_cli.config.read_raw_config", return_value={
            "commands": {"descriptions": {42: True}}
        }):
            result = _resolve_description_overrides()
            assert result == {"42": "True"}


class TestTelegramBotCommandsOverrides:
    """Tests for telegram_bot_commands() with description overrides."""

    def test_override_appears_in_telegram_commands(self):
        from hermes_cli.commands import telegram_bot_commands

        with patch("hermes_cli.config.read_raw_config", return_value={
            "commands": {"descriptions": {"new": "开始新会话"}}
        }):
            cmds = telegram_bot_commands()
            new_entry = next(((n, d) for n, d in cmds if n == "new"), None)
            assert new_entry is not None
            assert new_entry[1] == "开始新会话"

    def test_default_description_when_no_override(self):
        from hermes_cli.commands import telegram_bot_commands, COMMAND_REGISTRY

        default_desc = next(c.description for c in COMMAND_REGISTRY if c.name == "help")

        with patch("hermes_cli.config.read_raw_config", return_value={}):
            cmds = telegram_bot_commands()
            help_entry = next(((n, d) for n, d in cmds if n == "help"), None)
            assert help_entry is not None
            assert help_entry[1] == default_desc

    def test_override_does_not_affect_unrelated_commands(self):
        from hermes_cli.commands import telegram_bot_commands, COMMAND_REGISTRY

        default_help_desc = next(c.description for c in COMMAND_REGISTRY if c.name == "help")

        with patch("hermes_cli.config.read_raw_config", return_value={
            "commands": {"descriptions": {"new": "custom new"}}
        }):
            cmds = telegram_bot_commands()
            help_entry = next(((n, d) for n, d in cmds if n == "help"), None)
            assert help_entry is not None
            assert help_entry[1] == default_help_desc


class TestGatewayHelpLinesOverrides:
    """Tests for gateway_help_lines() with description overrides."""

    def test_override_appears_in_help_lines(self):
        from hermes_cli.commands import gateway_help_lines

        with patch("hermes_cli.config.read_raw_config", return_value={
            "commands": {"descriptions": {"new": "开始新会话"}}
        }):
            lines = gateway_help_lines()
            new_line = next((l for l in lines if l.startswith("`/new")), None)
            assert new_line is not None
            assert "开始新会话" in new_line

    def test_default_description_in_help_when_no_override(self):
        from hermes_cli.commands import gateway_help_lines, COMMAND_REGISTRY

        default_desc = next(c.description for c in COMMAND_REGISTRY if c.name == "help")

        with patch("hermes_cli.config.read_raw_config", return_value={}):
            lines = gateway_help_lines()
            help_line = next((l for l in lines if l.startswith("`/help")), None)
            assert help_line is not None
            assert default_desc in help_line
