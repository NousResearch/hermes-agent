"""Regression tests for Hermes CLI slash command aliases."""

from hermes_cli.commands import COMMANDS, COMMANDS_BY_CATEGORY, resolve_command


def test_models_alias_resolves_to_model_command():
    command = resolve_command("models")

    assert command is not None
    assert command.name == "model"
    assert "models" in command.aliases


def test_models_alias_is_listed_for_cli_help_surfaces():
    assert "/models" in COMMANDS
    assert "alias for /model" in COMMANDS["/models"]
    assert "/models" in COMMANDS_BY_CATEGORY["Configuration"]
