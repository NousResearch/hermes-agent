"""Every CLI registry command and alias must reach a built-in handler.

Regression guard for #74594. Unlike the historical dispatch-table snapshot,
derive coverage from the live registry so newly advertised commands are checked
automatically, including /whoami. Only handler bodies are stubbed: the real
registry resolution and process_command dispatch run without interactive or
filesystem side effects.
"""

from unittest.mock import patch

import pytest

from cli import HermesCLI
from hermes_cli.commands import COMMAND_REGISTRY


CLI_SPELLINGS = [
    pytest.param(command.name, spelling, id=spelling)
    for command in COMMAND_REGISTRY
    if not command.gateway_only
    for spelling in (command.name, *command.aliases)
]


@pytest.mark.parametrize("canonical,spelling", CLI_SPELLINGS)
@pytest.mark.parametrize("handler_result", [None, False], ids=["continue", "exit"])
def test_registered_cli_command_reaches_handler(canonical, spelling, handler_result):
    cli = HermesCLI.__new__(HermesCLI)
    entry = HermesCLI._slash_handler(canonical)
    assert entry is not None, f"/{canonical} is advertised but has no CLI handler"
    method_name, pass_arg = entry
    assert callable(getattr(HermesCLI, method_name))
    # Mixed-case arguments must survive alias resolution unchanged.
    command = f"/{spelling} KeepThisCase"

    with (
        patch.object(cli, method_name, autospec=True, return_value=handler_result) as handler,
        patch.object(cli, "_process_unregistered_slash") as fallback,
        patch("hermes_cli.plugins.fire_pre_command_hook"),
    ):
        result = cli.process_command(command)

    handler.assert_called_once_with(*((command,) if pass_arg else ()))
    fallback.assert_not_called()
    assert result is (handler_result is not False)


def test_unregistered_command_reaches_fallback():
    """Negative control: the fallback spy detects a real dispatch miss."""
    cli = HermesCLI.__new__(HermesCLI)
    command = "/definitely-not-a-registered-command KeepThisCase"
    with patch.object(cli, "_process_unregistered_slash", return_value=False) as fallback:
        assert cli.process_command(command) is False
    fallback.assert_called_once_with(command, command.lower())
