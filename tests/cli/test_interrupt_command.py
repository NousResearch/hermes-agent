"""Tests for the /interrupt CLI command."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestInterruptCommandRegistry(unittest.TestCase):
    def test_interrupt_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = [c.name for c in COMMAND_REGISTRY]
        assert "interrupt" in names

    def test_interrupt_not_cli_only(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        cmd = next(c for c in COMMAND_REGISTRY if c.name == "interrupt")
        assert not cmd.cli_only

    def test_interrupt_alias_i(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        cmd = next(c for c in COMMAND_REGISTRY if c.name == "interrupt")
        assert "i" in cmd.aliases

    def test_interrupt_in_gateway_known_commands(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "interrupt" in GATEWAY_KNOWN_COMMANDS
        assert "i" in GATEWAY_KNOWN_COMMANDS


class TestHandleInterruptCommand(unittest.TestCase):
    def test_interrupt_with_active_agent(self):
        cli_mod = _import_cli()
        mock_agent = MagicMock()
        stub = SimpleNamespace(agent=mock_agent)

        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_interrupt_command(stub)

        mock_agent.interrupt.assert_called_once()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Interrupted" in printed

    def test_interrupt_with_no_agent(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(agent=None)

        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_interrupt_command(stub)

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "No active run" in printed

    def test_interrupt_with_agent_without_interrupt_method(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(agent=SimpleNamespace())  # no interrupt method

        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_interrupt_command(stub)

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "No active run" in printed


class TestResolveInterruptAlias(unittest.TestCase):
    def test_resolve_i_to_interrupt(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("/i")
        assert cmd.name == "interrupt"

    def test_resolve_interrupt(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("/interrupt")
        assert cmd.name == "interrupt"


if __name__ == "__main__":
    unittest.main()
