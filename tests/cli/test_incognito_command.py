"""Tests for the /incognito CLI command."""

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


class TestHandleIncognitoCommand:
    def _make_cli(self, incognito=False, with_agent=True):
        return SimpleNamespace(
            _incognito_mode=incognito,
            agent=MagicMock() if with_agent else None,
        )

    def test_status_shows_off_by_default(self):
        cli_mod = _import_cli()
        stub = self._make_cli(incognito=False)

        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_incognito_command(stub, "/incognito status")

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Incognito mode" in printed
        assert "OFF" in printed

    def test_on_toggles_mode_and_disables_persistence(self):
        cli_mod = _import_cli()
        stub = self._make_cli(incognito=False)

        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_incognito_command(stub, "/incognito on")

        assert stub._incognito_mode is True
        assert stub.agent.persist_session is False
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Incognito ON" in printed
        assert "NOT be persisted" in printed

    def test_off_toggles_mode_and_restores_persistence(self):
        cli_mod = _import_cli()
        stub = self._make_cli(incognito=True)
        stub.agent.persist_session = False

        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_incognito_command(stub, "/incognito off")

        assert stub._incognito_mode is False
        assert stub.agent.persist_session is True
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Incognito OFF" in printed
        assert "persistence resumed" in printed

    def test_invalid_argument_shows_usage(self):
        cli_mod = _import_cli()
        stub = self._make_cli(incognito=False)

        with patch.object(cli_mod, "_cprint") as mock_cprint:
            cli_mod.HermesCLI._handle_incognito_command(stub, "/incognito maybe")

        assert stub._incognito_mode is False
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        assert "Usage: /incognito [on|off|status]" in printed
