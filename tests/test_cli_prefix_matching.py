"""Tests for slash command prefix matching in HermesCLI.process_command."""
from unittest.mock import MagicMock, patch
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    return cli_obj


class TestSlashCommandPrefixMatching:
    def test_unique_prefix_dispatches_command(self):
        """/con should dispatch to /config when it uniquely matches."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, 'show_config') as mock_config:
            cli_obj.process_command("/con")
        mock_config.assert_called_once()

    def test_unique_prefix_with_args_dispatches_command(self):
        """/mo with argument should dispatch to /model."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, 'process_command', wraps=cli_obj.process_command):
            with patch("hermes_cli.models.fetch_api_models", return_value=None), \
                 patch("cli.save_config_value"):
                cli_obj.model = "current-model"
                cli_obj.provider = "openrouter"
                cli_obj.base_url = "https://openrouter.ai/api/v1"
                cli_obj.api_key = "test"
                cli_obj._explicit_api_key = None
                cli_obj._explicit_base_url = None
                cli_obj.requested_provider = "openrouter"
                # /mod uniquely matches /model
                result = cli_obj.process_command("/mod")
        assert result is True

    def test_ambiguous_prefix_shows_suggestions(self):
        """/re matches /reset, /retry, /reload-mcp, /reasoning, /rollback — should show suggestions."""
        cli_obj = _make_cli()
        cli_obj.process_command("/re")
        # Should print ambiguous message, not unknown command
        printed = " ".join(str(c) for c in cli_obj.console.print.call_args_list)
        assert "Ambiguous" in printed or "Did you mean" in printed

    def test_unknown_command_shows_error(self):
        """/xyz should show unknown command error."""
        cli_obj = _make_cli()
        cli_obj.process_command("/xyz")
        printed = " ".join(str(c) for c in cli_obj.console.print.call_args_list)
        assert "Unknown command" in printed

    def test_exact_command_still_works(self):
        """/help should still work as exact match."""
        cli_obj = _make_cli()
        with patch.object(cli_obj, 'show_help') as mock_help:
            cli_obj.process_command("/help")
        mock_help.assert_called_once()
