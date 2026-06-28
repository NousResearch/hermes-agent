from unittest.mock import MagicMock, patch

from cli import HermesCLI


def test_cli_proactive_command_delegates_to_shared_handler():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._console_print = MagicMock()

    with patch(
        "hermes_cli.proactive_cmd.handle_proactive_command",
        return_value="proactive output",
    ) as handler:
        cli_obj._handle_proactive_command('/proactive --days 14 --source cli')

    handler.assert_called_once_with("--days 14 --source cli")
    cli_obj._console_print.assert_called_once_with("proactive output")
