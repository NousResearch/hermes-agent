from unittest.mock import MagicMock, patch


def test_limits_command_registered_for_cli_and_gateway():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("limits")

    assert cmd is not None
    assert cmd.name == "limits"
    assert cmd.category == "Info"
    assert cmd.cli_only is False
    assert cmd.gateway_only is False
    assert resolve_command("codex-limits").name == "limits"
    assert resolve_command("climits").name == "limits"


def test_limits_cli_handler_prints_pretty_output_on_live_console():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.console = MagicMock()
    cli._app = object()
    live_console = MagicMock()
    state = {
        "source": {"provider": "app_server", "captured_at": "2026-02-01T00:00:00Z"},
        "rate_limits": [
            {
                "name": "Rate limits remaining",
                "limit_id": "codex",
                "five_h": {"remaining_percent": 75, "reset_at": "2026-02-02T02:40:00Z"},
                "week": {"remaining_percent": 95, "reset_at": "2026-02-07T21:33:00Z"},
            }
        ],
    }

    with patch("cli.ChatConsole", return_value=live_console), \
         patch("agent.codex_limits.get_codex_limits", return_value=state):
        cli._handle_limits_command("/limits")

    printed = "\n".join(str(call.args[0]) for call in live_console.print.call_args_list if call.args)
    assert "Codex limits remaining (app_server)" in printed
    assert "Rate limits remaining: 5h 75% reset" in printed
    cli.console.print.assert_not_called()


def test_limits_cli_handler_supports_json_flag():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.console = MagicMock()
    cli._app = None
    state = {"source": {"provider": "app_server"}, "rate_limits": []}

    with patch("agent.codex_limits.get_codex_limits", return_value=state):
        cli._handle_limits_command("/limits --json --provider wham")

    printed = "\n".join(str(call.args[0]) for call in cli.console.print.call_args_list if call.args)
    assert '"provider": "app_server"' in printed
