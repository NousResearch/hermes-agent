"""Tests for the /version slash command."""

from types import SimpleNamespace
from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command


def test_version_command_is_registered():
    cmd = resolve_command("version")
    assert cmd is not None
    assert cmd.name == "version"
    assert cmd.category == "Info"
    assert resolve_command("v") is cmd


def test_version_is_gateway_known():
    assert "version" in GATEWAY_KNOWN_COMMANDS
    assert "v" in GATEWAY_KNOWN_COMMANDS


def test_process_command_version_prints_version_info():
    cli_obj = HermesCLI.__new__(HermesCLI)

    with patch("hermes_cli.main._print_version_info") as mock_print:
        assert cli_obj.process_command("/version") is True

    mock_print.assert_called_once_with(check_updates=True)


def test_print_version_info_reports_unknown_update_count(capsys):
    from hermes_cli import main

    with (
        patch(
            "hermes_cli.slash_exec.execute_command",
            return_value=SimpleNamespace(text="Hermes Agent v0.20.0"),
        ),
        patch("hermes_cli.config.detect_install_method", return_value="git"),
        patch("hermes_cli.config.recommended_update_command", return_value="hermes update"),
        patch("hermes_cli.banner.check_for_updates", return_value=-1),
    ):
        main._print_version_info(check_updates=True)

    output = capsys.readouterr().out
    assert "Update available" in output
    assert "commit count unavailable" in output
