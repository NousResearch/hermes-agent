"""Tests for the /version slash command."""

from unittest.mock import patch

import hermes_cli.banner as banner

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


def test_print_version_info_renders_no_count_update(capsys):
    """Package-version updates should not be rendered as a commit count."""
    from hermes_cli.main import _print_version_info

    with patch("hermes_cli.banner.check_for_updates", return_value=banner.UPDATE_AVAILABLE_NO_COUNT), \
         patch("hermes_cli.config.recommended_update_command", return_value="hermes update"):
        _print_version_info(check_updates=True)

    output = capsys.readouterr().out
    assert "Update available" in output
    assert "commit behind" not in output
    assert "commits behind" not in output
    assert "run 'hermes update'" in output
