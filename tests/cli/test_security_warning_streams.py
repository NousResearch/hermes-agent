"""Security warnings stay visible without corrupting quiet CLI answers."""

from unittest.mock import patch

import pytest

from cli import HermesCLI


@pytest.mark.parametrize("quiet", [True, False])
def test_missing_scanner_warning_preserves_output_streams(quiet, capsys):
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"security": {"tirith_enabled": True}}
    cli.tool_progress_mode = "off" if quiet else "full"
    cli._tirith_security_checked = False
    with (
        patch("tools.tirith_security.ensure_installed", return_value=None) as install,
        patch("tools.tirith_security.is_platform_supported", return_value=True),
    ):
        cli._ensure_tirith_security()
        cli._ensure_tirith_security()

    install.assert_called_once_with(log_failures=False)
    captured = capsys.readouterr()
    diagnostic = captured.err if quiet else captured.out
    assert diagnostic.count("tirith security scanner enabled but not available") == 1
    assert "pattern matching only" in diagnostic
    assert (captured.out if quiet else captured.err) == ""


@pytest.mark.parametrize(
    "installed,supported,enabled",
    [("tirith", True, True), (None, False, True), (None, True, False)],
)
def test_scanner_warning_only_when_security_is_degraded(
    installed, supported, enabled, capsys
):
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"security": {"tirith_enabled": enabled}}
    cli.tool_progress_mode = "off"
    cli._tirith_security_checked = False
    with (
        patch("tools.tirith_security.ensure_installed", return_value=installed),
        patch("tools.tirith_security.is_platform_supported", return_value=supported),
    ):
        cli._ensure_tirith_security()

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
