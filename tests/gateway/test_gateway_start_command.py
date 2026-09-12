"""Behavioral regression coverage for the gateway command shown by status (#98819)."""

from types import SimpleNamespace

from hermes_cli.cli_info_mixin import CLIInfoMixin


def test_gateway_status_advertises_installed_entrypoint(monkeypatch, capsys):
    """The status view points users to executable installed gateway commands."""
    config = SimpleNamespace(platforms={}, get_home_channel=lambda _platform: None)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("hermes_constants.display_hermes_home", lambda: "/tmp/hermes")

    CLIInfoMixin()._show_gateway_status()

    output = capsys.readouterr().out
    assert "To start the gateway:" in output
    assert "hermes gateway run" in output
    assert "hermes gateway start  (service mode)" in output
    assert "python cli.py --gateway" not in output
