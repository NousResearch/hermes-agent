"""Tests for the TUI /gateway status panel platform listing (#99788).

The panel used to enumerate a hardcoded subset of four bot-token platforms,
so platforms like Signal or A2A that were configured and connected were
reported as absent. The listing must include every platform present in the
loaded gateway config while keeping the bot-token setup hints.
"""

from cli import HermesCLI
from gateway.config import GatewayConfig, Platform, PlatformConfig


def _config_with(platforms):
    config = GatewayConfig()
    for platform, pconfig in platforms.items():
        config.platforms[platform] = pconfig
    return config


def _run_panel(monkeypatch, tmp_path, config):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._show_gateway_status()


def test_gateway_status_lists_configured_first_party_platforms(
    monkeypatch, tmp_path, capsys
):
    """Signal and A2A configured+enabled must be listed as Enabled (#99788)."""
    config = _config_with({
        Platform.SIGNAL: PlatformConfig(enabled=True),
        Platform("a2a"): PlatformConfig(enabled=True),
    })

    _run_panel(monkeypatch, tmp_path, config)
    out = capsys.readouterr().out

    assert "Signal" in out and "Enabled" in out
    assert "A2A" in out
    assert "Signal      Not configured" not in out


def test_gateway_status_keeps_bot_token_setup_hints(monkeypatch, tmp_path, capsys):
    """Unconfigured bot-token platforms keep their env-var setup hints."""
    _run_panel(monkeypatch, tmp_path, GatewayConfig())
    out = capsys.readouterr().out

    assert "Telegram" in out and "Not configured (TELEGRAM_BOT_TOKEN)" in out
    assert "Not configured (DISCORD_BOT_TOKEN)" in out
    assert "Not configured (SLACK_BOT_TOKEN)" in out
    assert "Not configured (WHATSAPP_ENABLED)" in out


def test_gateway_status_reports_disabled_extra_platform(monkeypatch, tmp_path, capsys):
    """A configured-but-disabled non-bot-token platform shows Disabled, not a hint."""
    config = _config_with({Platform.SIGNAL: PlatformConfig(enabled=False)})

    _run_panel(monkeypatch, tmp_path, config)
    out = capsys.readouterr().out

    assert "Signal       Disabled" in out
    assert "Not configured (SIGNAL" not in out


def test_gateway_status_skips_local_platform(monkeypatch, tmp_path, capsys):
    """The LOCAL pseudo-platform never appears in the messaging listing."""
    config = _config_with({Platform.LOCAL: PlatformConfig(enabled=True)})

    _run_panel(monkeypatch, tmp_path, config)
    out = capsys.readouterr().out

    assert "Local" not in out
