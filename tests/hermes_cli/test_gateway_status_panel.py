"""Gateway status panel platform rows (TUI `/gateway status` display)."""

from unittest.mock import patch

from cli import format_gateway_platform_rows
from gateway.config import GatewayConfig, Platform, PlatformConfig, SessionResetPolicy


def _config_with(platforms):
    return GatewayConfig(
        platforms=platforms,
        default_reset_policy=SessionResetPolicy(),
    )


def test_signal_and_a2a_only_configuration_renders_connected(tmp_path):
    """Signal+A2A-only setups show real runtime state, not four 'Not configured' rows."""
    a2a = Platform("a2a")  # bundled plugin platform
    config = _config_with(
        {
            Platform.SIGNAL: PlatformConfig(enabled=True),
            a2a: PlatformConfig(enabled=True),
        }
    )

    runtime_snapshot = {
        "platforms": {
            "signal": {"state": "connected", "needs_attention": False},
            "a2a": {"state": "connected", "needs_attention": False},
        }
    }

    with patch("gateway.status.read_runtime_status", return_value=runtime_snapshot):
        rows = format_gateway_platform_rows(config)

    text = "\n".join(rows)
    assert "Signal" in text
    assert "A2A" in text
    # No platform reports "Not configured": both are enabled and connected.
    assert "Not configured" not in text
    # Enabled rows keep the checkmark.
    assert sum(1 for r in rows if "✓" in r) == 2


def test_runtime_state_decorates_enabled_rows(tmp_path):
    """A retrying platform shows its live state instead of a bare 'Enabled'."""
    config = _config_with({Platform.SIGNAL: PlatformConfig(enabled=True)})
    runtime_snapshot = {
        "platforms": {
            "signal": {"state": "retrying", "needs_attention": True},
        }
    }
    with patch("gateway.status.read_runtime_status", return_value=runtime_snapshot):
        rows = format_gateway_platform_rows(config)
    text = "\n".join(rows)
    assert "Signal" in text
    assert "retrying" in text
    assert "needs attention" in text
    assert "✓" not in text


def test_classic_token_platforms_keep_env_hints(tmp_path):
    """Unconfigured classic platforms still show their credential env var."""
    config = _config_with({})
    with patch("gateway.status.read_runtime_status", return_value=None):
        rows = format_gateway_platform_rows(config)
    text = "\n".join(rows)
    assert "TELEGRAM_BOT_TOKEN" in text
    assert "DISCORD_BOT_TOKEN" in text
    assert "SLACK_BOT_TOKEN" in text
    assert "WHATSAPP_ENABLED" in text


def test_runtime_only_platform_appended(tmp_path):
    """A platform the gateway runs but config misses still gets a row."""
    config = _config_with({})
    runtime_snapshot = {
        "platforms": {
            "work:signal": {"state": "connected", "needs_attention": False},
        }
    }
    with patch("gateway.status.read_runtime_status", return_value=runtime_snapshot):
        rows = format_gateway_platform_rows(config)
    text = "\n".join(rows)
    assert "Signal" in text
    assert "connected" in text
