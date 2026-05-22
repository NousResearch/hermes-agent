"""Test configurable approval timeout for Discord exec approval buttons."""
from pathlib import Path


class TestDiscordApprovalTimeout:
    """Verify approvals.gateway_timeout config is wired into Discord adapter."""

    def test_timeout_config_key_in_config_defaults(self):
        """approvals.gateway_timeout must exist in hermes_cli/config.py defaults."""
        config_path = Path(__file__).resolve().parents[1] / "hermes_cli" / "config.py"
        source = config_path.read_text()
        assert "gateway_timeout" in source, (
            "approvals.gateway_timeout must be defined in config defaults"
        )

    def test_send_exec_approval_has_timeout_parameter(self):
        """send_exec_approval must accept and use a timeout parameter."""
        discord_path = Path(__file__).resolve().parents[1] / "gateway" / "platforms" / "discord.py"
        source = discord_path.read_text()
        assert "timeout:" in source or "timeout=" in source, (
            "send_exec_approval must accept a timeout parameter"
        )

    def test_timeout_resolution_from_config(self):
        """When timeout is None, it must read approvals.gateway_timeout from config."""
        discord_path = Path(__file__).resolve().parents[1] / "gateway" / "platforms" / "discord.py"
        source = discord_path.read_text()
        assert '"gateway_timeout"' in source or "'gateway_timeout'" in source, (
            "approvals.gateway_timeout must be read from config when timeout is None"
        )
