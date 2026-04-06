"""Tests for IRC platform adapter."""
import pytest
from unittest.mock import MagicMock, patch

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform Enum
# ---------------------------------------------------------------------------

class TestIRCPlatformEnum:
    def test_irc_enum_exists(self):
        assert Platform.IRC.value == "irc"

    def test_irc_in_platform_list(self):
        platforms = [p.value for p in Platform]
        assert "irc" in platforms


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

class TestIRCConfigLoading:
    def test_irc_disabled_without_server(self, monkeypatch):
        """IRC should not be enabled without IRC_SERVER."""
        monkeypatch.delenv("IRC_SERVER", raising=False)
        monkeypatch.setenv("IRC_NICK", "hermesbot")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.IRC not in config.platforms

    def test_irc_home_channel_from_env(self, monkeypatch):
        """IRC_HOME_CHANNEL should override the default."""
        monkeypatch.setenv("IRC_SERVER", "irc.example.com")
        monkeypatch.setenv("IRC_NICK", "hermesbot")
        monkeypatch.setenv("IRC_CHANNELS", "#bots,#test")
        monkeypatch.setenv("IRC_HOME_CHANNEL", "#test")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.get_home_channel(Platform.IRC)
        assert home is not None
        assert home.chat_id == "#test"

    def test_irc_tls_flag(self, monkeypatch):
        """IRC_USE_TLS should be parsed as boolean."""
        monkeypatch.setenv("IRC_SERVER", "irc.example.com")
        monkeypatch.setenv("IRC_NICK", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "true")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        mc = config.platforms[Platform.IRC]
        assert mc.extra.get("use_tls") is True

    def test_irc_not_enabled_without_creds(self, monkeypatch):
        """IRC should not appear in platforms without credentials."""
        monkeypatch.delenv("IRC_SERVER", raising=False)
        monkeypatch.delenv("IRC_NICK", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.IRC not in config.platforms


# ---------------------------------------------------------------------------
# Adapter Tests
# ---------------------------------------------------------------------------

class TestIRCAdapterFormatMessage:
    """Tests for IRCAdapter.format_message (passthrough, no markdown stripping)."""

    def test_format_message_returns_unchanged(self, monkeypatch):
        """format_message should return content unchanged."""
        monkeypatch.setenv("IRC_SERVER", "irc.example.com")
        monkeypatch.setenv("IRC_NICK", "hermesbot")

        from gateway.platforms.irc import IRCAdapter

        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={"server": "irc.example.com", "nick": "hermesbot"}))
        
        content = "**Hello world**"
        assert adapter.format_message(content) == content

    def test_format_message_preserves_markdown(self, monkeypatch):
        """format_message should preserve markdown syntax."""
        monkeypatch.setenv("IRC_SERVER", "irc.example.com")
        monkeypatch.setenv("IRC_NICK", "hermesbot")

        from gateway.platforms.irc import IRCAdapter

        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={"server": "irc.example.com", "nick": "hermesbot"}))

        content = "**bold** and _italic_ and `code`"
        assert adapter.format_message(content) == content

    def test_format_message_preserves_newlines(self, monkeypatch):
        """format_message should preserve newlines."""
        monkeypatch.setenv("IRC_SERVER", "irc.example.com")
        monkeypatch.setenv("IRC_NICK", "hermesbot")

        from gateway.platforms.irc import IRCAdapter

        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={"server": "irc.example.com", "nick": "hermesbot"}))

        content = "line 1\nline 2\nline 3"
        assert adapter.format_message(content) == content

    def test_format_message_empty_input(self, monkeypatch):
        """format_message with empty input returns empty string."""
        monkeypatch.setenv("IRC_SERVER", "irc.example.com")
        monkeypatch.setenv("IRC_NICK", "hermesbot")

        from gateway.platforms.irc import IRCAdapter

        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={"server": "irc.example.com", "nick": "hermesbot"}))

        assert adapter.format_message("") == ""


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIRCIntegration:
    def test_irc_not_in_connected_platforms_when_disabled(self, monkeypatch):
        """IRC should not appear in get_connected_platforms when not configured."""
        monkeypatch.delenv("IRC_SERVER", raising=False)
        monkeypatch.delenv("IRC_NICK", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        connected = config.get_connected_platforms()
        assert "irc" not in connected
