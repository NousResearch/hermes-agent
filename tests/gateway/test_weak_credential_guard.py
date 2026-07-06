"""Tests for gateway weak credential rejection at startup.

Ported from openclaw/openclaw#64586: rejects known-weak placeholder
tokens at gateway startup instead of letting them silently fail
against platform APIs.
"""

import logging

import pytest

from gateway.config import PlatformConfig, Platform, _validate_gateway_config


# ---------------------------------------------------------------------------
# Helper: create a minimal GatewayConfig with one enabled platform
# ---------------------------------------------------------------------------


def _make_gateway_config(platform, token, enabled=True, **extra_kwargs):
    """Create a minimal GatewayConfig-like object for validation testing."""
    from gateway.config import GatewayConfig

    config = GatewayConfig(platforms={})
    pconfig = PlatformConfig(enabled=enabled, token=token, **extra_kwargs)
    config.platforms[platform] = pconfig
    return config


def _validate_and_return(config):
    """Call _validate_gateway_config and return the config (mutated in place)."""
    _validate_gateway_config(config)
    return config


# ---------------------------------------------------------------------------
# Unit tests: platform token placeholder rejection
# ---------------------------------------------------------------------------


class TestPlatformTokenPlaceholderGuard:
    """Verify that _validate_gateway_config disables platforms with placeholder tokens."""

    def test_rejects_triple_asterisk(self, caplog):
        """'***' is the .env.example placeholder — should be rejected."""
        config = _make_gateway_config(Platform.TELEGRAM, "***")
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert config.platforms[Platform.TELEGRAM].enabled is False
        assert "placeholder" in caplog.text.lower()

    def test_rejects_changeme(self, caplog):
        config = _make_gateway_config(Platform.DISCORD, "changeme")
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert config.platforms[Platform.DISCORD].enabled is False

    def test_rejects_your_api_key(self, caplog):
        config = _make_gateway_config(Platform.SLACK, "your_api_key")
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert config.platforms[Platform.SLACK].enabled is False

    def test_rejects_placeholder(self, caplog):
        config = _make_gateway_config(Platform.MATRIX, "placeholder")
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert config.platforms[Platform.MATRIX].enabled is False

    def test_accepts_real_token(self, caplog):
        """A real-looking bot token should pass validation."""
        config = _make_gateway_config(
            Platform.TELEGRAM, "7123456789:AAHdqTcvCH1vGWJxfSeOfSAs0K5PALDsaw"
        )
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert config.platforms[Platform.TELEGRAM].enabled is True
        assert "placeholder" not in caplog.text.lower()

    def test_accepts_empty_token_without_error(self, caplog):
        """Empty tokens get a warning (existing behavior), not a placeholder error."""
        config = _make_gateway_config(Platform.TELEGRAM, "")
        with caplog.at_level(logging.WARNING):
            _validate_and_return(config)
        # Empty token doesn't trigger placeholder rejection — enabled stays True
        # (the existing empty-token warning is separate)
        assert config.platforms[Platform.TELEGRAM].enabled is True

    def test_disabled_platform_not_checked(self, caplog):
        """Disabled platforms should not be validated."""
        config = _make_gateway_config(Platform.TELEGRAM, "***", enabled=False)
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert "placeholder" not in caplog.text.lower()

    def test_rejects_whitespace_padded_placeholder(self, caplog):
        """Whitespace-padded placeholders should still be caught."""
        config = _make_gateway_config(Platform.TELEGRAM, "  ***  ")
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert config.platforms[Platform.TELEGRAM].enabled is False


# ---------------------------------------------------------------------------
# Integration test: API server placeholder key on network-accessible host
# ---------------------------------------------------------------------------


class TestAPIServerPlaceholderKeyGuard:
    """Verify that the API server rejects placeholder keys on network hosts."""

    @pytest.mark.asyncio
    async def test_refuses_wildcard_with_placeholder_key(self):
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "0.0.0.0", "key": "changeme"})
        )
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_refuses_wildcard_with_asterisk_key(self):
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "0.0.0.0", "key": "***"})
        )
        result = await adapter.connect()
        assert result is False

    def test_allows_loopback_with_placeholder_key(self):
        """Loopback with a placeholder key is fine — not network-exposed."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.platforms.base import is_network_accessible

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "key": "changeme"})
        )
        # On loopback the placeholder guard doesn't fire
        assert is_network_accessible(adapter._host) is False
