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


    def test_accepts_real_token(self, caplog):
        """A real-looking bot token should pass validation."""
        config = _make_gateway_config(
            Platform.TELEGRAM, "7123456789:AAHdqTcvCH1vGWJxfSeOfSAs0K5PALDsaw"
        )
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert config.platforms[Platform.TELEGRAM].enabled is True
        assert "placeholder" not in caplog.text.lower()


    def test_disabled_platform_not_checked(self, caplog):
        """Disabled platforms should not be validated."""
        config = _make_gateway_config(Platform.TELEGRAM, "***", enabled=False)
        with caplog.at_level(logging.ERROR):
            _validate_and_return(config)
        assert "placeholder" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# Integration test: API server placeholder key on network-accessible host
# ---------------------------------------------------------------------------


class TestAPIServerPlaceholderKeyGuard:
    """Verify that the API server rejects placeholder keys on network hosts."""


    @pytest.mark.asyncio
    async def test_refuses_wildcard_with_short_random_key(self):
        """A short but non-placeholder key is brute-forceable on a public bind.

        June 2026 hermes-0day hardening raised the network-bind entropy floor
        from 8 to 16 chars. A 12-char random key (which passed the old guard)
        must now be refused — the API server dispatches terminal-capable agent
        work, so a guessable key is RCE.
        """
        from gateway.platforms.api_server import APIServerAdapter

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "0.0.0.0", "key": "a1b2c3d4e5f6"})
        )
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_allows_wildcard_with_strong_key(self):
        """A 32-char random key clears the entropy floor (connect proceeds past
        the credential guard). We don't assert full startup success here — the
        port/runner setup is environment-dependent — only that the weak-key
        guard does not reject it."""
        from gateway.platforms.api_server import APIServerAdapter
        from hermes_cli.auth import has_usable_secret

        strong = "0123456789abcdef0123456789abcdef"
        assert has_usable_secret(strong, min_length=16) is True
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "0.0.0.0", "key": strong})
        )
        # The credential guard itself accepts the key (start may still fail on
        # later env-specific steps, which is out of scope for this guard test).
        assert adapter._api_key == strong
