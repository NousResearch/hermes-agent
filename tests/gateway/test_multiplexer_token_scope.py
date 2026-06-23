"""Tests for #51029 — Multiplexer: platform tokens must resolve through the
profile secret scope, not raw os.getenv.

When ``gateway.multiplex_profiles`` is enabled, ``_apply_env_overrides`` used
to call ``os.getenv("TELEGRAM_BOT_TOKEN")`` (and similar) which reads the
DEFAULT profile's process-global os.environ instead of the per-profile secret
scope.  The fix replaces those calls with ``get_secret(...)`` which:

  1. Reads from the active ``set_secret_scope`` mapping (profile-scoped) when
     in a multiplexed turn.
  2. Falls back to ``os.environ`` transparently when no scope is active
     (single-profile / non-multiplex setup).
  3. Raises ``UnscopedSecretError`` (fail-closed) when multiplex mode is active
     but no scope is installed — so stray unscoped reads surface loudly rather
     than silently leaking another profile's token.
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from agent import secret_scope as ss
from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    _apply_env_overrides,
)


@pytest.fixture(autouse=True)
def _reset_multiplex():
    """Ensure multiplex mode is off and no scope is installed before/after each test."""
    ss.set_multiplex_active(False)
    tok = ss.set_secret_scope(None)
    yield
    ss.set_multiplex_active(False)
    ss.reset_secret_scope(tok)


def _empty_config() -> GatewayConfig:
    return GatewayConfig()


# ---------------------------------------------------------------------------
# 1. Single-profile (non-multiplex) behaviour is unchanged — os.environ read
# ---------------------------------------------------------------------------

class TestSingleProfileFallback:
    """get_secret falls back to os.environ when no scope is active (default)."""

    def test_telegram_token_from_environ(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg-token-from-env")
        config = _empty_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].token == "tg-token-from-env"

    def test_discord_token_from_environ(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-token-from-env")
        config = _empty_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.DISCORD].token == "discord-token-from-env"

    def test_slack_token_from_environ(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "slack-token-from-env")
        config = _empty_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.SLACK].token == "slack-token-from-env"

    def test_matrix_token_from_environ(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "matrix-token-from-env")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.com")
        config = _empty_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].token == "matrix-token-from-env"

    def test_whatsapp_cloud_tokens_from_environ(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "12345")
        monkeypatch.setenv("WHATSAPP_CLOUD_ACCESS_TOKEN", "wa-cloud-token")
        config = _empty_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.WHATSAPP_CLOUD].extra["access_token"] == "wa-cloud-token"
        assert config.platforms[Platform.WHATSAPP_CLOUD].extra["phone_number_id"] == "12345"

    def test_mattermost_token_from_environ(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_TOKEN", "mm-token-from-env")
        monkeypatch.setenv("MATTERMOST_URL", "https://mattermost.example.com")
        config = _empty_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.MATTERMOST].token == "mm-token-from-env"

    def test_weixin_token_from_environ(self, monkeypatch):
        monkeypatch.setenv("WEIXIN_TOKEN", "weixin-token-from-env")
        config = _empty_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.WEIXIN].token == "weixin-token-from-env"


# ---------------------------------------------------------------------------
# 2. Profile-scoped read: get_secret reads from the installed scope, not
#    os.environ, so a secondary profile gets its OWN token.
# ---------------------------------------------------------------------------

class TestProfileScopedRead:
    """_apply_env_overrides reads tokens from the active profile secret scope."""

    def test_telegram_token_from_scope_not_environ(self, monkeypatch):
        """The profile scope token wins over os.environ token."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "default-profile-tg-token")
        tok = ss.set_secret_scope({"TELEGRAM_BOT_TOKEN": "secondary-profile-tg-token"})
        try:
            config = _empty_config()
            _apply_env_overrides(config)
            # Must read from scope, NOT from the os.environ default-profile token
            assert config.platforms[Platform.TELEGRAM].token == "secondary-profile-tg-token"
        finally:
            ss.reset_secret_scope(tok)

    def test_discord_token_from_scope_not_environ(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "default-discord-token")
        tok = ss.set_secret_scope({"DISCORD_BOT_TOKEN": "secondary-discord-token"})
        try:
            config = _empty_config()
            _apply_env_overrides(config)
            assert config.platforms[Platform.DISCORD].token == "secondary-discord-token"
        finally:
            ss.reset_secret_scope(tok)

    def test_slack_token_from_scope_not_environ(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "default-slack-token")
        tok = ss.set_secret_scope({"SLACK_BOT_TOKEN": "scoped-slack-token"})
        try:
            config = _empty_config()
            _apply_env_overrides(config)
            assert config.platforms[Platform.SLACK].token == "scoped-slack-token"
        finally:
            ss.reset_secret_scope(tok)

    def test_whatsapp_cloud_token_from_scope_not_environ(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_CLOUD_ACCESS_TOKEN", "default-wa-token")
        monkeypatch.setenv("WHATSAPP_CLOUD_PHONE_NUMBER_ID", "default-phone-id")
        tok = ss.set_secret_scope({
            "WHATSAPP_CLOUD_ACCESS_TOKEN": "scoped-wa-token",
            "WHATSAPP_CLOUD_PHONE_NUMBER_ID": "scoped-phone-id",
        })
        try:
            config = _empty_config()
            _apply_env_overrides(config)
            assert config.platforms[Platform.WHATSAPP_CLOUD].extra["access_token"] == "scoped-wa-token"
            assert config.platforms[Platform.WHATSAPP_CLOUD].extra["phone_number_id"] == "scoped-phone-id"
        finally:
            ss.reset_secret_scope(tok)

    def test_matrix_token_from_scope_not_environ(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "default-matrix-token")
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://default.matrix.org")
        tok = ss.set_secret_scope({
            "MATRIX_ACCESS_TOKEN": "scoped-matrix-token",
            "MATRIX_HOMESERVER": "https://secondary.matrix.org",
        })
        try:
            config = _empty_config()
            _apply_env_overrides(config)
            assert config.platforms[Platform.MATRIX].token == "scoped-matrix-token"
        finally:
            ss.reset_secret_scope(tok)

    def test_two_profiles_get_different_tokens(self, monkeypatch):
        """Simulates two successive multiplexed turns: each profile's config
        must only see its own token."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

        # Profile A
        tok_a = ss.set_secret_scope({"TELEGRAM_BOT_TOKEN": "tg-token-A"})
        try:
            config_a = _empty_config()
            _apply_env_overrides(config_a)
        finally:
            ss.reset_secret_scope(tok_a)

        # Profile B
        tok_b = ss.set_secret_scope({"TELEGRAM_BOT_TOKEN": "tg-token-B"})
        try:
            config_b = _empty_config()
            _apply_env_overrides(config_b)
        finally:
            ss.reset_secret_scope(tok_b)

        assert config_a.platforms[Platform.TELEGRAM].token == "tg-token-A"
        assert config_b.platforms[Platform.TELEGRAM].token == "tg-token-B"
        # Crucially: profile A did NOT get profile B's token
        assert config_a.platforms[Platform.TELEGRAM].token != config_b.platforms[Platform.TELEGRAM].token


# ---------------------------------------------------------------------------
# 3. Fail-closed: when multiplex is active and no scope is set, get_secret
#    raises UnscopedSecretError instead of silently falling back to os.environ.
# ---------------------------------------------------------------------------

class TestFailClosedUnscoped:
    """When multiplex mode is on and no scope is installed, platform token reads
    raise UnscopedSecretError rather than leaking os.environ values."""

    def test_telegram_fails_closed_without_scope(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "leaked-default-token")
        ss.set_multiplex_active(True)
        # No scope installed — must raise, not silently return the env token
        with pytest.raises(ss.UnscopedSecretError, match="TELEGRAM_BOT_TOKEN"):
            config = _empty_config()
            _apply_env_overrides(config)

    def test_discord_fails_closed_without_scope(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "leaked-discord-token")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            config = _empty_config()
            _apply_env_overrides(config)

    def test_slack_fails_closed_without_scope(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "leaked-slack-token")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            config = _empty_config()
            _apply_env_overrides(config)


# ---------------------------------------------------------------------------
# 4. Verify get_secret is actually called (not os.getenv) — patch-level proof
# ---------------------------------------------------------------------------

class TestGetSecretIsCalledNotOsGetenv:
    """Patch get_secret at the call site to confirm _apply_env_overrides routes
    through it, not through a direct os.getenv call."""

    def test_get_secret_called_for_telegram_token(self, monkeypatch):
        """get_secret is invoked for TELEGRAM_BOT_TOKEN."""
        call_log = []
        original_get_secret = ss.get_secret

        def spy_get_secret(name, default=None):
            call_log.append(name)
            return original_get_secret(name, default)

        with patch("gateway.config.get_secret", side_effect=spy_get_secret):
            config = _empty_config()
            _apply_env_overrides(config)

        assert "TELEGRAM_BOT_TOKEN" in call_log, (
            "get_secret was never called with TELEGRAM_BOT_TOKEN — "
            "_apply_env_overrides may still be using os.getenv directly"
        )

    def test_get_secret_called_for_discord_token(self, monkeypatch):
        call_log = []
        original_get_secret = ss.get_secret

        def spy_get_secret(name, default=None):
            call_log.append(name)
            return original_get_secret(name, default)

        with patch("gateway.config.get_secret", side_effect=spy_get_secret):
            config = _empty_config()
            _apply_env_overrides(config)

        assert "DISCORD_BOT_TOKEN" in call_log

    def test_get_secret_called_for_all_primary_tokens(self, monkeypatch):
        """All primary platform credentials must go through get_secret."""
        call_log = []
        original_get_secret = ss.get_secret

        def spy_get_secret(name, default=None):
            call_log.append(name)
            return original_get_secret(name, default)

        with patch("gateway.config.get_secret", side_effect=spy_get_secret):
            config = _empty_config()
            _apply_env_overrides(config)

        required_credentials = [
            "TELEGRAM_BOT_TOKEN",
            "DISCORD_BOT_TOKEN",
            "SLACK_BOT_TOKEN",
            "MATRIX_ACCESS_TOKEN",
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            "MATTERMOST_TOKEN",
            "WEIXIN_TOKEN",
            "TWILIO_ACCOUNT_SID",
            "HASS_TOKEN",
        ]
        missing = [c for c in required_credentials if c not in call_log]
        assert not missing, (
            f"These credentials were NOT routed through get_secret: {missing}. "
            "They may still be using os.getenv and would leak in multiplex mode."
        )
