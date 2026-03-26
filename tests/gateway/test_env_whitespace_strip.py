"""Tests for whitespace-only env var stripping in gateway config.

A .env entry like ``TELEGRAM_BOT_TOKEN=  `` (spaces only) should NOT
enable the platform — it should be treated as empty/unset. Without
stripping, the truthiness check passes and a whitespace token is stored,
causing a cryptic auth error on connection instead of silently skipping
the platform.
"""

import os
from unittest.mock import patch

from gateway.config import GatewayConfig, _apply_env_overrides, Platform


def _make_config():
    return GatewayConfig.from_dict({})


class TestWhitespaceTokensIgnored:

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "   "}, clear=False)
    def test_whitespace_telegram_token_does_not_enable_platform(self):
        config = _make_config()
        _apply_env_overrides(config)
        assert Platform.TELEGRAM not in config.platforms or not config.platforms[Platform.TELEGRAM].enabled

    @patch.dict(os.environ, {"DISCORD_BOT_TOKEN": " \t "}, clear=False)
    def test_whitespace_discord_token_does_not_enable_platform(self):
        config = _make_config()
        _apply_env_overrides(config)
        assert Platform.DISCORD not in config.platforms or not config.platforms[Platform.DISCORD].enabled

    @patch.dict(os.environ, {"SLACK_BOT_TOKEN": "  "}, clear=False)
    def test_whitespace_slack_token_does_not_enable_platform(self):
        config = _make_config()
        _apply_env_overrides(config)
        assert Platform.SLACK not in config.platforms or not config.platforms[Platform.SLACK].enabled

    @patch.dict(os.environ, {"MATTERMOST_TOKEN": " ", "MATTERMOST_URL": "https://mm.example.com"}, clear=False)
    def test_whitespace_mattermost_token_does_not_enable_platform(self):
        config = _make_config()
        _apply_env_overrides(config)
        assert Platform.MATTERMOST not in config.platforms or not config.platforms[Platform.MATTERMOST].enabled

    @patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "  "}, clear=False)
    def test_whitespace_twilio_sid_does_not_enable_platform(self):
        config = _make_config()
        _apply_env_overrides(config)
        assert Platform.SMS not in config.platforms or not config.platforms[Platform.SMS].enabled


class TestValidTokensStillWork:

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123456:ABC-DEF"}, clear=False)
    def test_real_telegram_token_enables_platform(self):
        config = _make_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].enabled is True
        assert config.platforms[Platform.TELEGRAM].token == "123456:ABC-DEF"

    @patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "MTIz.abc.xyz"}, clear=False)
    def test_real_discord_token_enables_platform(self):
        config = _make_config()
        _apply_env_overrides(config)
        assert config.platforms[Platform.DISCORD].enabled is True
        assert config.platforms[Platform.DISCORD].token == "MTIz.abc.xyz"
