"""Tests for the startup allowlist warning check in gateway/run.py."""

import os
from unittest.mock import patch

from gateway.config import Platform, PlatformConfig


def _enabled(platform: Platform) -> dict[Platform, PlatformConfig]:
    return {platform: PlatformConfig(enabled=True)}


def _would_warn(platforms=None):
    """Replicate the startup allowlist warning logic. Returns True if warning fires."""
    _any_allowlist = any(
        os.getenv(v)
        for v in (
            "TELEGRAM_ALLOWED_USERS",
            "DISCORD_ALLOWED_USERS",
            "WHATSAPP_ALLOWED_USERS",
            "SLACK_ALLOWED_USERS",
            "SIGNAL_ALLOWED_USERS",
            "SIGNAL_GROUP_ALLOWED_USERS",
            "EMAIL_ALLOWED_USERS",
            "SMS_ALLOWED_USERS",
            "MATTERMOST_ALLOWED_USERS",
            "MATRIX_ALLOWED_USERS",
            "DINGTALK_ALLOWED_USERS",
            "FEISHU_ALLOWED_USERS",
            "WECOM_ALLOWED_USERS",
            "GATEWAY_ALLOWED_USERS",
        )
    )
    _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes") or any(
        os.getenv(v, "").lower() in ("true", "1", "yes")
        for v in (
            "TELEGRAM_ALLOW_ALL_USERS",
            "DISCORD_ALLOW_ALL_USERS",
            "WHATSAPP_ALLOW_ALL_USERS",
            "SLACK_ALLOW_ALL_USERS",
            "SIGNAL_ALLOW_ALL_USERS",
            "EMAIL_ALLOW_ALL_USERS",
            "SMS_ALLOW_ALL_USERS",
            "MATTERMOST_ALLOW_ALL_USERS",
            "MATRIX_ALLOW_ALL_USERS",
            "DINGTALK_ALLOW_ALL_USERS",
            "FEISHU_ALLOW_ALL_USERS",
            "WECOM_ALLOW_ALL_USERS",
            "WECOM_CALLBACK_ALLOW_ALL_USERS",
            "WEIXIN_ALLOW_ALL_USERS",
            "BLUEBUBBLES_ALLOW_ALL_USERS",
            "QQ_ALLOW_ALL_USERS",
        )
    )
    auth_exempt_platforms = {
        Platform.API_SERVER,
        Platform.HOMEASSISTANT,
        Platform.WEBHOOK,
        Platform.GMAIL_PUSH,
    }
    has_user_authenticated_platforms = any(
        cfg.enabled and platform not in auth_exempt_platforms
        for platform, cfg in (platforms or {}).items()
    )
    return has_user_authenticated_platforms and not _any_allowlist and not _allow_all


class TestAllowlistStartupCheck:
    def test_no_enabled_platforms_does_not_warn(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _would_warn() is False

    def test_enabled_user_authenticated_platform_warns_without_allowlists(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _would_warn(_enabled(Platform.TELEGRAM)) is True

    def test_signal_group_allowed_users_suppresses_warning(self):
        with patch.dict(os.environ, {"SIGNAL_GROUP_ALLOWED_USERS": "user1"}, clear=True):
            assert _would_warn(_enabled(Platform.SIGNAL)) is False

    def test_telegram_allow_all_users_suppresses_warning(self):
        with patch.dict(os.environ, {"TELEGRAM_ALLOW_ALL_USERS": "true"}, clear=True):
            assert _would_warn(_enabled(Platform.TELEGRAM)) is False

    def test_gateway_allow_all_users_suppresses_warning(self):
        with patch.dict(os.environ, {"GATEWAY_ALLOW_ALL_USERS": "yes"}, clear=True):
            assert _would_warn(_enabled(Platform.TELEGRAM)) is False

    def test_only_gmail_push_enabled_does_not_warn(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _would_warn(_enabled(Platform.GMAIL_PUSH)) is False

    def test_gmail_push_plus_user_authenticated_platform_still_warns(self):
        platforms = {
            Platform.GMAIL_PUSH: PlatformConfig(enabled=True),
            Platform.TELEGRAM: PlatformConfig(enabled=True),
        }
        with patch.dict(os.environ, {}, clear=True):
            assert _would_warn(platforms) is True
