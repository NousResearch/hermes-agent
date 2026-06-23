"""Tests for the startup allowlist warning check in gateway/run.py."""

import asyncio
import logging
import os
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def _would_warn():
    """Replicate the startup allowlist warning logic. Returns True if warning fires."""
    _any_allowlist = any(
        os.getenv(v)
        for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                   "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                   "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
                   "EMAIL_ALLOWED_USERS",
                   "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
                   "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS", "FEISHU_ALLOWED_USERS", "WECOM_ALLOWED_USERS",
                   "GATEWAY_ALLOWED_USERS")
    )
    _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"} or any(
        os.getenv(v, "").lower() in {"true", "1", "yes"}
        for v in ("TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
                   "WHATSAPP_ALLOW_ALL_USERS", "SLACK_ALLOW_ALL_USERS",
                   "SIGNAL_ALLOW_ALL_USERS", "EMAIL_ALLOW_ALL_USERS",
                   "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
                   "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS", "FEISHU_ALLOW_ALL_USERS", "WECOM_ALLOW_ALL_USERS")
    )
    return not _any_allowlist and not _allow_all


class TestAllowlistStartupCheck:

    def test_no_config_emits_warning(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _would_warn() is True

    def test_signal_group_allowed_users_suppresses_warning(self):
        with patch.dict(os.environ, {"SIGNAL_GROUP_ALLOWED_USERS": "user1"}, clear=True):
            assert _would_warn() is False

    def test_telegram_allow_all_users_suppresses_warning(self):
        with patch.dict(os.environ, {"TELEGRAM_ALLOW_ALL_USERS": "true"}, clear=True):
            assert _would_warn() is False

    def test_gateway_allow_all_users_suppresses_warning(self):
        with patch.dict(os.environ, {"GATEWAY_ALLOW_ALL_USERS": "yes"}, clear=True):
            assert _would_warn() is False


def _clear_allowlist_env(monkeypatch):
    prefixes = (
        "TELEGRAM",
        "DISCORD",
        "WHATSAPP",
        "WHATSAPP_CLOUD",
        "SLACK",
        "SIGNAL",
        "EMAIL",
        "SMS",
        "MATTERMOST",
        "MATRIX",
        "DINGTALK",
        "FEISHU",
        "WECOM",
        "WECOM_CALLBACK",
        "WEIXIN",
        "BLUEBUBBLES",
        "QQ",
        "YUANBAO",
        "GATEWAY",
    )
    suffixes = ("ALLOWED_USERS", "ALLOW_ALL_USERS", "GROUP_ALLOWED_USERS", "GROUP_ALLOWED_CHATS")
    for prefix in prefixes:
        for suffix in suffixes:
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)


def _quiet_startup_after_allowlist_check(monkeypatch):
    async def _zero_async(*args, **kwargs):
        return 0

    async def _none_async(*args, **kwargs):
        return None

    monkeypatch.setattr(GatewayRunner, "_create_adapter", lambda self, platform, config: None)
    monkeypatch.setattr(GatewayRunner, "_start_secondary_profile_adapters", _zero_async)
    monkeypatch.setattr(GatewayRunner, "_send_update_notification", lambda self: _none_async())
    monkeypatch.setattr(GatewayRunner, "_send_restart_notification", lambda self: _none_async())
    monkeypatch.setattr(GatewayRunner, "_finish_startup_restore", lambda self: _none_async())
    monkeypatch.setattr(GatewayRunner, "_schedule_resume_pending_sessions", lambda self: None)
    monkeypatch.setattr(GatewayRunner, "_schedule_update_notification_watch", lambda self: None)


def _run_start_and_collect_messages(cfg, caplog, monkeypatch):
    _quiet_startup_after_allowlist_check(monkeypatch)
    runner = GatewayRunner(cfg)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(runner.start())
    return [record.getMessage() for record in caplog.records]


def test_api_and_webhook_only_do_not_emit_user_allowlist_warning(tmp_path, monkeypatch, caplog):
    _clear_allowlist_env(monkeypatch)
    cfg = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        platforms={
            Platform.API_SERVER: PlatformConfig(enabled=True),
            Platform.WEBHOOK: PlatformConfig(enabled=True),
            Platform.MSGRAPH_WEBHOOK: PlatformConfig(enabled=True),
        },
    )
    messages = _run_start_and_collect_messages(cfg, caplog, monkeypatch)
    assert not any("No user allowlists configured" in msg for msg in messages)


def test_messaging_platform_still_emits_user_allowlist_warning(tmp_path, monkeypatch, caplog):
    _clear_allowlist_env(monkeypatch)
    cfg = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        platforms={Platform.DISCORD: PlatformConfig(enabled=True)},
    )
    messages = _run_start_and_collect_messages(cfg, caplog, monkeypatch)
    assert any("No user allowlists configured" in msg for msg in messages)
