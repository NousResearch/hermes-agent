"""TELEGRAM_HOME_CHANNEL non-numeric chat-ID startup warning (#13206).

When TELEGRAM_HOME_CHANNEL is set to a @username instead of a numeric chat ID,
_apply_env_overrides still applies it (the Telegram Bot API accepts @username
for public channels/supergroups) but logs a warning so the misconfiguration is
visible before it causes a runtime "chat not found" error.
"""

from __future__ import annotations

import logging

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    _apply_env_overrides,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "TELEGRAM_HOME_CHANNEL_THREAD_ID",
    ):
        monkeypatch.delenv(var, raising=False)


def _config_with_telegram():
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig()
    return config


def test_non_numeric_home_channel_warns_and_still_applies(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "@mychannel")
    config = _config_with_telegram()

    with caplog.at_level(logging.WARNING):
        _apply_env_overrides(config)

    warnings = [
        r for r in caplog.records
        if "TELEGRAM_HOME_CHANNEL" in r.getMessage() and "numeric" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "@mychannel" in warnings[0].getMessage()
    # Warn-only: the value is still applied.
    assert config.platforms[Platform.TELEGRAM].home_channel.chat_id == "@mychannel"


def test_numeric_home_channel_does_not_warn(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "469682876")
    config = _config_with_telegram()

    with caplog.at_level(logging.WARNING):
        _apply_env_overrides(config)

    assert not [
        r for r in caplog.records
        if "TELEGRAM_HOME_CHANNEL" in r.getMessage() and "numeric" in r.getMessage()
    ]
    assert config.platforms[Platform.TELEGRAM].home_channel.chat_id == "469682876"


def test_negative_group_chat_id_does_not_warn(monkeypatch, caplog):
    # Group/channel chat IDs are negative numbers (e.g. -1001234567890) — the
    # leading "-" must not be mistaken for a non-numeric value.
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-1001234567890")
    config = _config_with_telegram()

    with caplog.at_level(logging.WARNING):
        _apply_env_overrides(config)

    assert not [
        r for r in caplog.records
        if "TELEGRAM_HOME_CHANNEL" in r.getMessage() and "numeric" in r.getMessage()
    ]
