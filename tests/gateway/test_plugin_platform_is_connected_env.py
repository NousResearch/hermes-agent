"""Env-only credentials must read as connected for the plugin platforms whose checker
inspected ``PlatformConfig.extra`` alone (#120870).

``hermes gateway setup`` hands every plugin platform a synthetic ``PlatformConfig(enabled=True)``
— empty ``extra``, no token — so an install whose credentials live in ``.env`` (the shape the
platform wizards themselves write, and what ``gateway/config_env.py`` seeds) was rendered as
"not configured" while the adapter was connected. Feishu, WeCom and WeCom Callback were the three
such checkers.
"""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_feishu = load_plugin_adapter("feishu")
_wecom = load_plugin_adapter("wecom")

_CREDENTIAL_KEYS = (
    "FEISHU_APP_ID", "FEISHU_APP_SECRET",
    "WECOM_BOT_ID", "WECOM_SECRET",
    "WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET",
)


@pytest.fixture(autouse=True)
def _clean_credentials(monkeypatch):
    """Each case starts from an empty env — a sibling test's credentials must not leak in."""
    for key in _CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)


def _picker_config() -> PlatformConfig:
    """The exact config shape ``_platform_status`` builds for a plugin platform."""
    return PlatformConfig(enabled=True)


class TestFeishuIsConnected:

    def test_env_only_credentials_are_connected(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_ID", "cli_env")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret_env")
        assert _feishu._is_connected(_picker_config()) is True

    def test_app_id_without_secret_is_not_connected(self, monkeypatch):
        """``connect()`` rejects a missing secret, so half a credential pair is not "ready"."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_env")
        assert _feishu._is_connected(_picker_config()) is False

    def test_extra_credentials_are_still_connected(self):
        config = PlatformConfig(enabled=True, extra={"app_id": "cli_yaml", "app_secret": "secret_yaml"})
        assert _feishu._is_connected(config) is True


class TestWeComIsConnected:

    def test_env_only_credentials_are_connected(self, monkeypatch):
        monkeypatch.setenv("WECOM_BOT_ID", "bot_env")
        monkeypatch.setenv("WECOM_SECRET", "secret_env")
        assert _wecom._is_connected(_picker_config()) is True

    def test_bot_id_without_secret_is_not_connected(self, monkeypatch):
        monkeypatch.setenv("WECOM_BOT_ID", "bot_env")
        assert _wecom._is_connected(_picker_config()) is False


class TestWeComCallbackIsConnected:

    def test_env_only_credentials_are_connected(self, monkeypatch):
        monkeypatch.setenv("WECOM_CALLBACK_CORP_ID", "corp_env")
        monkeypatch.setenv("WECOM_CALLBACK_CORP_SECRET", "secret_env")
        assert _wecom._callback_is_connected(_picker_config()) is True

    def test_multi_app_block_is_connected(self):
        config = PlatformConfig(enabled=True, extra={"apps": [{"corp_id": "corp_yaml", "corp_secret": "s"}]})
        assert _wecom._callback_is_connected(config) is True
