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
from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter
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


class TestWeComCheckerMatchesAdapter:
    """A "connected" verdict must match the credential pair the adapter actually stores.

    ``hermes gateway setup`` trusts ``is_connected``; if the checker resolves the env pair while
    the adapter reads only ``extra``, the picker reports "connected" for a platform whose
    ``connect()`` then bails out (``wecom_missing_credentials``). Env-first on both sides keeps
    the verdict and the stored pair one source of truth — including where env and ``extra``
    disagree, the case no earlier test combined.
    """

    def test_env_pair_beats_whitespace_only_config_value(self, monkeypatch):
        monkeypatch.setenv("WECOM_BOT_ID", "bot_env")
        monkeypatch.setenv("WECOM_SECRET", "secret_env")
        config = PlatformConfig(enabled=True, extra={"bot_id": "   "})
        assert _wecom._is_connected(config) is True
        adapter = _wecom.WeComAdapter(config)
        assert (adapter._bot_id, adapter._secret) == ("bot_env", "secret_env")

    def test_env_pair_beats_conflicting_config_values(self, monkeypatch):
        monkeypatch.setenv("WECOM_BOT_ID", "bot_env")
        monkeypatch.setenv("WECOM_SECRET", "secret_env")
        config = PlatformConfig(enabled=True, extra={"bot_id": "bot_yaml", "secret": "secret_yaml"})
        assert _wecom._is_connected(config) is True
        adapter = _wecom.WeComAdapter(config)
        assert (adapter._bot_id, adapter._secret) == ("bot_env", "secret_env")

    def test_extra_only_pair_is_stored_unchanged(self):
        config = PlatformConfig(enabled=True, extra={"bot_id": "bot_yaml", "secret": "secret_yaml"})
        assert _wecom._is_connected(config) is True
        adapter = _wecom.WeComAdapter(config)
        assert (adapter._bot_id, adapter._secret) == ("bot_yaml", "secret_yaml")

    def test_half_pair_is_not_connected_and_stores_no_secret(self):
        config = PlatformConfig(enabled=True, extra={"bot_id": "bot_yaml"})
        assert _wecom._is_connected(config) is False
        adapter = _wecom.WeComAdapter(config)
        assert (adapter._bot_id, adapter._secret) == ("bot_yaml", "")


class TestWeComCallbackCheckerMatchesAdapter:
    """Callback mode: ``_normalize_apps`` must build the very pair ``_callback_is_connected`` counts."""

    def test_env_only_pair_builds_the_pair_the_checker_counts(self, monkeypatch):
        monkeypatch.setenv("WECOM_CALLBACK_CORP_ID", "corp_env")
        monkeypatch.setenv("WECOM_CALLBACK_CORP_SECRET", "secret_env")
        monkeypatch.setenv("WECOM_CALLBACK_AGENT_ID", "9001")
        monkeypatch.setenv("WECOM_CALLBACK_TOKEN", "tok_env")
        monkeypatch.setenv("WECOM_CALLBACK_ENCODING_AES_KEY", "aes_env")
        config = _picker_config()
        assert _wecom._callback_is_connected(config) is True
        app = WecomCallbackAdapter(config)._apps[0]
        assert (app["corp_id"], app["corp_secret"]) == ("corp_env", "secret_env")
        assert (app["agent_id"], app["token"], app["encoding_aes_key"]) == ("9001", "tok_env", "aes_env")

    def test_env_pair_beats_conflicting_config_values(self, monkeypatch):
        monkeypatch.setenv("WECOM_CALLBACK_CORP_ID", "corp_env")
        monkeypatch.setenv("WECOM_CALLBACK_CORP_SECRET", "secret_env")
        config = PlatformConfig(enabled=True, extra={"corp_id": "corp_yaml", "corp_secret": "secret_yaml"})
        assert _wecom._callback_is_connected(config) is True
        app = WecomCallbackAdapter(config)._apps[0]
        assert (app["corp_id"], app["corp_secret"]) == ("corp_env", "secret_env")

    def test_extra_only_pair_is_built_unchanged(self):
        config = PlatformConfig(enabled=True, extra={"corp_id": "corp_yaml", "corp_secret": "secret_yaml"})
        assert _wecom._callback_is_connected(config) is True
        assert WecomCallbackAdapter(config)._apps[0]["corp_id"] == "corp_yaml"

    def test_half_pair_is_not_connected_and_builds_no_app(self):
        config = PlatformConfig(enabled=True, extra={"corp_id": "corp_yaml"})
        assert _wecom._callback_is_connected(config) is False
        assert WecomCallbackAdapter(config)._apps == []
