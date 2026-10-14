"""Weixin setup reuses profile-local logins and persists settings the gateway reads."""

import os
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.platforms import weixin
from hermes_cli import config, gateway


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in list(os.environ):
        if name.startswith(("WEIXIN_", "GATEWAY_ALLOW")):
            monkeypatch.delenv(name)
    config.invalidate_env_cache()
    qr = AsyncMock(side_effect=AssertionError("Saved logins must not require QR login"))
    monkeypatch.setattr(weixin, "qr_login", qr)
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda question, default=True: True)
    return tmp_path, qr


def test_saved_login_can_be_configured_without_rescanning(setup_env, monkeypatch):
    home, qr = setup_env
    (home / "config.yaml").write_text(
        "# Keep my settings\nmodel:\n  default: my-model\nplatforms:\n"
        "  weixin:\n    extra:\n      send_chunk_delay_seconds: 2.5 # tuned delay\n",
        encoding="utf-8",
    )
    weixin.save_weixin_account(str(home), account_id="saved-bot", token="saved-token",
                              base_url=weixin.ILINK_BASE_URL, user_id="scanner-user")
    weixin.save_weixin_account(str(home / "other-profile"), account_id="foreign-bot", token="foreign-token",
                              base_url=weixin.ILINK_BASE_URL)
    choices_seen = []

    def choose(question, choices, default=0):
        choices_seen.extend(choices)
        if "direct messages" in question or "group chats" in question:
            return 2
        return default

    def prompt(question, default=None, password=False):
        if "Allowed Weixin user IDs" in question:
            assert default == "scanner-user"
            return "scanner-user, another-user"
        if "Allowed group chat IDs" in question:
            return "allowed-group"
        return default or ""

    monkeypatch.setattr(gateway, "prompt_choice", choose)
    monkeypatch.setattr(gateway, "prompt", prompt)
    gateway._configure_platform({"key": "weixin"})

    qr.assert_not_awaited()
    assert not any("foreign-bot" in choice for choice in choices_seen)
    loaded = load_gateway_config().platforms[Platform.WEIXIN]
    adapter = weixin.WeixinAdapter(loaded)
    assert loaded.enabled
    assert adapter._account_id == "saved-bot"
    assert adapter._token == "saved-token"
    assert adapter._is_dm_allowed("another-user")
    assert not adapter._is_dm_allowed("stranger")
    assert adapter._is_group_allowed("allowed-group")
    assert not adapter._is_group_allowed("other-group")
    assert loaded.home_channel.chat_id == "scanner-user"
    assert loaded.extra["send_chunk_delay_seconds"] == 2.5
    assert gateway._platform_status({"key": "weixin", "token_var": "WEIXIN_ACCOUNT_ID"}) == "configured"
    raw = config.read_raw_config()
    assert raw["model"]["default"] == "my-model"
    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert "# Keep my settings" in text and "# tuned delay" in text
    assert config.load_env()["WEIXIN_TOKEN"] == "saved-token"
    assert not any(name.startswith("WEIXIN_") and name != "WEIXIN_TOKEN" for name in config.load_env())


def test_setup_preserves_current_voice_progress_and_quote_preferences(setup_env, monkeypatch):
    home, qr = setup_env
    config.save_env_value("WEIXIN_TOKEN", "current-token")
    config.write_platform_config_field("weixin", "enabled", True, raw=True)
    config.write_platform_config_field("weixin", "extra", {
        "account_id": "current-bot", "use_platform_transcription": False, "reply_progress_messages": False,
        "quote_cache": {"enabled": False, "max_messages_per_account": 1234},
    }, raw=True)
    defaults_seen = {}

    def yes_no(question, default=True):
        defaults_seen[question] = default
        return default

    monkeypatch.setattr(gateway, "prompt_yes_no", yes_no)
    monkeypatch.setattr(gateway, "prompt_choice", lambda question, choices, default=0: default)
    monkeypatch.setattr(gateway, "prompt", lambda question, default=None, password=False: default or "")
    gateway._configure_platform({"key": "weixin"})
    loaded = load_gateway_config().platforms[Platform.WEIXIN]
    adapter = weixin.WeixinAdapter(loaded)
    assert adapter._use_platform_transcription is False
    assert adapter.native_task_cards_enabled() is False
    assert adapter._quote_store.enabled is False
    assert adapter._quote_store.policy["max_messages_per_account"] == 1234
    assert any("voice transcription" in question and default is False for question, default in defaults_seen.items())
    qr.assert_not_awaited()


def test_new_qr_login_uses_current_application_identity(setup_env, monkeypatch):
    home, qr = setup_env
    config.write_platform_config_field("weixin", "extra", {"bot_agent": "Hermes/9.9 (Desktop)", "route_tag": 42}, raw=True)
    qr.side_effect = None
    qr.return_value = {
        "account_id": "new-bot", "token": "new-token", "base_url": weixin.ILINK_BASE_URL, "user_id": "scanner-user",
    }
    monkeypatch.setattr(weixin, "check_weixin_requirements", lambda: True)
    monkeypatch.setattr(gateway, "prompt_choice", lambda question, choices, default=0: default)
    monkeypatch.setattr(gateway, "prompt", lambda question, default=None, password=False: default or "")

    gateway._configure_platform({"key": "weixin"})

    qr.assert_awaited_once_with(str(home), bot_agent="Hermes/9.9 (Desktop)", route_tag=42)
    loaded = load_gateway_config().platforms[Platform.WEIXIN]
    assert loaded.extra["bot_agent"] == "Hermes/9.9 (Desktop)"
    assert loaded.extra["route_tag"] == 42
    assert loaded.extra["account_id"] == "new-bot"
    assert loaded.token == "new-token"


@pytest.mark.parametrize("legacy_env", [False, True])
@pytest.mark.parametrize("new_policy", ["open", "disabled"])
def test_reconfiguration_keeps_defaults_and_changes_effective_policy(setup_env, monkeypatch, legacy_env, new_policy):
    home, qr = setup_env
    extra = {"account_id": "current-bot", "dm_policy": "allowlist", "allow_from": ["current-user"],
             "group_policy": "allowlist", "group_allow_from": ["current-group"],
             "base_url": weixin.ILINK_BASE_URL, "cdn_base_url": weixin.WEIXIN_CDN_BASE_URL}
    config.save_env_value("WEIXIN_TOKEN", "current-token")
    if legacy_env:
        for name, value in {"WEIXIN_ACCOUNT_ID": "current-bot", "WEIXIN_DM_POLICY": "allowlist",
                            "WEIXIN_ALLOWED_USERS": "current-user", "WEIXIN_GROUP_POLICY": "allowlist",
                            "WEIXIN_GROUP_ALLOWED_USERS": "current-group",
                            "WEIXIN_HOME_CHANNEL": "current-user", "WEIXIN_ALLOW_ALL_USERS": "false"}.items():
            config.save_env_value(name, value)
    else:
        config.write_platform_config_field("weixin", "enabled", True, raw=True)
        config.write_platform_config_field("weixin", "extra", extra, raw=True)
        config.write_platform_config_field("weixin", "home_channel", {
            "platform": "weixin", "chat_id": "current-user", "name": "My home"}, raw=True)

    def choose(question, choices, default=0):
        if "direct messages" in question:
            assert default == 2
            return 1 if new_policy == "open" else 3
        if "group chats" in question:
            assert default == 2
        return default

    def prompt(question, default=None, password=False):
        if "Allowed group chat IDs" in question:
            assert default == "current-group"
        if "Home channel ID" in question:
            assert default == "current-user"
            return "new-home"
        return default or ""

    monkeypatch.setattr(gateway, "prompt_choice", choose)
    monkeypatch.setattr(gateway, "prompt", prompt)
    gateway._configure_platform({"key": "weixin"})

    qr.assert_not_awaited()
    loaded = load_gateway_config().platforms[Platform.WEIXIN]
    adapter = weixin.WeixinAdapter(loaded)
    assert adapter._account_id == "current-bot"
    assert adapter._token == "current-token"
    assert adapter._dm_policy == new_policy
    assert adapter._is_dm_allowed("stranger") is (new_policy == "open")
    assert adapter._is_group_allowed("current-group")
    assert not adapter._is_group_allowed("other-group")
    assert loaded.home_channel.chat_id == "new-home"
    assert not any(name.startswith("WEIXIN_") and name != "WEIXIN_TOKEN" for name in config.load_env())
