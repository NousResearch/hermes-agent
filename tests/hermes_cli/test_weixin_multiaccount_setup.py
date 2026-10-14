"""Adding and switching iLink logins preserves the other bot's effective settings."""

from hermes_cli import config
from hermes_cli.gateway_setup_weixin import _save_weixin_settings
from gateway.config import Platform, load_gateway_config
from gateway.platforms.weixin_group import account_configs


def test_add_switch_and_disable_preserve_existing_account(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("WEIXIN_TOKEN", raising=False)
    config.invalidate_env_cache()
    first = {"account_id": "first", "token": "first-secret", "base_url": "https://example.test", "user_id": "user1"}
    second = {"account_id": "second", "token": "second-secret", "base_url": "https://example.test", "user_id": "user2"}
    _save_weixin_settings(first, {"dm_policy": "allowlist", "allow_from": ["user1"]}, None)
    _save_weixin_settings(second, {"dm_policy": "pairing", "reply_progress_messages": False}, None, set_default=False)
    loaded = load_gateway_config().platforms[Platform.WEIXIN]
    effective = account_configs(loaded, tmp_path)
    assert set(effective) == {"first", "second"}
    assert loaded.extra["default_account"] == "first"
    assert effective["first"].extra["allow_from"] == ["user1"]
    assert effective["second"].extra["dm_policy"] == "pairing"
    _save_weixin_settings(second, {"dm_policy": "open", "allow_all_users": True}, None)
    loaded = load_gateway_config().platforms[Platform.WEIXIN]
    effective = account_configs(loaded, tmp_path)
    assert loaded.extra["default_account"] == "second"
    assert effective["first"].token == "first-secret"
    assert effective["first"].extra["dm_policy"] == "allowlist"
    _save_weixin_settings(first, {}, None, enabled=False, set_default=False)
    loaded = load_gateway_config().platforms[Platform.WEIXIN]
    assert set(account_configs(loaded, tmp_path)) == {"second"}
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "first-secret" not in text and "second-secret" not in text
