"""Webex config and thread lookup follow the owning profile."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.session import build_session_key
from plugins.platforms.webex.adapter import WebexAdapter


def test_env_token_preserves_yaml_webhook_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WEBEX_BOT_TOKEN", "bot-token")
    monkeypatch.delenv("WEBEX_CONNECTION_MODE", raising=False)
    (tmp_path / "config.yaml").write_text(
        "platforms:\n  webex:\n    enabled: true\n    extra:\n"
        "      connection_mode: webhook\n"
        "      public_url: https://bot.example.com\n"
        "      secret: signing-secret\n"
    )
    config = load_gateway_config().platforms[Platform("webex")]
    adapter = WebexAdapter(config)
    assert adapter._connection_mode == "webhook"
    assert adapter._public_url == "https://bot.example.com"


def test_active_thread_lookup_follows_shared_bot_route():
    adapter = WebexAdapter(PlatformConfig(token="token"))
    adapter.gateway_runner = SimpleNamespace(
        _profile_name_for_source=lambda source, **kwargs: "satellite"
    )
    source = adapter.build_source(
        chat_id="room", chat_type="group", thread_id="parent", user_id="alice"
    )
    expected_key = build_session_key(source, profile="satellite")
    store = SimpleNamespace(
        lookup_by_session_key=MagicMock(
            side_effect=lambda key: object() if key == expected_key else None
        )
    )
    adapter.set_session_store(store)
    assert adapter._has_active_session_for_thread(
        room_id="room", chat_type="group", parent_id="parent", user_id="alice"
    )
    store.lookup_by_session_key.assert_called_once_with(expected_key)


def test_yaml_token_and_scoped_env_mode_agree_on_listener(monkeypatch):
    from agent import secret_scope
    from plugins.platforms.webex.adapter import _is_port_binding, _validate_config

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    scope = secret_scope.set_secret_scope({
        "WEBEX_CONNECTION_MODE": "webhook",
        "WEBEX_WEBHOOK_SECRET": "signing-secret",
        "WEBEX_WEBHOOK_PUBLIC_URL": "https://bot.example.com/p/secondary",
    })
    try:
        config = PlatformConfig(token="yaml-token")
        adapter = WebexAdapter(config)
        assert adapter._connection_mode == "webhook"
        assert _validate_config(config)
        assert _is_port_binding(config)
    finally:
        secret_scope.reset_secret_scope(scope)
