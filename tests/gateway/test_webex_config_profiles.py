"""Webex config and thread lookup follow the owning profile."""

from pathlib import Path
from types import SimpleNamespace

from unittest.mock import MagicMock

import pytest

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


@pytest.mark.asyncio
async def test_secondary_startup_configures_listener_in_own_scope(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home

    home = tmp_path / "hermes"
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants._default_hermes_root_memo", None)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    (profile / ".env").write_text(
        "WEBEX_CONNECTION_MODE=webhook\n"
        "WEBEX_WEBHOOK_SECRET=profile-secret\n"
        "WEBEX_WEBHOOK_PUBLIC_URL=https://bot.example/p/work\n"
    )
    (profile / "config.yaml").write_text(
        "platforms:\n  webex:\n    enabled: true\n    token: yaml-bot-credential\n"
    )
    connected = []

    async def connect(adapter, _platform):
        connected.append((adapter._connection_mode,
                          getattr(adapter, "_shared_listener_profile", None),
                          get_hermes_home()))
        return True

    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", connect)
    assert await runner._start_one_profile_adapters("work", profile, {}) == 1
    assert connected == [("webhook", "work", profile)]
