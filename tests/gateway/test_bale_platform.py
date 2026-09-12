"""Tests for the bundled Bale platform plugin.

No test contacts Bale; transport construction and authorization are exercised
with mocks so the suite is deterministic and safe for contributor CI.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


bale = load_plugin_adapter("bale")


def test_api_base_normalizes_for_python_telegram_bot(monkeypatch):
    monkeypatch.delenv("BALE_API_BASE_URL", raising=False)
    assert bale._api_base() == "https://tapi.bale.ai/bot"

    monkeypatch.setenv("BALE_API_BASE_URL", "https://example.test/api/")
    assert bale._api_base() == "https://example.test/api/bot"

    monkeypatch.setenv("BALE_API_BASE_URL", "https://example.test/bot")
    assert bale._api_base() == "https://example.test/bot"


def test_adapter_keeps_bale_identity_and_authorization_separate(monkeypatch):
    monkeypatch.setenv("BALE_BOT_TOKEN", "bale-token")
    monkeypatch.setenv("BALE_ALLOWED_USERS", "101, 202")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")

    config = PlatformConfig(enabled=True)
    adapter = bale.BaleAdapter(config)

    assert adapter.platform == Platform("bale")
    assert config.token == "bale-token"
    assert config.extra["base_url"] == "https://tapi.bale.ai/bot"
    assert config.extra["proxy_env_var"] == "BALE_PROXY"
    assert config.extra["allow_from"] == ["101", "202"]
    assert adapter._is_callback_user_authorized("101") is True
    assert adapter._is_callback_user_authorized("999") is False


def test_message_auth_source_is_bale(monkeypatch):
    monkeypatch.setenv("BALE_ALLOWED_USERS", "101")
    adapter = bale.BaleAdapter(PlatformConfig(enabled=True, token="token"))
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=101, username="mirhadi", full_name="Mirhadi"),
        sender_chat=None,
        chat=SimpleNamespace(id=101, type="private", is_forum=False),
        message_thread_id=None,
        is_topic_message=False,
    )

    source = adapter._source_from_message_for_auth(message)

    assert source.platform == Platform("bale")
    assert source.chat_type == "dm"
    assert source.user_id == "101"


def test_registration_includes_standalone_and_auth_hooks():
    ctx = MagicMock()

    bale.register(ctx)

    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "bale"
    assert kwargs["required_env"] == ["BALE_BOT_TOKEN"]
    assert kwargs["allowed_users_env"] == "BALE_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "BALE_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "BALE_HOME_CHANNEL"
    assert callable(kwargs["standalone_sender_fn"])


def test_multiplex_profiles_do_not_borrow_bale_credentials_or_allowlists(monkeypatch):
    from agent import secret_scope

    monkeypatch.setenv("BALE_BOT_TOKEN", "default-profile-token")
    monkeypatch.setenv("BALE_ALLOWED_USERS", "999")
    monkeypatch.setenv("BALE_ALLOW_ALL_USERS", "true")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    scope = secret_scope.set_secret_scope({})
    try:
        config = PlatformConfig(enabled=True)
        adapter = bale.BaleAdapter(config)
        assert not config.token
        assert not bale._is_connected(config)
        assert not adapter._is_callback_user_authorized("999")
        assert not adapter._telegram_auth_env_configured()
    finally:
        secret_scope.reset_secret_scope(scope)


def test_standalone_send_uses_current_profile_token(monkeypatch):
    from agent import secret_scope

    monkeypatch.setenv("BALE_BOT_TOKEN", "default-profile-token")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    sender = AsyncMock(return_value={"success": True})
    scope = secret_scope.set_secret_scope({"BALE_BOT_TOKEN": "secondary-token"})
    try:
        with patch("tools.send_message_senders._send_telegram", sender):
            asyncio.run(bale._standalone_send(PlatformConfig(enabled=True), "101", "hello"))
        assert sender.await_args.args[0] == "secondary-token"
    finally:
        secret_scope.reset_secret_scope(scope)


def test_standalone_send_uses_bale_endpoint(monkeypatch):
    monkeypatch.setenv("BALE_BOT_TOKEN", "bale-token")
    sender = AsyncMock(return_value={"success": True, "message_id": "7"})

    with patch("tools.send_message_senders._send_telegram", sender):
        result = asyncio.run(
            bale._standalone_send(
                PlatformConfig(enabled=True),
                "101",
                "hello",
                thread_id="3",
            )
        )

    assert result["success"] is True
    sender.assert_awaited_once()
    kwargs = sender.await_args.kwargs
    assert kwargs["base_url"] == "https://tapi.bale.ai/bot"
    assert kwargs["base_file_url"] == "https://tapi.bale.ai/bot"
    assert kwargs["proxy_env_var"] == "BALE_PROXY"
    assert kwargs["thread_id"] == "3"
