"""Tests for the registry-backed Bale platform plugin."""

import inspect

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType, SendResult
from plugins.platforms.bale.adapter import (
    BaleAdapter,
    _apply_yaml_config,
    _env_enablement,
    _standalone_send,
    check_requirements,
    is_connected,
    register,
)


def _config(token: str = "test-token") -> PlatformConfig:
    config = PlatformConfig(enabled=True)
    config.extra["token"] = token
    return config


def test_bale_uses_dynamic_platform_identity():
    adapter = BaleAdapter(_config())
    assert adapter.platform == Platform("bale")
    assert adapter.platform != Platform.TELEGRAM
    assert adapter.api_base == "https://tapi.bale.ai/bot"
    assert adapter.file_base == "https://tapi.bale.ai/file/bot"


def test_requirement_probe_is_passive_and_connection_check_accepts_config(monkeypatch):
    monkeypatch.delenv("BALE_BOT_TOKEN", raising=False)
    assert check_requirements() is True
    assert len(inspect.signature(is_connected).parameters) == 1
    assert is_connected(_config()) is True
    assert is_connected(PlatformConfig(enabled=True)) is False


def test_register_exposes_registry_owned_hooks():
    captured = {}

    class Context:
        def register_platform(self, **kwargs):
            captured.update(kwargs)

    register(Context())
    assert captured["name"] == "bale"
    assert captured["required_env"] == ["BALE_BOT_TOKEN"]
    assert captured["allowed_users_env"] == "BALE_ALLOWED_USERS"
    assert captured["allow_all_env"] == "BALE_ALLOW_ALL_USERS"
    assert captured["cron_deliver_env_var"] == "BALE_HOME_CHANNEL"
    assert captured["env_enablement_fn"] is _env_enablement
    assert captured["standalone_sender_fn"] is _standalone_send
    assert inspect.iscoroutinefunction(captured["standalone_sender_fn"])


def test_yaml_bridge_maps_auth_and_runtime_settings(monkeypatch):
    for key in (
        "BALE_BOT_TOKEN",
        "BALE_ALLOWED_USERS",
        "BALE_ALLOW_ALL_USERS",
        "BALE_REQUIRE_MENTION",
        "BALE_HOME_CHANNEL",
    ):
        monkeypatch.delenv(key, raising=False)

    extra = _apply_yaml_config({}, {
        "bot_token": "yaml-token",
        "allowed_users": ["123", "456"],
        "allow_all_users": False,
        "require_mention": True,
        "home_channel": "-10042",
        "poll_timeout": 30,
    })

    assert extra["token"] == "yaml-token"
    assert extra["require_mention"] is True
    assert extra["home_channel"] == "-10042"
    assert extra["poll_timeout"] == 30
    assert __import__("os").environ["BALE_BOT_TOKEN"] == "yaml-token"
    assert __import__("os").environ["BALE_ALLOWED_USERS"] == "123,456"
    assert __import__("os").environ["BALE_ALLOW_ALL_USERS"] == "false"


def test_env_enablement_seeds_plugin_config(monkeypatch):
    monkeypatch.setenv("BALE_BOT_TOKEN", "env-token")
    monkeypatch.setenv("BALE_HOME_CHANNEL", "777")
    monkeypatch.setenv("BALE_REQUIRE_MENTION", "false")
    monkeypatch.setenv("BALE_POLL_TIMEOUT", "20")

    seed = _env_enablement()
    assert seed["token"] == "env-token"
    assert seed["require_mention"] is False
    assert seed["poll_timeout"] == 20
    assert seed["home_channel"]["chat_id"] == "777"


@pytest.mark.asyncio
async def test_send_uses_bale_bot_api_contract():
    adapter = BaleAdapter(_config())
    calls = []

    async def fake_api(method, payload=None, **kwargs):
        calls.append((method, payload))
        return {"ok": True, "result": {"message_id": 91}}

    adapter._api = fake_api
    result = await adapter.send("1234", "hello")
    assert result.success is True
    assert result.message_id == "91"
    assert calls == [("sendMessage", {"chat_id": "1234", "text": "hello"})]


@pytest.mark.asyncio
async def test_inbound_update_keeps_bale_source_identity():
    adapter = BaleAdapter(_config())
    adapter._bot_id = "99"
    adapter._bot_username = "hermesbot"
    events = []

    async def handle(event):
        events.append(event)

    adapter.handle_message = handle
    await adapter._dispatch_update({
        "update_id": 10,
        "message": {
            "message_id": 11,
            "date": 1_700_000_000,
            "from": {"id": 7, "first_name": "Ali"},
            "chat": {"id": 42, "type": "private"},
            "text": "hello",
        },
    })

    assert len(events) == 1
    event = events[0]
    assert event.source.platform == Platform("bale")
    assert event.source.chat_id == "42"
    assert event.source.user_id == "7"
    assert event.message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_group_require_mention_is_enforced_and_stripped():
    adapter = BaleAdapter(_config())
    adapter._bot_id = "99"
    adapter._bot_username = "hermesbot"
    events = []

    async def handle(event):
        events.append(event)

    adapter.handle_message = handle
    base = {
        "date": 1_700_000_000,
        "from": {"id": 7, "first_name": "Ali"},
        "chat": {"id": -42, "type": "group", "title": "Test"},
    }
    await adapter._dispatch_update({"update_id": 1, "message": {**base, "message_id": 1, "text": "hello"}})
    await adapter._dispatch_update({"update_id": 2, "message": {**base, "message_id": 2, "text": "@hermesbot hello"}})
    assert len(events) == 1
    assert events[0].text == "hello"


@pytest.mark.asyncio
async def test_standalone_sender_contract(monkeypatch):
    class FakeClient:
        async def aclose(self):
            pass

    async def fake_send(self, chat_id, content, reply_to=None, metadata=None):
        assert chat_id == "55"
        assert content == "hello"
        return SendResult(success=True, message_id="88")

    monkeypatch.setattr(BaleAdapter, "_client_factory", lambda self: FakeClient())
    monkeypatch.setattr(BaleAdapter, "send", fake_send)
    result = await _standalone_send(_config(), "55", "hello")
    assert result == {"success": True, "message_id": "88", "platform": "bale"}
