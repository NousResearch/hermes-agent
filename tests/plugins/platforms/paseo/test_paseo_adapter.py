import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig


def test_paseo_env_enablement_is_opt_in(monkeypatch):
    from plugins.platforms.paseo.adapter import _env_enablement

    for name in (
        "PASEO_GATEWAY_ENABLED",
        "PASEO_GATEWAY_HOST",
        "PASEO_GATEWAY_PORT",
        "PASEO_GATEWAY_TOKEN",
        "PASEO_HOME_CHANNEL",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _env_enablement() is None

    monkeypatch.setenv("PASEO_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("PASEO_GATEWAY_HOST", "127.0.0.1")
    monkeypatch.setenv("PASEO_GATEWAY_PORT", "8767")
    monkeypatch.setenv("PASEO_GATEWAY_TOKEN", "secret-token")
    monkeypatch.setenv("PASEO_HOME_CHANNEL", "paseo-default")

    seed = _env_enablement()

    assert seed == {
        "host": "127.0.0.1",
        "port": 8767,
        "token": "secret-token",
        "home_channel": {"chat_id": "paseo-default", "name": "Paseo"},
    }


@pytest.mark.asyncio
async def test_paseo_adapter_http_message_round_trip():
    from plugins.platforms.paseo.adapter import PaseoAdapter

    adapter = PaseoAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "token": "secret-token"},
        )
    )

    async def handler(event):
        assert event.text == "hello Hermes"
        assert event.source.platform.value == "paseo"
        assert event.source.chat_id == "chat-1"
        assert event.source.user_id == "user-1"
        return "hello Paseo"

    adapter.set_message_handler(handler)

    client = TestClient(TestServer(adapter.build_app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer secret-token"},
            json={
                "text": "hello Hermes",
                "chat_id": "chat-1",
                "user_id": "user-1",
                "message_id": "msg-1",
            },
        )
        assert resp.status == 202
        body = await resp.json()
        assert body["ok"] is True
        assert body["chat_id"] == "chat-1"

        for _ in range(20):
            out_resp = await client.get(
                "/v1/messages/chat-1",
                headers={"Authorization": "Bearer secret-token"},
            )
            assert out_resp.status == 200
            messages = (await out_resp.json())["messages"]
            if messages:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("adapter did not buffer outbound response")
    finally:
        await client.close()

    assert messages[-1]["content"] == "hello Paseo"
    assert messages[-1]["chat_id"] == "chat-1"


@pytest.mark.asyncio
async def test_paseo_adapter_rejects_unauthorized_requests():
    from plugins.platforms.paseo.adapter import PaseoAdapter

    adapter = PaseoAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "token": "secret-token"},
        )
    )
    client = TestClient(TestServer(adapter.build_app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/messages",
            json={"text": "hello", "chat_id": "chat-1", "user_id": "user-1"},
        )

        assert resp.status == 401
        assert await resp.json() == {"error": "unauthorized"}
    finally:
        await client.close()


def test_paseo_plugin_registers_platform_entry(monkeypatch):
    from gateway.platform_registry import platform_registry
    from plugins.platforms.paseo.adapter import register

    class _Manifest:
        name = "paseo-platform"

    class _Manager:
        _plugin_platform_names = set()

    class _Context:
        manifest = _Manifest()
        _manager = _Manager()

        def register_platform(self, **kwargs):
            from gateway.platform_registry import PlatformEntry
            kwargs.setdefault("source", "plugin")
            platform_registry.register(PlatformEntry(**kwargs))
            self._manager._plugin_platform_names.add(kwargs["name"])

    platform_registry.unregister("paseo")
    try:
        register(_Context())
        entry = platform_registry.get("paseo")
        assert entry is not None
        assert entry.label == "Paseo"
        assert entry.allow_all_env == "PASEO_ALLOW_ALL_USERS"
        assert entry.allowed_users_env == "PASEO_ALLOWED_USERS"
        assert entry.cron_deliver_env_var == "PASEO_HOME_CHANNEL"
    finally:
        platform_registry.unregister("paseo")
