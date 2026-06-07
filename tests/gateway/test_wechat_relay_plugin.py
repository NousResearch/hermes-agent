from __future__ import annotations

import asyncio
import json
import socket

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from plugins.platforms.wechat_relay.adapter import (
    DEFAULT_PORT,
    WeChatRelayAdapter,
    _env_enablement,
    is_connected,
    register,
    validate_config,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeCtx:
    def __init__(self):
        self.kwargs: dict[str, object] = {}

    def register_platform(self, **kw):
        self.kwargs = kw


class TestRegister:
    def test_registers_independent_wechat_relay_platform(self):
        ctx = _FakeCtx()
        register(ctx)
        assert ctx.kwargs is not None
        assert ctx.kwargs["name"] == "wechat_relay"
        assert ctx.kwargs["label"] == "WeChat Relay"
        assert ctx.kwargs["allow_update_command"] is False
        hint = str(ctx.kwargs["platform_hint"])
        assert "WeChat Relay / 微信中转" in hint
        assert "built-in Weixin plugin" in hint

    def test_register_does_not_claim_weixin_identity(self):
        ctx = _FakeCtx()
        register(ctx)
        assert ctx.kwargs["name"] != "weixin"
        assert ctx.kwargs["allowed_users_env"] == "WECHAT_RELAY_ALLOWED_LOGICAL_KEYS"
        assert ctx.kwargs["allow_all_env"] == "WECHAT_RELAY_ALLOW_ALL_USERS"
        assert ctx.kwargs["cron_deliver_env_var"] == "WECHAT_RELAY_HOME_CHANNEL"


class TestConfig:
    def test_env_enablement_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("WECHAT_RELAY_ENABLED", raising=False)
        assert _env_enablement() is None

    def test_env_enablement_seeds_dedicated_port(self, monkeypatch):
        monkeypatch.setenv("WECHAT_RELAY_ENABLED", "true")
        monkeypatch.delenv("WECHAT_RELAY_PORT", raising=False)
        result = _env_enablement()
        assert result["port"] == DEFAULT_PORT
        assert result["host"] == "127.0.0.1"

    def test_validate_rejects_openclaw_mobile_shell_port_8787(self):
        cfg = PlatformConfig(enabled=True, extra={"port": 8787})
        assert validate_config(cfg) is False
        assert is_connected(cfg) is False

    def test_adapter_init_refuses_port_8787(self):
        cfg = PlatformConfig(enabled=True, extra={"port": 8787})
        with pytest.raises(ValueError, match="refuses port 8787"):
            WeChatRelayAdapter(cfg)


class TestPayloadDispatch:
    def test_text_payload_normalizes_logical_key_session(self):
        cfg = PlatformConfig(enabled=True, extra={"auto_reply": False})
        adapter = WeChatRelayAdapter(cfg)
        result = asyncio.run(adapter._dispatch_payload({
            "logicalKey": "room-1",
            "title": "测试群",
            "senderId": "u1",
            "senderName": "Alice",
            "text": "hello",
        }))
        assert result == {
            "ok": True,
            "dispatched": False,
            "logicalKey": "room-1",
            "chatId": "wechat:logical:room-1",
        }

    def test_media_placeholder_is_reserved_not_dispatched(self):
        cfg = PlatformConfig(enabled=True, extra={"auto_reply": False})
        adapter = WeChatRelayAdapter(cfg)
        result = asyncio.run(adapter._dispatch_payload({
            "logicalKey": "room-1",
            "text": "[图片]",
        }))
        assert result["ok"] is True
        assert result["ignored"] is True
        assert result["event"] == "wechat.media_inbound"

    def test_auto_reply_calls_base_message_handler_with_wechat_relay_source(self):
        cfg = PlatformConfig(enabled=True, extra={"auto_reply": True})
        adapter = WeChatRelayAdapter(cfg)
        seen: list[MessageEvent] = []

        async def handler(event: MessageEvent):
            seen.append(event)
            return None

        adapter.set_message_handler(handler)
        result = asyncio.run(adapter._dispatch_payload({
            "logicalKey": "room-2",
            "title": "群2",
            "senderId": "u2",
            "senderName": "Bob",
            "messageId": "m1",
            "text": "ping",
        }))
        assert result["dispatched"] is True
        assert seen[0].text == "ping"
        assert seen[0].source.platform == Platform("wechat_relay")
        assert seen[0].source.chat_id == "wechat:logical:room-2"
        assert seen[0].source.chat_id_alt == "room-2"

    def test_group_message_without_configured_mention_is_collected_but_not_dispatched(self, tmp_path):
        state_file = tmp_path / "mpm_state.json"
        cfg = PlatformConfig(enabled=True, extra={
            "auto_reply": True,
            "require_mention_in_groups": True,
            "mention_names": "超进化",
            "collect_group_messages": True,
            "collection_state_file": str(state_file),
            "collection_project": "班旗项目",
        })
        adapter = WeChatRelayAdapter(cfg)
        seen: list[MessageEvent] = []

        async def handler(event: MessageEvent):
            seen.append(event)
            return None

        adapter.set_message_handler(handler)
        result = asyncio.run(adapter._dispatch_payload({
            "logicalKey": "room-3",
            "title": "ct5机械手视觉交流",
            "senderName": "寻梦像扑火",
            "messageId": "m-silent-1",
            "text": "机械手到控制器的线 有几根需要接的 有需要焊接的插头吗",
        }))
        assert result["ok"] is True
        assert result["dispatched"] is False
        assert result["ignored"] is True
        assert result["collected"] is True
        assert result["reason"] == "group_mention_required"
        assert seen == []
        state = json.loads(state_file.read_text(encoding="utf-8"))
        item = state["message_inbox"][-1]
        assert item["project"] == "班旗项目"
        assert item["chat_type"] == "group"
        assert item["sender_name"] == "寻梦像扑火"
        assert item["category"] == "询问"
        assert item["device_hint"] == "机械手"
        assert item["requires_reply"] is False

    def test_group_message_with_configured_mention_is_dispatched(self):
        cfg = PlatformConfig(enabled=True, extra={
            "auto_reply": True,
            "require_mention_in_groups": True,
            "mention_names": "超进化",
        })
        adapter = WeChatRelayAdapter(cfg)
        seen: list[MessageEvent] = []

        async def handler(event: MessageEvent):
            seen.append(event)
            return None

        adapter.set_message_handler(handler)
        result = asyncio.run(adapter._dispatch_payload({
            "logicalKey": "room-4",
            "title": "ct5机械手视觉交流",
            "senderName": "程",
            "text": "@超进化 这根线怎么接",
        }))
        assert result["dispatched"] is True
        assert seen[0].text == "@超进化 这根线怎么接"
        assert seen[0].source.chat_type == "group"


@pytest.mark.asyncio
async def test_http_ws_health_inbound_and_outbound_smoke():
    import aiohttp

    port = _free_port()
    cfg = PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1",
        "port": port,
        "auto_reply": True,
        "shared_secret": "smoke-secret",
    })
    adapter = WeChatRelayAdapter(cfg)
    seen: list[str] = []

    async def handler(event: MessageEvent):
        seen.append(event.text)
        return "Hermes reply"

    adapter.set_message_handler(handler)
    assert await adapter.connect()
    try:
        async with aiohttp.ClientSession() as session:
            # healthz is always accessible without auth
            async with session.get(f"http://127.0.0.1:{port}/_openclaw/notify/healthz") as resp:
                health = await resp.json()
            assert health["ok"] is True
            assert health["platform"] == "wechat_relay"

            auth_headers = {"authorization": "Bearer smoke-secret"}
            async with session.ws_connect(
                f"http://127.0.0.1:{port}/_openclaw/notify/ws?clientRole=wechat-relay&logicalKey=room-ws&title=Room",
                headers=auth_headers,
            ) as ws:
                hello = await ws.receive_json()
                assert hello["type"] == "wechat_relay.connected"

                await ws.send_json({"text": "hello over ws", "messageId": "in-1"})
                ack = await ws.receive_json()
                assert ack["ok"] is True
                assert ack["dispatched"] is True

                outbound = await ws.receive_json(timeout=2)
                assert outbound["type"] == "outbound.sendText"
                assert outbound["logicalKey"] == "room-ws"
                assert outbound["text"] == "Hermes reply"

                async with session.post(
                    f"http://127.0.0.1:{port}/_openclaw/wechat/send",
                    json={"logicalKey": "room-ws", "text": "manual outbound"},
                    headers={"x-wechat-relay-secret": "smoke-secret"},
                ) as resp:
                    send_ack = await resp.json()
                assert send_ack["ok"] is True
                assert send_ack["deliveredVia"] == "websocket"

                outbound2 = await ws.receive_json(timeout=2)
                assert outbound2["type"] == "outbound.sendText"
                assert outbound2["text"] == "manual outbound"

        assert seen == ["hello over ws"]
    finally:
        await adapter.disconnect()


class TestAuthEnforcement:
    class _FakeRequest:
        def __init__(self, headers=None):
            self.headers = headers or {}

    def test_no_secret_no_insecure_opt_in_rejected(self):
        cfg = PlatformConfig(enabled=True, extra={"shared_secret": ""})
        adapter = WeChatRelayAdapter(cfg)
        req = self._FakeRequest()
        assert adapter._authorized_request(req) is False

    def test_allow_insecure_local_bypasses_secret_requirement(self):
        cfg = PlatformConfig(enabled=True, extra={"shared_secret": "", "allow_insecure_local": True})
        adapter = WeChatRelayAdapter(cfg)
        req = self._FakeRequest()
        assert adapter._authorized_request(req) is True

    def test_secret_configured_no_header_rejected(self):
        cfg = PlatformConfig(enabled=True, extra={"shared_secret": "s3cr3t"})
        adapter = WeChatRelayAdapter(cfg)
        req = self._FakeRequest()
        assert adapter._authorized_request(req) is False

    def test_secret_wrong_x_header_rejected(self):
        cfg = PlatformConfig(enabled=True, extra={"shared_secret": "s3cr3t"})
        adapter = WeChatRelayAdapter(cfg)
        req = self._FakeRequest({"x-wechat-relay-secret": "wrong"})
        assert adapter._authorized_request(req) is False

    def test_secret_correct_x_header_accepted(self):
        cfg = PlatformConfig(enabled=True, extra={"shared_secret": "s3cr3t"})
        adapter = WeChatRelayAdapter(cfg)
        req = self._FakeRequest({"x-wechat-relay-secret": "s3cr3t"})
        assert adapter._authorized_request(req) is True

    def test_secret_correct_bearer_accepted(self):
        cfg = PlatformConfig(enabled=True, extra={"shared_secret": "s3cr3t"})
        adapter = WeChatRelayAdapter(cfg)
        req = self._FakeRequest({"authorization": "Bearer s3cr3t"})
        assert adapter._authorized_request(req) is True

    def test_secret_wrong_bearer_rejected(self):
        cfg = PlatformConfig(enabled=True, extra={"shared_secret": "s3cr3t"})
        adapter = WeChatRelayAdapter(cfg)
        req = self._FakeRequest({"authorization": "Bearer wrong"})
        assert adapter._authorized_request(req) is False


@pytest.mark.asyncio
async def test_send_endpoint_blocked_without_any_auth_config():
    import aiohttp

    port = _free_port()
    cfg = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": port})
    adapter = WeChatRelayAdapter(cfg)
    assert await adapter.connect()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/_openclaw/wechat/send",
                json={"logicalKey": "test", "text": "should be blocked"},
            ) as resp:
                body = await resp.json()
            assert resp.status == 401
            assert body["error"] == "unauthorized"
    finally:
        await adapter.disconnect()
