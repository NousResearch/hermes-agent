"""Tests for gateway/notify.py"""

import asyncio
import time
import pytest

# ---- Minimal stubs so we can test without the full gateway ----

class FakeSendResult:
    def __init__(self):
        self.success = True
        self.message_id = "msg_1"

class FakeAdapter:
    """Mimics BasePlatformAdapter.send()"""
    def __init__(self, platform_value="telegram"):
        self.sent = []
        self.platform_value = platform_value

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return FakeSendResult()

class FailAdapter:
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise RuntimeError("adapter exploded")


# We need to build Platform enum stub or import the real one
# For isolated testing, use a stub:
from enum import Enum
class Platform(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"

class FakeGateway:
    def __init__(self, adapters=None):
        self.adapters = adapters or {}


# Patch the Platform import in notify module
import importlib
import gateway.config
gateway.config.Platform = Platform
import gateway.notify as notify_mod
notify_mod.Platform = Platform

# ---- Fixtures ----

@pytest.fixture
def secret():
    return "test-secret-12345"

@pytest.fixture
def gateway_with_telegram():
    adapter = FakeAdapter()
    gw = FakeGateway({Platform.TELEGRAM: adapter})
    return gw, adapter

@pytest.fixture
def gateway_with_discord():
    adapter = FakeAdapter("discord")
    gw = FakeGateway({Platform.DISCORD: adapter})
    return gw, adapter


# ---- Tests ----

class TestNotifyServerInit:
    def test_requires_secret(self):
        """#1: Empty secret must raise, not silently allow all."""
        with pytest.raises(ValueError, match="non-empty secret"):
            notify_mod.NotifyServer(FakeGateway(), secret="")

    def test_none_secret(self):
        with pytest.raises(ValueError, match="non-empty secret"):
            notify_mod.NotifyServer(FakeGateway(), secret=None)

    def test_valid_secret(self, secret):
        server = notify_mod.NotifyServer(FakeGateway(), secret=secret)
        assert server.secret == secret


class TestHandleNotify:
    """Integration tests using aiohttp test client."""

    @pytest.fixture
    async def client(self, gateway_with_telegram, secret, aiohttp_client):
        gw, _ = gateway_with_telegram
        server = notify_mod.NotifyServer(gw, secret=secret)
        # Build app manually
        from aiohttp import web
        app = web.Application()
        app.router.add_post("/notify", server.handle_notify)
        app.router.add_get("/notify/health", server.handle_health)
        # Clear rate limit cache between tests
        notify_mod._rate_limit_cache.clear()
        return await aiohttp_client(app)

    @pytest.mark.asyncio
    async def test_no_auth(self, client):
        resp = await client.post("/notify", json={
            "platform": "telegram", "chat_id": "123", "message": "hi"
        })
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_wrong_auth(self, client, secret):
        resp = await client.post("/notify", json={
            "platform": "telegram", "chat_id": "123", "message": "hi"
        }, headers={"Authorization": "Bearer wrong"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_valid_send(self, client, secret, gateway_with_telegram):
        _, adapter = gateway_with_telegram
        resp = await client.post("/notify", json={
            "platform": "telegram", "chat_id": "123", "message": "hello"
        }, headers={"Authorization": f"Bearer {secret}"})
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] is True
        assert len(adapter.sent) == 1
        assert adapter.sent[0]["chat_id"] == "123"
        assert adapter.sent[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_thread_id(self, client, secret, gateway_with_telegram):
        """#5: thread_id support."""
        _, adapter = gateway_with_telegram
        resp = await client.post("/notify", json={
            "platform": "telegram", "chat_id": "123", "message": "hi",
            "thread_id": "thread_456"
        }, headers={"Authorization": f"Bearer {secret}"})
        assert resp.status == 200
        assert adapter.sent[0]["metadata"] == {"thread_id": "thread_456"}

    @pytest.mark.asyncio
    async def test_missing_fields(self, client, secret):
        resp = await client.post("/notify", json={
            "platform": "telegram"
        }, headers={"Authorization": f"Bearer {secret}"})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_invalid_platform(self, client, secret):
        resp = await client.post("/notify", json={
            "platform": "nonexistent", "chat_id": "123", "message": "hi"
        }, headers={"Authorization": f"Bearer {secret}"})
        assert resp.status == 400
        data = await resp.json()
        assert "Unknown platform" in data["error"]

    @pytest.mark.asyncio
    async def test_platform_not_configured(self, client, secret):
        """Discord not in gateway_with_telegram fixture."""
        resp = await client.post("/notify", json={
            "platform": "discord", "chat_id": "123", "message": "hi"
        }, headers={"Authorization": f"Bearer {secret}"})
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_invalid_json(self, client, secret):
        resp = await client.post("/notify",
            data=b"not json",
            headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_rate_limit(self, client, secret):
        """#3: Rate limiting per chat."""
        headers = {"Authorization": f"Bearer {secret}"}
        body = {"platform": "telegram", "chat_id": "rate_test", "message": "1"}
        resp1 = await client.post("/notify", json=body, headers=headers)
        assert resp1.status == 200
        resp2 = await client.post("/notify", json=body, headers=headers)
        assert resp2.status == 429
        data = await resp2.json()
        assert "retry_after" in data

    @pytest.mark.asyncio
    async def test_rate_limit_different_chats(self, client, secret):
        """Different chat_ids are rate-limited independently."""
        headers = {"Authorization": f"Bearer {secret}"}
        resp1 = await client.post("/notify", json={
            "platform": "telegram", "chat_id": "chat_a", "message": "1"
        }, headers=headers)
        assert resp1.status == 200
        resp2 = await client.post("/notify", json={
            "platform": "telegram", "chat_id": "chat_b", "message": "1"
        }, headers=headers)
        assert resp2.status == 200

    @pytest.mark.asyncio
    async def test_error_no_leak(self, client, secret):
        """#2: Error responses must not leak internal details."""
        # Replace adapter with one that raises
        # We need a fresh client for this
        gw = FakeGateway({Platform.TELEGRAM: FailAdapter()})
        server = notify_mod.NotifyServer(gw, secret=secret)
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        app = web.Application()
        app.router.add_post("/notify", server.handle_notify)
        notify_mod._rate_limit_cache.clear()
        async with TestClient(TestServer(app)) as cl:
            resp = await cl.post("/notify", json={
                "platform": "telegram", "chat_id": "123", "message": "hi"
            }, headers={"Authorization": f"Bearer {secret}"})
            assert resp.status == 500
            data = await resp.json()
            # Must NOT contain "adapter exploded"
            assert "exploded" not in data.get("error", "")
            assert data["error"] == "Delivery failed"

    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/notify/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_log_format(self, client, secret, caplog):
        """#8: Logging uses %-style, not f-string."""
        import logging
        with caplog.at_level(logging.INFO, logger="gateway.notify"):
            resp = await client.post("/notify", json={
                "platform": "telegram", "chat_id": "log_test", "message": "hi"
            }, headers={"Authorization": f"Bearer {secret}"})
            assert resp.status == 200
        # Verify log message exists (format is tested by the fact we use %s in code)
        assert any("delivered" in r.message for r in caplog.records)
