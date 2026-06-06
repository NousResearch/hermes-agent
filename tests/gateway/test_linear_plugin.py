"""Tests for the native Linear AgentSession platform plugin."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platform_registry import PlatformEntry, PlatformRegistry
from plugins.platforms.linear import adapter as linear_adapter
from plugins.platforms.linear.adapter import LinearAgentSessionAdapter


def _signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _adapter(secret="linear-secret", token="linear-token"):
    return LinearAgentSessionAdapter(PlatformConfig(enabled=True, extra={"webhook_secret": secret, "token": token}))


def _app(adapter):
    app = web.Application()
    app.router.add_post("/linear/agent-sessions", adapter._handle_agent_session_webhook)
    return app


@pytest.mark.asyncio
async def test_agent_session_signature_session_mapping_and_created_thought():
    payload = {
        "type": "AgentSessionEvent",
        "action": "created",
        "agentSession": {"id": "as_123", "promptContext": "Investigate LIN-1"},
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    adapter = _adapter()
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_thought"})
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post(
            "/linear/agent-sessions",
            data=body,
            headers={"Linear-Signature": _signature(body, "linear-secret"), "Linear-Delivery": "delivery-1"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "accepted"

    await adapter.wait_background_tasks()
    adapter.client.create_agent_activity.assert_awaited_once()
    _, kwargs = adapter.client.create_agent_activity.call_args
    assert kwargs["agent_session_id"] == "as_123"
    assert kwargs["content"]["type"] == "thought"

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "Investigate LIN-1"
    assert event.source.chat_type == "dm"
    assert event.source.chat_id == "agentSession:as_123"
    assert event.message_id == "delivery-1"
    assert event.raw_message == payload


@pytest.mark.asyncio
async def test_agent_session_duplicate_delivery_is_acked_without_agent_run():
    payload = {"action": "prompted", "agentSession": {"id": "as_123"}, "agentActivity": {"id": "act_1", "body": "hello"}}
    body = json.dumps(payload).encode()
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app(adapter))) as cli:
        headers = {"Linear-Signature": _signature(body, "linear-secret"), "Linear-Delivery": "same-delivery"}
        resp1 = await cli.post("/linear/agent-sessions", data=body, headers=headers)
        resp2 = await cli.post("/linear/agent-sessions", data=body, headers=headers)
        assert resp1.status == 200
        assert resp2.status == 200
        data = await resp2.json()
        assert data["status"] == "duplicate"

    await adapter.wait_background_tasks()
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_agent_session_duplicate_fallback_prefers_webhook_id():
    payload = {
        "webhookId": "webhook-delivery-1",
        "id": "payload-id-should-not-be-used",
        "action": "prompted",
        "agentSession": {"id": "as_123"},
        "agentActivity": {"id": "act_1", "body": "hello"},
    }
    body = json.dumps(payload).encode()
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app(adapter))) as cli:
        headers = {"Linear-Signature": _signature(body, "linear-secret")}
        resp1 = await cli.post("/linear/agent-sessions", data=body, headers=headers)
        resp2 = await cli.post("/linear/agent-sessions", data=body, headers=headers)
        assert resp1.status == 200
        data = await resp1.json()
        assert data["delivery_id"] == "webhook-delivery-1"
        assert resp2.status == 200
        data = await resp2.json()
        assert data["status"] == "duplicate"
        assert data["delivery_id"] == "webhook-delivery-1"

    await adapter.wait_background_tasks()
    assert adapter.handle_message.await_count == 1


def test_requirements_only_check_dependencies_not_credentials(monkeypatch):
    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", True)
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    assert linear_adapter.check_requirements() is True

    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", False)
    assert linear_adapter.check_requirements() is False


def test_registry_can_create_adapter_from_config_credentials(monkeypatch):
    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", True)
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    registry = PlatformRegistry()
    registry.register(PlatformEntry(
        name="linear",
        label="Linear",
        adapter_factory=lambda cfg: LinearAgentSessionAdapter(cfg),
        check_fn=linear_adapter.check_requirements,
        validate_config=linear_adapter.validate_config,
    ))

    cfg = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "token": "config-token"})
    assert isinstance(registry.create_adapter("linear", cfg), LinearAgentSessionAdapter)


def test_config_validation_accepts_config_token_as_outbound_token(monkeypatch):
    monkeypatch.delenv("LINEAR_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    missing_token = PlatformConfig(enabled=True, extra={"webhook_secret": "secret"})
    with_token = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "token": "config-token"})
    with_access_token = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "access_token": "config-access-token"})
    with_api_key = PlatformConfig(enabled=True, extra={"webhook_secret": "secret", "api_key": "config-api-key"})
    assert linear_adapter.validate_config(missing_token) is False
    assert linear_adapter.is_connected(missing_token) is False
    assert linear_adapter.validate_config(with_token) is True
    assert linear_adapter.is_connected(with_token) is True
    assert LinearAgentSessionAdapter(with_token)._token == "config-token"
    assert linear_adapter.validate_config(with_access_token) is True
    assert LinearAgentSessionAdapter(with_access_token)._token == "config-access-token"
    assert linear_adapter.validate_config(with_api_key) is True
    assert LinearAgentSessionAdapter(with_api_key)._token == "config-api-key"


def test_env_enablement_requires_outbound_token(monkeypatch):
    monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "secret")
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    assert linear_adapter._env_enablement() is None

    monkeypatch.setenv("LINEAR_API_KEY", "api-key")
    seed = linear_adapter._env_enablement()
    assert seed["webhook_secret"] == "secret"
    assert seed["token"] == "api-key"


def test_register_declares_secret_and_token_requirements():
    class _Ctx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    ctx = _Ctx()
    linear_adapter.register(ctx)
    assert ctx.kwargs["required_env"] == ["LINEAR_WEBHOOK_SECRET"]
    assert "LINEAR_ACCESS_TOKEN" in ctx.kwargs["install_hint"]
    assert "LINEAR_API_KEY" in ctx.kwargs["install_hint"]


@pytest.mark.asyncio
async def test_connect_refuses_missing_outbound_token(monkeypatch):
    monkeypatch.setattr(linear_adapter, "AIOHTTP_AVAILABLE", True)
    adapter = LinearAgentSessionAdapter(PlatformConfig(enabled=True, extra={"webhook_secret": "secret"}))
    assert await adapter.connect() is False


@pytest.mark.asyncio
async def test_agent_session_prompted_and_stop_signal_mapping():
    adapter = _adapter()
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    prompted = {"action": "prompted", "agentSession": {"id": "as_1", "promptContext": "ctx"}, "agentActivity": {"id": "act_2", "body": "do it"}}
    stop = {"action": "prompted", "agentSession": {"id": "as_1"}, "agentActivity": {"id": "act_3", "body": "ignored", "signal": "stop"}}

    await adapter._process_agent_session_event(prompted, "delivery-prompt")
    await adapter._process_agent_session_event(stop, "delivery-stop")

    assert captured[0].text == "do it\n\nContext:\nctx"
    assert captured[0].message_id == "act_2"
    assert captured[1].text == "/stop"
    assert captured[1].source.chat_id == "agentSession:as_1"


@pytest.mark.asyncio
async def test_send_creates_agent_activity_response_payload():
    adapter = _adapter()
    adapter.client.create_agent_activity = AsyncMock(return_value={"id": "act_response"})

    result = await adapter.send("agentSession:as_123", "done", metadata={"content_type": "response"})

    assert result == SendResult(success=True, message_id="act_response")
    adapter.client.create_agent_activity.assert_awaited_once_with(
        agent_session_id="as_123",
        content={"type": "response", "body": "done"},
    )


@pytest.mark.asyncio
async def test_invalid_signature_rejected():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post(
            "/linear/agent-sessions",
            json={"agentSession": {"id": "as_123"}},
            headers={"Linear-Signature": "bad", "Linear-Delivery": "delivery-1"},
        )
        assert resp.status == 401
    adapter.handle_message.assert_not_called()
