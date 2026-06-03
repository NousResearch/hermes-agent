import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.linear_aig import LinearAIGAdapter


def _make_adapter(**extra):
    config = PlatformConfig(
        enabled=True,
        token="lin_test_token",
        extra={
            "webhook_secret": "linear-secret",
            **extra,
        },
    )
    return LinearAIGAdapter(config)


def test_authorization_header_distinguishes_api_key_and_oauth_token():
    api_key_adapter = LinearAIGAdapter(
        PlatformConfig(
            enabled=True,
            api_key="lin_api_test_key",
            extra={"webhook_secret": "linear-secret"},
        )
    )
    assert api_key_adapter._authorization_header() == "lin_api_test_key"

    oauth_adapter = LinearAIGAdapter(
        PlatformConfig(
            enabled=True,
            token="lin_oauth_test_token",
            extra={"webhook_secret": "linear-secret"},
        )
    )
    assert oauth_adapter._authorization_header() == "Bearer lin_oauth_test_token"


def test_hermes_env_token_takes_priority_over_user_api_key(monkeypatch):
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_LINEAR_AIG_ACCESS_TOKEN", "lin_oauth_hermes_app")
    monkeypatch.setenv("HERMES_LINEAR_AIG_WEBHOOK_SECRET", "hermes-secret")
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_user_key")

    adapter = LinearAIGAdapter(PlatformConfig(enabled=True))

    assert adapter._resolved_access_token() == "lin_oauth_hermes_app"
    assert adapter._authorization_header() == "Bearer lin_oauth_hermes_app"
    assert adapter._resolved_webhook_secret() == "hermes-secret"


def test_env_token_and_secret_values_are_trimmed(monkeypatch):
    monkeypatch.setenv("LINEAR_ACCESS_TOKEN", "  lin_oauth_trimmed  ")
    monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "\tlinear-secret\n")

    adapter = LinearAIGAdapter(PlatformConfig(enabled=True))

    assert adapter._resolved_access_token() == "lin_oauth_trimmed"
    assert adapter._authorization_header() == "Bearer lin_oauth_trimmed"
    assert adapter._resolved_webhook_secret() == "linear-secret"


def _app(adapter):
    app = web.Application()
    app.router.add_post("/linear/aig", adapter._handle_webhook)
    app.router.add_get("/health", adapter._handle_health)
    return app


def _signature(body: bytes, secret: str = "linear-secret") -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _created_payload():
    return {
        "type": "AgentSessionEvent",
        "action": "created",
        "webhookTimestamp": int(time.time() * 1000),
        "createdAt": "2026-06-03T00:00:00.000Z",
        "actor": {"id": "user-1", "name": "Edgar"},
        "agentSession": {"id": "session-123"},
        "promptContext": "<issue identifier=\"WHO-192\"><title>Do it</title></issue>",
    }


@pytest.mark.asyncio
async def test_created_webhook_verifies_signature_acks_and_dispatches():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._graphql = AsyncMock(
        return_value={
            "agentActivityCreate": {
                "success": True,
                "agentActivity": {"id": "activity-ack"},
            }
        }
    )
    client = TestClient(TestServer(_app(adapter)))
    await client.start_server()
    try:
        body = json.dumps(_created_payload()).encode()
        response = await client.post(
            "/linear/aig",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Linear-Signature": "sha256=" + _signature(body),
                "Linear-Delivery": "delivery-1",
            },
        )
        assert response.status == 202
        data = await response.json()
        assert data["status"] == "accepted"
        assert data["agent_session_id"] == "session-123"

        await asyncio_sleep()
        adapter._graphql.assert_awaited_once()
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.source.platform == Platform.LINEAR_AIG
        assert event.source.chat_id == "linear_aig:session-123"
        assert "WHO-192" in event.text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_invalid_signature_is_rejected_before_dispatch():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._graphql = AsyncMock()
    client = TestClient(TestServer(_app(adapter)))
    await client.start_server()
    try:
        body = json.dumps(_created_payload()).encode()
        response = await client.post(
            "/linear/aig",
            data=body,
            headers={"Linear-Signature": "bad"},
        )
        assert response.status == 401
        adapter._graphql.assert_not_called()
        adapter.handle_message.assert_not_called()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stale_webhook_timestamp_is_rejected():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._graphql = AsyncMock()
    payload = _created_payload()
    payload["webhookTimestamp"] = int((time.time() - 120) * 1000)
    client = TestClient(TestServer(_app(adapter)))
    await client.start_server()
    try:
        body = json.dumps(payload).encode()
        response = await client.post(
            "/linear/aig",
            data=body,
            headers={"Linear-Signature": _signature(body)},
        )
        assert response.status == 401
        data = await response.json()
        assert data["error"] == "Stale webhook timestamp"
        adapter._graphql.assert_not_called()
        adapter.handle_message.assert_not_called()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_prompted_webhook_uses_agent_activity_body_without_initial_ack():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._graphql = AsyncMock()
    payload = {
        "type": "AgentSessionEvent",
        "action": "prompted",
        "createdAt": "2026-06-03T00:00:01.000Z",
        "agentSession": {"id": "session-123"},
        "agentActivity": {
            "id": "prompt-1",
            "content": {"type": "prompt", "body": "Please continue."},
        },
    }
    client = TestClient(TestServer(_app(adapter)))
    await client.start_server()
    try:
        body = json.dumps(payload).encode()
        response = await client.post(
            "/linear/aig",
            data=body,
            headers={"Linear-Signature": _signature(body)},
        )
        assert response.status == 202
        await asyncio_sleep()
        adapter._graphql.assert_not_called()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "Please continue."
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_send_maps_final_progress_and_error_to_agent_activities():
    adapter = _make_adapter()
    adapter._linear_sessions["linear_aig:session-123"] = "session-123"
    adapter._graphql = AsyncMock(
        return_value={
            "agentActivityCreate": {
                "success": True,
                "agentActivity": {"id": "activity-1"},
            }
        }
    )

    result = await adapter.send("linear_aig:session-123", "Done.")
    assert result == SendResult(success=True, message_id="activity-1")
    first_content = adapter._graphql.await_args_list[-1].args[1]["input"]["content"]
    assert first_content == {"type": "response", "body": "Done."}

    await adapter.send_or_update_status(
        "linear_aig:session-123",
        "tool_progress",
        "Running tests",
    )
    progress_content = adapter._graphql.await_args_list[-1].args[1]["input"]["content"]
    assert progress_content["type"] == "action"
    assert progress_content["parameter"] == "Running tests"

    await adapter.send("linear_aig:session-123", "Sorry, I encountered an error (Boom).")
    error_content = adapter._graphql.await_args_list[-1].args[1]["input"]["content"]
    assert error_content["type"] == "error"


async def asyncio_sleep():
    import asyncio

    await asyncio.sleep(0)
