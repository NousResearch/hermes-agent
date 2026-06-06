"""Linear-specific coverage for the generic webhook adapter."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter


def _make_adapter(routes, **extra):
    config = PlatformConfig(enabled=True, extra={"routes": routes, "host": "0.0.0.0", "port": 0, **extra})
    return WebhookAdapter(config)


def _app(adapter):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _linear_signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


@pytest.mark.asyncio
async def test_linear_signature_event_delivery_and_200_acceptance():
    secret = "linear-secret"
    body = json.dumps(
        {
            "action": "create",
            "type": "Comment",
            "data": {"id": "comment-1", "body": "please fix", "issue": {"id": "issue-1"}},
        },
        separators=(",", ":"),
    ).encode()
    adapter = _make_adapter(
        {
            "linear": {
                "secret": secret,
                "events": ["Comment"],
                "prompt": "Linear comment: {data.body}",
                "session_scope": "issue",
            }
        }
    )
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/linear",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Linear-Signature": _linear_signature(body, secret),
                "Linear-Event": "Comment",
                "Linear-Delivery": "delivery-1",
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "accepted"
        assert data["delivery_id"] == "delivery-1"

    assert len(captured) == 1
    event = captured[0]
    assert event.message_id == "delivery-1"
    assert event.source.chat_id == "webhook:linear:issue-1"
    assert event.text == "Linear comment: please fix"


@pytest.mark.asyncio
async def test_linear_invalid_signature_rejected_and_does_not_run_agent():
    adapter = _make_adapter({"linear": {"secret": "linear-secret", "prompt": "x"}})
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/linear",
            json={"type": "Comment"},
            headers={"Linear-Signature": "bad", "Linear-Delivery": "delivery-2"},
        )
        assert resp.status == 401

    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_linear_trigger_regex_ignores_before_agent_run():
    secret = "linear-secret"
    body = b'{"type":"Comment","data":{"body":"ordinary comment","issue":{"id":"issue-1"}}}'
    adapter = _make_adapter(
        {
            "linear": {
                "secret": secret,
                "events": ["Comment"],
                "prompt": "{data.body}",
                "trigger_regex": "@hermes",
                "session_scope": "issue",
            }
        }
    )
    adapter.handle_message = AsyncMock()

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/linear",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Linear-Signature": _linear_signature(body, secret),
                "Linear-Event": "Comment",
                "Linear-Delivery": "delivery-3",
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ignored"

    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_linear_comment_delivery_uses_graphql_without_leaking_token(monkeypatch):
    monkeypatch.setenv("LINEAR_ACCESS_TOKEN", "lin_access_token")
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_key")
    adapter = _make_adapter({})
    delivery = {"deliver_extra": {"issue_id": "issue-123", "parent_id": "parent-456"}}

    class _Response:
        status_code = 200
        text = '{"data":{"commentCreate":{"success":true,"comment":{"id":"comment-new"}}}}'

        def json(self):
            return json.loads(self.text)

    post_mock = AsyncMock(return_value=_Response())

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            return await post_mock(*args, **kwargs)

    with patch("gateway.platforms.webhook.httpx.AsyncClient", return_value=_Client()):
        result = await adapter._direct_deliver("hello Linear", {"deliver": "linear_comment", **delivery})

    assert result == SendResult(success=True, message_id="comment-new")
    _, kwargs = post_mock.call_args
    assert kwargs["headers"]["Authorization"] == "lin_access_token"
    variables = kwargs["json"]["variables"]
    assert variables["input"] == {"issueId": "issue-123", "parentId": "parent-456", "body": "hello Linear"}


@pytest.mark.asyncio
async def test_linear_comment_delivery_missing_token_is_safe(monkeypatch):
    monkeypatch.delenv("LINEAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    adapter = _make_adapter({})
    result = await adapter._direct_deliver("hello", {"deliver": "linear_comment", "deliver_extra": {"issue_id": "issue-1"}})
    assert result.success is False
    assert "lin_" not in (result.error or "")


@pytest.mark.asyncio
async def test_linear_issue_event_session_scope_uses_data_id():
    secret = "linear-secret"
    body = json.dumps({"type": "Issue", "action": "update", "data": {"id": "issue-data-1", "title": "Bug"}}).encode()
    adapter = _make_adapter(
        {"linear": {"secret": secret, "events": ["Issue"], "prompt": "Issue: {data.title}", "session_scope": "issue"}}
    )
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post(
            "/webhooks/linear",
            data=body,
            headers={"Linear-Signature": _linear_signature(body, secret), "Linear-Event": "Issue", "Linear-Delivery": "delivery-issue"},
        )
        assert resp.status == 200

    assert captured[0].source.chat_id == "webhook:linear:issue-data-1"


def test_session_key_template_missing_field_falls_back_to_delivery_id():
    adapter = _make_adapter({})
    chat_id = adapter._build_session_chat_id(
        "linear",
        {"session_key_template": "issue-{data.missing}"},
        {"data": {"id": "issue-1"}},
        "delivery-missing",
    )
    assert chat_id == "webhook:linear:delivery-missing"


def test_session_key_template_normalizes_and_hashes_long_values():
    adapter = _make_adapter({})
    payload = {"data": {"id": "issue-1", "title": "Line one\n" + "x" * 300}}
    chat_id = adapter._build_session_chat_id(
        "linear",
        {"session_key_template": "{data.id}   {data.title}"},
        payload,
        "delivery-long",
    )
    assert chat_id.startswith("webhook:linear:issue-1-Line-one-x")
    assert "\n" not in chat_id
    assert "  " not in chat_id
    assert len(chat_id) <= len("webhook:linear:") + 160
    assert "x" * 200 not in chat_id
