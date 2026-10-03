"""HTTP contract tests for the A2A gateway outbound message route."""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(url: str, *, method: str, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read().decode())


@pytest.fixture
def live_a2a(monkeypatch):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))
    assert asyncio.run(adapter.connect()) is True
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        asyncio.run(adapter.disconnect())


def test_get_exists_as_authenticated_method_not_allowed(live_a2a, monkeypatch):
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)

    status, payload = _request(live_a2a + "/api/v1/messages/send", method="GET")

    assert status == 405
    assert payload == {"ok": False, "error": "method not allowed"}


def test_happy_path_calls_send_tool_and_returns_delivery_receipt(live_a2a, monkeypatch):
    calls = []

    def fake_send(args):
        calls.append(args)
        return json.dumps({"success": True, "platform": "telegram", "message_id": "msg-51"})

    monkeypatch.setattr("tools.send_message_tool.send_message_tool", fake_send)
    status, payload = _request(
        live_a2a + "/api/v1/messages/send",
        method="POST",
        body={"target": "telegram:#alerts", "message": "health degraded"},
    )

    assert calls == [{"action": "send", "target": "telegram:#alerts", "message": "health degraded"}]
    assert status == 200
    assert payload == {"ok": True, "platform": "telegram", "message_id": "msg-51"}
    assert "jsonrpc" not in payload


def test_delivery_failure_is_never_http_200(live_a2a, monkeypatch):
    monkeypatch.setattr(
        "tools.send_message_tool.send_message_tool",
        lambda args: json.dumps({"error": "Unknown platform: nowhere"}),
    )

    status, payload = _request(
        live_a2a + "/api/v1/messages/send",
        method="POST",
        body={"target": "nowhere", "message": "hello"},
    )

    assert status != 200
    assert status == 502
    assert payload == {"ok": False, "error": "Unknown platform: nowhere"}


@pytest.mark.parametrize(
    "body,error",
    [
        ({"message": "hello"}, "target must be a non-empty string"),
        ({"target": 3, "message": "hello"}, "target must be a non-empty string"),
        ({"target": "telegram"}, "message must be a non-empty string"),
        ({"target": "telegram", "message": ""}, "message must be a non-empty string"),
    ],
)
def test_invalid_body_is_400(live_a2a, body, error):
    status, payload = _request(live_a2a + "/api/v1/messages/send", method="POST", body=body)
    assert status == 400
    assert payload == {"ok": False, "error": error}


def test_send_exception_is_non_200_with_summary(live_a2a, monkeypatch):
    def explode(args):
        raise RuntimeError("transport exploded")

    monkeypatch.setattr("tools.send_message_tool.send_message_tool", explode)
    status, payload = _request(
        live_a2a + "/api/v1/messages/send",
        method="POST",
        body={"target": "telegram", "message": "hello"},
    )
    assert status != 200
    assert status == 500
    assert payload["ok"] is False
    assert "transport exploded" in payload["error"]


def test_get_and_post_require_a2a_credentials(monkeypatch):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "route-secret")
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    port = _free_port()
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))
    assert asyncio.run(adapter.connect()) is True
    base = f"http://127.0.0.1:{port}/api/v1/messages/send"
    try:
        get_status, get_payload = _request(base, method="GET")
        post_status, post_payload = _request(
            base, method="POST", body={"target": "telegram", "message": "hello"}
        )
    finally:
        asyncio.run(adapter.disconnect())

    assert (get_status, get_payload) == (401, {"ok": False, "error": "unauthorized"})
    assert (post_status, post_payload) == (401, {"ok": False, "error": "unauthorized"})


def test_unknown_jsonrpc_method_remains_jsonrpc_method_not_found(live_a2a):
    status, payload = _request(
        live_a2a + "/",
        method="POST",
        body={"jsonrpc": "2.0", "id": "unknown", "method": "not/a/method", "params": {}},
    )
    assert status == 200
    assert payload["error"]["code"] == -32601
