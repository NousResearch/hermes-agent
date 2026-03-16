"""Tests for the control API."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.control_api import ControlAPI, REQUIRED_HEADER


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_control_api():
    """Create a ControlAPI with a mock runner."""
    runner = MagicMock()
    runner._running_agents = {"sess1": MagicMock(model="test-model", provider="test")}
    runner.inject_message = MagicMock(return_value=None)
    api = ControlAPI(runner)
    return api, runner


def _make_request(key, body=None):
    request = MagicMock()
    request.match_info = {"key": key}
    if body is not None:
        request.json = AsyncMock(return_value=body)
    return request


# ── send_message tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_message_calls_inject_message():
    """POST /message delegates to runner.inject_message."""
    api, runner = _make_control_api()
    request = _make_request("sess1", {"text": "hello", "mode": "interrupt"})

    resp = await api.send_message(request)

    assert resp.status == 200
    runner.inject_message.assert_called_once_with("sess1", "hello", interrupt=True)


@pytest.mark.asyncio
async def test_send_message_queue_mode():
    """Queue mode passes interrupt=False."""
    api, runner = _make_control_api()
    request = _make_request("sess1", {"text": "/compact", "mode": "queue"})

    resp = await api.send_message(request)

    assert resp.status == 200
    runner.inject_message.assert_called_once_with("sess1", "/compact", interrupt=False)


@pytest.mark.asyncio
async def test_send_message_defaults_to_interrupt():
    """Mode defaults to interrupt when not specified."""
    api, runner = _make_control_api()
    request = _make_request("_any", {"text": "stop"})

    resp = await api.send_message(request)

    assert resp.status == 200
    runner.inject_message.assert_called_once_with("_any", "stop", interrupt=True)


@pytest.mark.asyncio
async def test_send_message_empty_text_returns_400():
    """Empty text should return 400."""
    api, runner = _make_control_api()
    request = _make_request("sess1", {"text": "  "})

    resp = await api.send_message(request)

    assert resp.status == 400
    runner.inject_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_invalid_mode_returns_400():
    """Invalid mode should return 400."""
    api, runner = _make_control_api()
    request = _make_request("sess1", {"text": "hello", "mode": "bogus"})

    resp = await api.send_message(request)

    assert resp.status == 400
    runner.inject_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_lookup_error_returns_404():
    """LookupError from inject_message should return 404."""
    api, runner = _make_control_api()
    runner.inject_message = MagicMock(side_effect=LookupError("No session found"))
    request = _make_request("bad_key", {"text": "hello"})

    resp = await api.send_message(request)

    assert resp.status == 404
    body = json.loads(resp.text)
    assert "No session found" in body["error"]


@pytest.mark.asyncio
async def test_send_message_awaits_coroutine():
    """If inject_message returns a coroutine, it should be awaited."""
    api, runner = _make_control_api()
    runner.inject_message = MagicMock(return_value=AsyncMock()())
    request = _make_request("sess1", {"text": "hello"})

    resp = await api.send_message(request)

    assert resp.status == 200


# ── list_commands tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_commands_returns_commands():
    """GET /commands should return the available commands list."""
    api, _ = _make_control_api()
    request = MagicMock()

    resp = await api.list_commands(request)

    assert resp.status == 200
    body = json.loads(resp.text)
    assert "commands" in body
    commands = [c["command"] for c in body["commands"]]
    assert "/reset" in commands
    assert "/model" in commands


# ── CSRF protection tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_header_returns_403():
    """Requests without X-Hermes-Control header should be rejected."""
    api, _ = _make_control_api()

    from aiohttp.test_utils import TestClient, TestServer
    async with TestClient(TestServer(api.app)) as client:
        resp = await client.get("/health")
        assert resp.status == 403
        body = await resp.json()
        assert "Missing required header" in body["error"]


@pytest.mark.asyncio
async def test_valid_header_passes():
    """Requests with the correct header should succeed."""
    api, _ = _make_control_api()

    from aiohttp.test_utils import TestClient, TestServer
    async with TestClient(TestServer(api.app)) as client:
        resp = await client.get("/health",
                                headers={REQUIRED_HEADER: "1"})
        assert resp.status == 200
        body = await resp.json()
        assert body["status"] == "ok"
