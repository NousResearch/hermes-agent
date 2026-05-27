"""Tests for conversation-native API input injection."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import MessageType
from gateway.session import SessionEntry, SessionSource


def _make_adapter(api_key: str = "test-key") -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": api_key})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/conversations/input", adapter._handle_conversation_input)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


def _entry(session_key: str = "matrix:room:user") -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=now,
        updated_at=now,
        origin=SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:lumeny.io",
            chat_type="dm",
            user_id="user-1",
            user_name="Agent Peer",
            thread_id="thread-1",
        ),
        platform=Platform.MATRIX,
        chat_type="dm",
    )


class _FakeSessionStore:
    def __init__(self, entry=None):
        self.entry = entry

    def get_entry(self, session_key):
        if self.entry and self.entry.session_key == session_key:
            return self.entry
        return None


class _FakeAdapter:
    def __init__(self):
        self._active_sessions = {}
        self._pending_messages = {}
        self.started = []

    def _heal_stale_session_lock(self, session_key):
        return False

    def _start_session_processing(self, event, session_key):
        self.started.append((session_key, event))
        return True


@pytest.mark.asyncio
async def test_conversation_input_requires_api_key_even_on_local_server():
    adapter = _make_adapter(api_key="")
    adapter.gateway_runner = MagicMock()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/conversations/input",
            json={"session_key": "matrix:room:user", "text": "hello"},
        )
        data = await resp.json()

    assert resp.status == 403
    assert data["error"]["code"] == "api_key_required"


@pytest.mark.asyncio
async def test_conversation_input_steers_running_session():
    adapter = _make_adapter()
    runner = MagicMock()
    runner.inject_session_input = AsyncMock(return_value={
        "ok": True,
        "object": "hermes.conversation.input",
        "session_key": "matrix:room:user",
        "session_id": "sess-1",
        "accepted": True,
        "action": "steered",
        "mode": "auto",
        "fallback": "message",
        "input_visibility": "silent",
        "output_delivery": "conversation",
    })
    adapter.gateway_runner = runner
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/conversations/input",
            headers={"Authorization": "Bearer test-key"},
            json={"session_key": "matrix:room:user", "text": "notice this", "mode": "auto"},
        )
        data = await resp.json()

    assert resp.status == 202
    assert data["action"] == "steered"
    runner.inject_session_input.assert_awaited_once()


@pytest.mark.asyncio
async def test_conversation_input_unknown_session_returns_404():
    adapter = _make_adapter()
    runner = MagicMock()
    runner.inject_session_input = AsyncMock(return_value={
        "ok": False,
        "status": 404,
        "code": "session_not_found",
        "message": "No gateway session found for key: missing",
    })
    adapter.gateway_runner = runner
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/conversations/input",
            headers={"Authorization": "Bearer test-key"},
            json={"session_key": "missing", "text": "hello"},
        )
        data = await resp.json()

    assert resp.status == 404
    assert data["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_gateway_runner_inject_session_input_starts_idle_platform_delivery():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = _FakeSessionStore(_entry())
    platform_adapter = _FakeAdapter()
    runner.adapters = {Platform.MATRIX: platform_adapter}
    runner._running_agents = {}
    runner._queued_events = {}

    result = await GatewayRunner.inject_session_input(
        runner,
        session_key="matrix:room:user",
        text="peer update",
        mode="auto",
        fallback="message",
    )

    assert result["ok"] is True
    assert result["action"] == "started"
    assert result["output_delivery"] == "conversation"
    session_key, event = platform_adapter.started[0]
    assert session_key == "matrix:room:user"
    assert event.text == "peer update"
    assert event.internal is True
    assert event.message_type == MessageType.TEXT
    assert event.source.thread_id == "thread-1"


@pytest.mark.asyncio
async def test_gateway_runner_inject_session_input_steers_without_queueing():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = _FakeSessionStore(_entry())
    platform_adapter = _FakeAdapter()
    runner.adapters = {Platform.MATRIX: platform_adapter}
    agent = MagicMock()
    agent.steer.return_value = True
    runner._running_agents = {"matrix:room:user": agent}
    runner._queued_events = {}

    result = await GatewayRunner.inject_session_input(
        runner,
        session_key="matrix:room:user",
        text="prefer concise",
        mode="steer",
        fallback="message",
    )

    assert result["ok"] is True
    assert result["action"] == "steered"
    agent.steer.assert_called_once_with("prefer concise")
    assert platform_adapter.started == []
    assert platform_adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_gateway_runner_inject_session_input_strict_steer_rejects_when_idle():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = _FakeSessionStore(_entry())
    runner.adapters = {Platform.MATRIX: _FakeAdapter()}
    runner._running_agents = {}
    runner._queued_events = {}

    result = await GatewayRunner.inject_session_input(
        runner,
        session_key="matrix:room:user",
        text="must steer",
        mode="steer",
        fallback="reject",
    )

    assert result["ok"] is False
    assert result["status"] == 409
    assert result["code"] == "not_steerable"
