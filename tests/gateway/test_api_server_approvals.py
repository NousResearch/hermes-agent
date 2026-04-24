"""Approval-gate coverage for the API-server dashboard chat routes."""

import asyncio
import json
import time
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


class _FakeSessionDb:
    def get_session(self, session_id):
        return {"session_id": session_id, "title": "Approval Test"}

    def ensure_session(self, session_id, source="web"):
        return None

    def get_messages_as_conversation(self, session_id):
        return []


class _FakeAgent:
    def run_conversation(self, user_content, conversation_history=None, persist_user_message=True):
        from tools.approval import _gateway_notify_cbs

        callback = _gateway_notify_cbs.get("sess-approval")
        if callback:
            callback(
                {
                    "approval_id": "approval-1",
                    "command": "python3 <<'EOF'\nprint('hi')\nEOF",
                    "description": "script execution via heredoc",
                    "pattern_key": "script execution via heredoc",
                    "pattern_keys": ["script execution via heredoc"],
                }
            )

        # Give the stream loop a chance to drain the approval frame before the
        # fake run completes, matching a real blocked approval wait.
        time.sleep(0.05)
        return {
            "final_response": "Done",
            "completed": True,
            "messages": [],
            "api_calls": 1,
        }


class _SlowFakeAgent:
    def __init__(self, release_event):
        self.release_event = release_event
        self.interrupted = False

    def interrupt(self, reason):
        self.interrupted = True
        self.release_event.set()

    def run_conversation(self, user_content, conversation_history=None, persist_user_message=True):
        self.release_event.wait(timeout=1)
        return {
            "final_response": "Stream owned by backend",
            "completed": True,
            "messages": [],
            "api_calls": 0,
        }


class _ToolProgressFakeAgent:
    def __init__(self, tool_progress_callback):
        self.tool_progress_callback = tool_progress_callback

    def run_conversation(self, user_content, conversation_history=None, persist_user_message=True):
        self.tool_progress_callback("tool.started", "terminal", "pwd", {"command": "pwd"})
        return {
            "final_response": "Tool finished",
            "completed": True,
            "messages": [],
            "api_calls": 1,
        }


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/api/sessions/{session_id}/chat/start", adapter._handle_session_chat_start)
    app.router.add_get("/api/chat/stream", adapter._handle_chat_stream_attach)
    app.router.add_get("/api/chat/stream/status", adapter._handle_chat_stream_status)
    app.router.add_post("/api/chat/cancel", adapter._handle_chat_stream_cancel)
    app.router.add_post("/api/approval/respond", adapter._handle_approval_respond)
    return app


@pytest.mark.asyncio
async def test_session_chat_stream_emits_approval_event(adapter=None):
    adapter = _make_adapter()
    app = _create_app(adapter)

    with (
        patch.object(adapter, "_get_session_db", return_value=_FakeSessionDb()),
        patch.object(adapter, "_create_agent", return_value=_FakeAgent()),
    ):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/api/sessions/sess-approval/chat/stream",
                json={"message": "write the script"},
            )

            assert response.status == 200
            body = await response.text()

    assert "event: approval" in body
    assert '"approval_id": "approval-1"' in body
    assert "script execution via heredoc" in body


@pytest.mark.asyncio
async def test_session_chat_start_returns_stream_id_that_can_be_attached():
    import threading

    adapter = _make_adapter()
    app = _create_app(adapter)
    release_event = threading.Event()
    fake_agent = _SlowFakeAgent(release_event)

    with (
        patch.object(adapter, "_get_session_db", return_value=_FakeSessionDb()),
        patch.object(adapter, "_create_agent", return_value=fake_agent),
    ):
        async with TestClient(TestServer(app)) as cli:
            started = await cli.post(
                "/api/sessions/sess-webui/chat/start",
                json={"message": "run this like WebUI"},
            )

            assert started.status == 200
            started_payload = await started.json()
            stream_id = started_payload["stream_id"]
            assert stream_id
            assert started_payload["session_id"] == "sess-webui"

            status = await cli.get(f"/api/chat/stream/status?stream_id={stream_id}")
            assert status.status == 200
            status_payload = await status.json()
            assert status_payload["active"] is True
            assert status_payload["session_id"] == "sess-webui"

            release_event.set()
            attached = await cli.get(f"/api/chat/stream?stream_id={stream_id}")
            body = await attached.text()

            assert attached.status == 200
            assert "event: message.started" in body
            assert "event: assistant.completed" in body
            assert "Stream owned by backend" in body

            final_status = await cli.get(f"/api/chat/stream/status?stream_id={stream_id}")
            final_payload = await final_status.json()
            assert final_payload["active"] is False


@pytest.mark.asyncio
async def test_backend_owned_stream_sends_heartbeat_while_agent_is_quiet(monkeypatch):
    import threading
    from gateway.platforms import api_server as api_server_module

    monkeypatch.setattr(api_server_module, "CHAT_STREAM_HEARTBEAT_SECONDS", 0.05, raising=False)
    adapter = _make_adapter()
    app = _create_app(adapter)
    release_event = threading.Event()
    fake_agent = _SlowFakeAgent(release_event)

    with (
        patch.object(adapter, "_get_session_db", return_value=_FakeSessionDb()),
        patch.object(adapter, "_create_agent", return_value=fake_agent),
    ):
        async with TestClient(TestServer(app)) as cli:
            started = await cli.post(
                "/api/sessions/sess-heartbeat/chat/start",
                json={"message": "run quietly for a while"},
            )
            stream_id = (await started.json())["stream_id"]
            attached = await cli.get(f"/api/chat/stream?stream_id={stream_id}")

            blocks = []
            try:
                for _ in range(4):
                    block = await asyncio.wait_for(attached.content.readuntil(b"\n\n"), timeout=1)
                    blocks.append(block.decode())
            finally:
                release_event.set()
                attached.close()

    assert any("event: heartbeat" in block for block in blocks)


@pytest.mark.asyncio
async def test_backend_owned_stream_emits_live_tool_before_final_answer():
    adapter = _make_adapter()
    app = _create_app(adapter)

    def create_agent(**kwargs):
        return _ToolProgressFakeAgent(kwargs["tool_progress_callback"])

    with (
        patch.object(adapter, "_get_session_db", return_value=_FakeSessionDb()),
        patch.object(adapter, "_create_agent", side_effect=create_agent),
    ):
        async with TestClient(TestServer(app)) as cli:
            started = await cli.post(
                "/api/sessions/sess-tools/chat/start",
                json={"message": "check pwd"},
            )
            stream_id = (await started.json())["stream_id"]
            attached = await cli.get(f"/api/chat/stream?stream_id={stream_id}")
            body = await attached.text()

    assert attached.status == 200
    tool_index = body.index("event: tool")
    final_index = body.index("event: assistant.completed")
    assert tool_index < final_index
    assert '"name": "terminal"' in body
    assert '"command": "pwd"' in body


@pytest.mark.asyncio
async def test_approval_respond_validates_and_resolves_one_request():
    adapter = _make_adapter(api_key="sk-secret")
    app = _create_app(adapter)

    with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/api/approval/respond",
                headers={"Authorization": "Bearer sk-secret"},
                json={
                    "session_id": "sess-approval",
                    "choice": "once",
                    "approval_id": "approval-1",
                },
            )

            assert response.status == 200
            payload = await response.json()

    assert payload == {"ok": True, "choice": "once", "resolved": 1}
    resolve.assert_called_once_with("sess-approval", "once", resolve_all=False, approval_id="approval-1")


@pytest.mark.asyncio
async def test_approval_respond_rejects_bad_choice():
    adapter = _make_adapter()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/approval/respond",
            json={"session_id": "sess-approval", "choice": "definitely"},
        )

        assert response.status == 400
        payload = await response.json()

    assert "Invalid choice" in json.dumps(payload)
