"""Tests for the API server WebSocket bridge."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"ws_enabled": True}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_ws_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/ws", adapter._handle_ws)
    return app


async def _read_until_type(ws, frame_type: str, *, limit: int = 10):
    for _ in range(limit):
        frame = await ws.receive_json(timeout=3)
        if frame.get("type") == frame_type:
            return frame
    raise AssertionError(f"Did not receive frame type {frame_type!r}")


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


class TestWebSocketAuth:
    @pytest.mark.asyncio
    async def test_websocket_requires_bearer_auth_when_key_configured(self, auth_adapter):
        app = _create_ws_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with pytest.raises(WSServerHandshakeError) as exc:
                await cli.ws_connect("/v1/ws")
            assert exc.value.status == 401

            ws = await cli.ws_connect(
                "/v1/ws",
                headers={"Authorization": "Bearer sk-secret"},
            )
            await ws.close()


class TestWebSocketStatus:
    @pytest.mark.asyncio
    async def test_status_get_returns_bridge_status(self, adapter):
        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            ws = await cli.ws_connect("/v1/ws")
            await ws.send_json({"id": "status-1", "type": "status.get"})

            frame = await ws.receive_json(timeout=3)
            assert frame["id"] == "status-1"
            assert frame["type"] == "status.result"
            assert frame["status"] == "ok"
            assert frame["model"] == "hermes-agent"
            assert frame["auth_required"] is False
            await ws.close()


class TestWebSocketAgentMessage:
    @pytest.mark.asyncio
    async def test_message_send_streams_deltas_and_done(self, adapter):
        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _run_conversation(user_message=None, conversation_history=None, task_id=None):
                    callback = mock_create.call_args.kwargs["stream_delta_callback"]
                    callback("Hel")
                    callback("lo")
                    return {"final_response": "Hello"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_agent.session_prompt_tokens = 3
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 5
                mock_create.return_value = mock_agent

                ws = await cli.ws_connect("/v1/ws")
                await ws.send_json({
                    "id": "msg-1",
                    "type": "agent.message.send",
                    "text": "hello",
                    "session_id": "mobile-session",
                })

                first_delta = await _read_until_type(ws, "agent.message.delta")
                second_delta = await _read_until_type(ws, "agent.message.delta")
                done = await _read_until_type(ws, "agent.message.done")

                assert first_delta["id"] == "msg-1"
                assert first_delta["delta"] == "Hel"
                assert second_delta["delta"] == "lo"
                assert done["id"] == "msg-1"
                assert done["output"] == "Hello"
                assert done["usage"] == {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                }
                mock_agent.run_conversation.assert_called_once_with(
                    user_message="hello",
                    conversation_history=[],
                    task_id="mobile-session",
                )
                await ws.close()

    @pytest.mark.asyncio
    async def test_disconnect_interrupts_running_agent(self, adapter):
        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                ready = threading.Event()
                interrupted = threading.Event()
                mock_agent = MagicMock()

                def _interrupt(message=None):
                    interrupted.set()

                def _run_conversation(user_message=None, conversation_history=None, task_id=None):
                    ready.set()
                    interrupted.wait(timeout=5)
                    return {"final_response": "stopped"}

                mock_agent.interrupt.side_effect = _interrupt
                mock_agent.run_conversation.side_effect = _run_conversation
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                ws = await cli.ws_connect("/v1/ws")
                await ws.send_json({"id": "msg-1", "type": "agent.message.send", "text": "hello"})
                for _ in range(60):
                    if ready.is_set():
                        break
                    await asyncio.sleep(0.05)
                assert ready.is_set()

                await ws.close()
                for _ in range(20):
                    if interrupted.is_set():
                        break
                    await asyncio.sleep(0.05)

                assert interrupted.is_set()
                mock_agent.interrupt.assert_called_once_with("WebSocket client disconnected")

    @pytest.mark.asyncio
    async def test_message_send_rejects_unknown_history_role(self, adapter):
        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                ws = await cli.ws_connect("/v1/ws")
                await ws.send_json({
                    "id": "msg-2",
                    "type": "agent.message.send",
                    "text": "hello",
                    "conversation_history": [{"role": "tool", "content": "not allowed"}],
                })

                frame = await ws.receive_json(timeout=3)
                assert frame["id"] == "msg-2"
                assert frame["type"] == "agent.message.error"
                assert frame["code"] == "invalid_request"
                assert "conversation_history[0] role" in frame["message"]
                mock_create.assert_not_called()
                await ws.close()


class TestWebSocketSessions:
    @pytest.mark.asyncio
    async def test_session_list_and_get_use_session_db(self, adapter):
        fake_db = MagicMock()
        fake_db.list_sessions_rich.return_value = [
            {"id": "session-1", "title": "Mobile", "message_count": 2}
        ]
        fake_db.get_session.return_value = {"id": "session-1", "source": "api_server"}
        fake_db.get_messages.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        adapter._session_db = fake_db

        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            ws = await cli.ws_connect("/v1/ws")

            await ws.send_json({"id": "list-1", "type": "session.list", "limit": 5})
            listed = await ws.receive_json(timeout=3)
            assert listed == {
                "id": "list-1",
                "type": "session.list.result",
                "sessions": [{"id": "session-1", "title": "Mobile", "message_count": 2}],
            }
            fake_db.list_sessions_rich.assert_called_once_with(limit=5, offset=0)

            await ws.send_json({"id": "get-1", "type": "session.get", "session_id": "session-1"})
            got = await ws.receive_json(timeout=3)
            assert got["id"] == "get-1"
            assert got["type"] == "session.get.result"
            assert got["session"]["id"] == "session-1"
            assert got["messages"] == [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            await ws.close()

    @pytest.mark.asyncio
    async def test_session_list_rejects_non_integer_limit(self, adapter):
        fake_db = MagicMock()
        adapter._session_db = fake_db

        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            ws = await cli.ws_connect("/v1/ws")

            await ws.send_json({"id": "list-2", "type": "session.list", "limit": "5"})
            frame = await ws.receive_json(timeout=3)
            assert frame == {
                "id": "list-2",
                "type": "agent.message.error",
                "code": "invalid_request",
                "message": "limit must be an integer",
            }
            fake_db.list_sessions_rich.assert_not_called()
            await ws.close()

    @pytest.mark.asyncio
    async def test_session_reset_clears_messages(self, adapter):
        fake_db = MagicMock()
        fake_db.get_session.return_value = {"id": "session-1", "source": "api_server"}
        adapter._session_db = fake_db

        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            ws = await cli.ws_connect("/v1/ws")
            await ws.send_json({"id": "reset-1", "type": "session.reset", "session_id": "session-1"})

            frame = await ws.receive_json(timeout=3)
            assert frame == {
                "id": "reset-1",
                "type": "session.reset.result",
                "session_id": "session-1",
                "ok": True,
            }
            fake_db.clear_messages.assert_called_once_with("session-1")
            await ws.close()


class TestWebSocketErrors:
    @pytest.mark.asyncio
    async def test_malformed_json_returns_error_frame(self, adapter):
        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            ws = await cli.ws_connect("/v1/ws")
            await ws.send_str("{not-json")

            frame = await ws.receive_json(timeout=3)
            assert frame["type"] == "agent.message.error"
            assert frame["code"] == "invalid_json"
            await ws.close()

    @pytest.mark.asyncio
    async def test_unknown_type_returns_error_frame(self, adapter):
        app = _create_ws_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            ws = await cli.ws_connect("/v1/ws")
            await ws.send_json({"id": "bad-1", "type": "unknown.type"})

            frame = await ws.receive_json(timeout=3)
            assert frame["id"] == "bad-1"
            assert frame["type"] == "agent.message.error"
            assert frame["code"] == "unknown_type"
            await ws.close()
