"""Tests for the REST/WebSocket API platform adapter."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api import APIPlatformAdapter, check_api_requirements
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource, build_session_key


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_adapter() -> APIPlatformAdapter:
    config = PlatformConfig(enabled=True)
    adapter = APIPlatformAdapter(config)
    return adapter


def _build_api_session_key(chat_id: str) -> str:
    return build_session_key(SessionSource(
        platform=Platform.API,
        chat_id=chat_id,
        user_id=chat_id,
        chat_type="channel",
    ))


# ── Platform enum ────────────────────────────────────────────────────────


class TestPlatformEnum:
    def test_api_in_platform_enum(self):
        assert hasattr(Platform, "API")
        assert Platform.API.value == "api"


# ── check_api_requirements ───────────────────────────────────────────────


class TestCheckRequirements:
    def test_returns_true_when_fastapi_available(self):
        assert check_api_requirements() is True

    def test_returns_false_when_fastapi_missing(self):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "fastapi":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            assert check_api_requirements() is False


# ── Constructor ──────────────────────────────────────────────────────────


class TestAdapterConstructor:
    def test_platform_is_api(self):
        adapter = _make_adapter()
        assert adapter.platform == Platform.API

    def test_default_port(self):
        adapter = _make_adapter()
        assert adapter._port == 8765

    def test_custom_port(self):
        with patch.dict(os.environ, {"API_PORT": "9999"}):
            adapter = _make_adapter()
            assert adapter._port == 9999

    def test_empty_response_queues(self):
        adapter = _make_adapter()
        assert adapter._response_queues == {}


# ── Queue management ────────────────────────────────────────────────────


class TestQueueManagement:
    def test_register_queue_creates_queue(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("session-1")
        assert isinstance(queue, asyncio.Queue)

    def test_register_queue_keyed_by_session_key(self):
        adapter = _make_adapter()
        adapter.register_queue("session-1")
        expected_key = _build_api_session_key("session-1")
        assert expected_key in adapter._response_queues

    def test_unregister_queue_removes(self):
        adapter = _make_adapter()
        adapter.register_queue("session-1")
        adapter.unregister_queue("session-1")
        expected_key = _build_api_session_key("session-1")
        assert expected_key not in adapter._response_queues

    def test_unregister_nonexistent_is_noop(self):
        adapter = _make_adapter()
        adapter.unregister_queue("nonexistent")  # should not raise


# ── Send methods route to queue ──────────────────────────────────────────


class TestSendToQueue:
    @pytest.mark.asyncio
    async def test_send_text(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send("s1", "Hello world")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg == {"type": "message", "content": "Hello world"}

    @pytest.mark.asyncio
    async def test_send_image(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_image("s1", "https://img.png", caption="test")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "image"
        assert msg["url"] == "https://img.png"
        assert msg["caption"] == "test"

    @pytest.mark.asyncio
    async def test_send_voice(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_voice("s1", "/tmp/audio.ogg")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "audio"
        assert msg["path"] == "/tmp/audio.ogg"

    @pytest.mark.asyncio
    async def test_send_video(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_video("s1", "/tmp/video.mp4")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "video"
        assert msg["path"] == "/tmp/video.mp4"

    @pytest.mark.asyncio
    async def test_send_document(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_document("s1", "/tmp/doc.pdf")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "document"
        assert msg["path"] == "/tmp/doc.pdf"

    @pytest.mark.asyncio
    async def test_send_image_file(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_image_file("s1", "/tmp/photo.jpg")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "image"
        assert msg["path"] == "/tmp/photo.jpg"

    @pytest.mark.asyncio
    async def test_send_no_queue_is_noop(self):
        adapter = _make_adapter()
        result = await adapter.send("no-queue", "test")
        assert result.success is True
        # no queue registered, so nothing happens and no error


# ── Session key ──────────────────────────────────────────────────────────


class TestSessionKey:
    def test_different_sessions_get_different_keys(self):
        adapter = _make_adapter()
        key1 = adapter._build_session_key("session-a")
        key2 = adapter._build_session_key("session-b")
        assert key1 != key2

    def test_session_key_includes_chat_id(self):
        adapter = _make_adapter()
        key = adapter._build_session_key("my-session")
        assert "my-session" in key

    def test_session_key_includes_api_platform(self):
        adapter = _make_adapter()
        key = adapter._build_session_key("test")
        assert "api" in key


# ── get_chat_info ────────────────────────────────────────────────────────


class TestGetChatInfo:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        adapter = _make_adapter()
        info = await adapter.get_chat_info("test-123")
        assert info["name"] == "api-test-123"
        assert info["type"] == "channel"


# ── _process_message_background done signal ──────────────────────────────


class TestDoneSignal:
    @pytest.mark.asyncio
    async def test_done_put_on_queue_after_processing(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        session_key = _build_api_session_key("s1")

        # Mock the parent's _process_message_background to just put a text msg
        async def fake_parent(event, sk):
            q = adapter._response_queues.get(sk)
            if q:
                await q.put({"type": "message", "content": "response"})

        event = MessageEvent(
            text="hi",
            source=SessionSource(
                platform=Platform.API, chat_id="s1", user_id="s1", chat_type="channel"
            ),
        )

        with patch(
            "gateway.platforms.base.BasePlatformAdapter._process_message_background",
            side_effect=fake_parent,
        ):
            await adapter._process_message_background(event, session_key)

        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())
        types = [m["type"] for m in messages]
        assert "message" in types
        assert types[-1] == "done"

    @pytest.mark.asyncio
    async def test_done_sent_even_on_error(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        session_key = _build_api_session_key("s1")

        async def failing_parent(event, sk):
            raise RuntimeError("agent crashed")

        event = MessageEvent(
            text="hi",
            source=SessionSource(
                platform=Platform.API, chat_id="s1", user_id="s1", chat_type="channel"
            ),
        )

        with patch(
            "gateway.platforms.base.BasePlatformAdapter._process_message_background",
            side_effect=failing_parent,
        ):
            # Should not raise — error is caught
            await adapter._process_message_background(event, session_key)

        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())
        assert messages[-1]["type"] == "done"


# ── handle_request ───────────────────────────────────────────────────────


class TestHandleRequest:
    @pytest.mark.asyncio
    async def test_calls_handle_message(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        await adapter.handle_request("session-1", "hello")

        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args[0][0]
        assert isinstance(event, MessageEvent)
        assert event.text == "hello"
        assert event.source.platform == Platform.API
        assert event.source.chat_id == "session-1"
        assert event.source.chat_type == "channel"

    @pytest.mark.asyncio
    async def test_custom_user_id(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        await adapter.handle_request("session-1", "hi", user_id="user-42")

        event = adapter.handle_message.call_args[0][0]
        assert event.source.user_id == "user-42"


# ── Config integration ───────────────────────────────────────────────────


class TestConfigIntegration:
    def test_api_in_connected_platforms_when_enabled(self):
        from gateway.config import GatewayConfig
        config = GatewayConfig(
            platforms={Platform.API: PlatformConfig(enabled=True)}
        )
        connected = config.get_connected_platforms()
        assert Platform.API in connected

    def test_api_not_connected_when_disabled(self):
        from gateway.config import GatewayConfig
        config = GatewayConfig(
            platforms={Platform.API: PlatformConfig(enabled=False)}
        )
        connected = config.get_connected_platforms()
        assert Platform.API not in connected

    def test_env_override_enables_api(self):
        from gateway.config import _apply_env_overrides, GatewayConfig
        config = GatewayConfig()
        with patch.dict(os.environ, {"API_ENABLED": "true"}):
            _apply_env_overrides(config)
        assert Platform.API in config.platforms
        assert config.platforms[Platform.API].enabled is True


# ── FastAPI app ──────────────────────────────────────────────────────────


class TestFastAPIApp:
    def test_create_app_returns_fastapi(self):
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)

    def test_health_route_exists(self):
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        routes = [r.path for r in app.routes]
        assert "/v1/health" in routes

    def test_chat_route_exists(self):
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        routes = [r.path for r in app.routes]
        assert "/v1/chat" in routes

    def test_stream_route_exists(self):
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        routes = [r.path for r in app.routes]
        assert "/v1/chat/stream" in routes


# ── HTTP endpoint tests (via TestClient) ─────────────────────────────────


class TestHTTPEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        self.adapter = adapter
        app = create_app(adapter)
        return TestClient(app)

    def test_health_no_auth(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_chat_no_auth_returns_422(self, client):
        resp = client.post("/v1/chat", json={"message": "hi"})
        assert resp.status_code == 422  # missing Authorization header

    def test_chat_wrong_key_returns_401(self, client):
        with patch.dict(os.environ, {"API_KEY": "correct-key"}):
            resp = client.post(
                "/v1/chat",
                json={"message": "hi"},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp.status_code == 401

    def test_interrupt_no_active_session(self, client):
        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/chat/interrupt",
                params={"session_id": "nonexistent"},
                headers={"Authorization": "Bearer test-key"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["interrupted"] is False

    def test_list_sessions_empty(self, client):
        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.get(
                "/v1/sessions",
                headers={"Authorization": "Bearer test-key"},
            )
            assert resp.status_code == 200
            assert resp.json()["sessions"] == []


# ── Toolset wiring ───────────────────────────────────────────────────────


# ── Security hardening tests ─────────────────────────────────────────────


class TestSecurityHardening:
    """Tests for the security fixes: binding, message limits, requirements."""

    def test_default_host_is_localhost(self):
        adapter = _make_adapter()
        assert adapter._host == "127.0.0.1"

    def test_custom_host_from_env(self):
        with patch.dict(os.environ, {"API_HOST": "0.0.0.0"}):
            adapter = _make_adapter()
            assert adapter._host == "0.0.0.0"

    def test_check_requirements_also_checks_uvicorn(self):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            assert check_api_requirements() is False

    def test_message_max_length_enforced(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)
        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/chat",
                json={"message": "x" * 100_001},
                headers={"Authorization": "Bearer test-key"},
            )
            assert resp.status_code == 422  # validation error

    def test_message_within_limit_accepted(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()

        async def fake_handle(session_id, message, user_id=None):
            key = adapter._build_session_key(session_id)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": "ok"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)
        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/chat",
                json={"message": "x" * 1000},
                headers={"Authorization": "Bearer test-key"},
            )
            assert resp.status_code == 200


# ── Successful chat flow ─────────────────────────────────────────────────


class TestChatFlow:
    """Test a complete REST chat request-response cycle."""

    @pytest.mark.asyncio
    async def test_successful_chat_returns_response(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()

        # Mock handle_request to simulate agent putting a response on the queue
        async def fake_handle(session_id, message, user_id=None):
            key = adapter._build_session_key(session_id)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": "Hello back!"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/chat",
                json={"message": "Hi", "session_id": "test-1"},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello back!"
        assert data["session_id"] == "test-1"

    @pytest.mark.asyncio
    async def test_chat_with_media(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()

        async def fake_handle(session_id, message, user_id=None):
            key = adapter._build_session_key(session_id)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": "Here's an image"})
                await q.put({"type": "image", "url": "https://example.com/img.png"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/chat",
                json={"message": "Show me something"},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Here's an image"
        assert len(data["media"]) == 1
        assert data["media"][0]["type"] == "image"


# ── WebSocket tests ──────────────────────────────────────────────────────


class TestWebSocket:
    """Test WebSocket auth, messaging, and edge cases."""

    def test_ws_auth_success(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "ws-key"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "ws-key", "session_id": "ws-1"})
                resp = ws.receive_json()
                assert resp["type"] == "auth_ok"

    def test_ws_auth_wrong_key_closes(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "correct"}):
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/chat/stream") as ws:
                    ws.send_json({"type": "auth", "token": "wrong"})
                    ws.receive_json()  # should close

    def test_ws_chat_flow(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()

        async def fake_handle(session_id, message, user_id=None):
            key = adapter._build_session_key(session_id)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": "WS response"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "ws-key"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "ws-key", "session_id": "ws-1"})
                auth_resp = ws.receive_json()
                assert auth_resp["type"] == "auth_ok"

                ws.send_json({"message": "Hello via WS"})
                msg = ws.receive_json()
                assert msg["type"] == "message"
                assert msg["content"] == "WS response"

                done = ws.receive_json()
                assert done["type"] == "done"

    def test_ws_empty_message_returns_error(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "ws-key"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "ws-key"})
                ws.receive_json()  # auth_ok

                ws.send_json({"message": ""})
                resp = ws.receive_json()
                assert resp["type"] == "error"


# ── Session transcript ───────────────────────────────────────────────────


class TestSessionTranscript:
    def test_get_session_without_store(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.get(
                "/v1/sessions/test-session",
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["messages"] == []
        assert "not available" in data.get("note", "")

    def test_get_session_with_store(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        mock_store = MagicMock()
        mock_store.load_transcript.return_value = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        adapter._session_store = mock_store
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.get(
                "/v1/sessions/test-session",
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 2


class TestToolsetWiring:
    def test_hermes_api_toolset_exists(self):
        from toolsets import TOOLSETS
        assert "hermes-api" in TOOLSETS

    def test_hermes_gateway_includes_api(self):
        from toolsets import TOOLSETS
        includes = TOOLSETS["hermes-gateway"]["includes"]
        assert "hermes-api" in includes
