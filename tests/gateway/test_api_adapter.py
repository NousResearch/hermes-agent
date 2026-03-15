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
        assert "/v1/media/" in msg["url"]
        assert "audio.ogg" in msg["url"]

    @pytest.mark.asyncio
    async def test_send_video(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_video("s1", "/tmp/video.mp4")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "video"
        assert "/v1/media/" in msg["url"]
        assert "video.mp4" in msg["url"]

    @pytest.mark.asyncio
    async def test_send_document(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_document("s1", "/tmp/doc.pdf")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "document"
        assert "/v1/media/" in msg["url"]
        assert "doc.pdf" in msg["url"]

    @pytest.mark.asyncio
    async def test_send_image_file(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        result = await adapter.send_image_file("s1", "/tmp/photo.jpg")
        assert result.success is True
        msg = queue.get_nowait()
        assert msg["type"] == "image"
        assert "/v1/media/" in msg["url"]
        assert "photo.jpg" in msg["url"]

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


# ── Media download tests ─────────────────────────────────────────────────


class TestMediaDownload:
    def test_media_registered_and_downloadable(self, tmp_path):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)

        # Create a test audio file
        audio_file = tmp_path / "test_audio.ogg"
        audio_file.write_bytes(b"\x00" * 100)

        # Register the media file
        url = adapter._register_media(str(audio_file))
        assert "/v1/media/" in url
        assert "test_audio.ogg" in url

        client = TestClient(app)
        resp = client.get(url)
        assert resp.status_code == 200
        assert len(resp.content) == 100

    def test_media_invalid_token_rejected(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.get("/v1/media/badtoken/fake.ogg")
        assert resp.status_code == 404

    def test_media_survives_original_deletion(self, tmp_path):
        """Media copy persists even after the original file is deleted."""
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)

        audio_file = tmp_path / "ephemeral.ogg"
        audio_file.write_bytes(b"\x00" * 10)
        url = adapter._register_media(str(audio_file))

        # Delete the ORIGINAL (simulates auto-TTS cleanup)
        audio_file.unlink()

        # Download still works from the api_media copy
        client = TestClient(app)
        resp = client.get(url)
        assert resp.status_code == 200
        assert len(resp.content) == 10

    @pytest.mark.asyncio
    async def test_voice_response_has_downloadable_url(self, tmp_path):
        """End-to-end: send_voice produces a URL that can be downloaded."""
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)

        audio_file = tmp_path / "tts_output.ogg"
        audio_file.write_bytes(b"fake-audio-data")

        queue = adapter.register_queue("s1")
        await adapter.send_voice("s1", str(audio_file))

        msg = queue.get_nowait()
        assert msg["type"] == "audio"

        client = TestClient(app)
        resp = client.get(msg["url"])
        assert resp.status_code == 200
        assert resp.content == b"fake-audio-data"


# ── Upload endpoint tests ────────────────────────────────────────────────


class TestUploadEndpoint:
    def test_upload_returns_url(self, tmp_path):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.txt", b"hello world", "text/plain")},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "url" in data
        assert data["filename"] == "test.txt"
        assert data["size"] == 11

    def test_upload_file_downloadable(self, tmp_path):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            up_resp = client.post(
                "/v1/upload",
                files={"file": ("doc.pdf", b"pdf-content", "application/pdf")},
                headers={"Authorization": "Bearer test-key"},
            )
            url = up_resp.json()["url"]
            dl_resp = client.get(url)

        assert dl_resp.status_code == 200
        assert dl_resp.content == b"pdf-content"

    def test_upload_no_auth_rejected(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.post(
            "/v1/upload",
            files={"file": ("test.txt", b"data", "text/plain")},
        )
        assert resp.status_code == 422  # missing Authorization header

    def test_upload_too_large_rejected(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        # 26 MB > 25 MB limit
        large_data = b"x" * (26 * 1024 * 1024)
        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/upload",
                files={"file": ("big.bin", large_data, "application/octet-stream")},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 413


# ── Voice chat endpoint tests ────────────────────────────────────────────


class TestVoiceChatEndpoint:
    def test_voice_chat_transcribes_and_responds(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()

        async def fake_handle(session_id, message, user_id=None):
            key = adapter._build_session_key(session_id)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": f"You said: {message}"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)

        fake_result = {
            "success": True,
            "transcript": "hello from voice",
            "language": "en",
            "language_probability": 0.95,
        }

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "test-key"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            resp = client.post(
                "/v1/chat/voice",
                files={"file": ("voice.webm", b"fake-audio", "audio/webm")},
                data={"session_id": "voice-1"},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "hello from voice" in data["response"]

    def test_voice_chat_transcription_failure(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        fake_result = {
            "success": False,
            "transcript": "",
            "error": "No speech detected",
        }

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "test-key"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            resp = client.post(
                "/v1/chat/voice",
                files={"file": ("voice.webm", b"silence", "audio/webm")},
                data={"session_id": "voice-2"},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 422
        assert "Transcription failed" in resp.json()["detail"]

    def test_voice_chat_no_auth_rejected(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/voice",
            files={"file": ("voice.webm", b"data", "audio/webm")},
        )
        assert resp.status_code == 422  # missing auth header


# ── Web UI tests ─────────────────────────────────────────────────────────


class TestWebUI:
    def test_root_serves_html(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Hermes Agent" in resp.text
        assert "text/html" in resp.headers["content-type"]

    def test_root_has_voice_button(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.get("/")
        assert "voice-btn" in resp.text
        assert "toggleVoice" in resp.text

    def test_root_has_file_upload(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.get("/")
        assert "file-input" in resp.text
        assert "/v1/upload" in resp.text

    def test_root_has_vad_and_voice_mode(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.get("/")
        assert "startRecording" in resp.text
        assert "SILENCE_THRESHOLD" in resp.text
        assert "voiceMode" in resp.text
        assert "enterVoiceMode" in resp.text
        assert "exitVoiceMode" in resp.text


# ── Media registration and HMAC tests ────────────────────────────────────


class TestMediaRegistration:
    def test_register_media_returns_url_with_token(self, tmp_path):
        adapter = _make_adapter()
        f = tmp_path / "test.ogg"
        f.write_bytes(b"data")
        url = adapter._register_media(str(f))
        assert "/v1/media/" in url
        assert "test.ogg" in url
        # URL has token between /media/ and /filename
        parts = url.split("/")
        token = parts[-2]
        assert len(token) == 64  # Full SHA256 HMAC hex

    def test_register_media_copies_to_api_media_dir(self, tmp_path):
        adapter = _make_adapter()
        f = tmp_path / "ephemeral.mp3"
        f.write_bytes(b"audio-content")
        adapter._register_media(str(f))
        # File should exist in api_media dir
        import glob
        copies = glob.glob(os.path.join(adapter._MEDIA_DIR, "ephemeral.mp3"))
        assert len(copies) == 1

    def test_register_media_different_files_get_different_tokens(self, tmp_path):
        adapter = _make_adapter()
        f1 = tmp_path / "a.ogg"
        f2 = tmp_path / "b.ogg"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        url1 = adapter._register_media(str(f1))
        url2 = adapter._register_media(str(f2))
        assert url1 != url2

    def test_media_files_dict_tracks_registrations(self, tmp_path):
        adapter = _make_adapter()
        f = tmp_path / "track.wav"
        f.write_bytes(b"wav")
        adapter._register_media(str(f))
        assert len(adapter._media_files) == 1
        # Value is the api_media copy path
        path = list(adapter._media_files.values())[0]
        assert os.path.isfile(path)


class TestMediaTTLCleanup:
    def test_old_files_cleaned_on_register(self, tmp_path):
        adapter = _make_adapter()
        adapter._MEDIA_DIR = str(tmp_path / "media")
        os.makedirs(adapter._MEDIA_DIR)

        # Create an "old" file (fake mtime)
        old_file = os.path.join(adapter._MEDIA_DIR, "old.ogg")
        with open(old_file, "wb") as f:
            f.write(b"old")
        os.utime(old_file, (0, 0))  # Set mtime to epoch (very old)

        # Create a "new" file
        new_file = tmp_path / "new.ogg"
        new_file.write_bytes(b"new")

        # Register new file triggers cleanup
        adapter._register_media(str(new_file))

        assert not os.path.exists(old_file), "Old file should be cleaned up"
        # New file copy should exist
        assert os.path.exists(os.path.join(adapter._MEDIA_DIR, "new.ogg"))

    def test_recent_files_kept(self, tmp_path):
        adapter = _make_adapter()
        adapter._MEDIA_DIR = str(tmp_path / "media")
        os.makedirs(adapter._MEDIA_DIR)

        # Create a recent file
        recent = os.path.join(adapter._MEDIA_DIR, "recent.ogg")
        with open(recent, "wb") as f:
            f.write(b"recent")

        # Create another file to trigger cleanup
        new_file = tmp_path / "trigger.ogg"
        new_file.write_bytes(b"trigger")
        adapter._register_media(str(new_file))

        assert os.path.exists(recent), "Recent file should NOT be cleaned up"

    def test_stale_registry_entries_evicted(self, tmp_path):
        adapter = _make_adapter()
        adapter._MEDIA_DIR = str(tmp_path / "media")
        os.makedirs(adapter._MEDIA_DIR)

        # Manually add a stale entry (file doesn't exist)
        adapter._media_files["fake_token/gone.ogg"] = "/nonexistent/gone.ogg"

        new_file = tmp_path / "real.ogg"
        new_file.write_bytes(b"real")
        adapter._register_media(str(new_file))

        assert "fake_token/gone.ogg" not in adapter._media_files


class TestGuessMediaType:
    def test_common_types(self):
        from gateway.api_server import _guess_media_type
        assert _guess_media_type("test.ogg") == "audio/ogg"
        assert _guess_media_type("test.mp3") == "audio/mpeg"
        assert _guess_media_type("test.mp4") == "video/mp4"
        assert _guess_media_type("test.jpg") == "image/jpeg"
        assert _guess_media_type("test.png") == "image/png"
        assert _guess_media_type("test.pdf") == "application/pdf"
        assert _guess_media_type("test.gif") == "image/gif"

    def test_unknown_extension(self):
        from gateway.api_server import _guess_media_type
        assert _guess_media_type("test.xyz") == "application/octet-stream"

    def test_case_insensitive(self):
        from gateway.api_server import _guess_media_type
        assert _guess_media_type("TEST.MP3") == "audio/mpeg"
        assert _guess_media_type("photo.JPG") == "image/jpeg"


class TestMediaSignature:
    def test_sign_produces_consistent_result(self):
        from gateway.api_server import _sign_media_path
        sig1 = _sign_media_path("/tmp/test.ogg")
        sig2 = _sign_media_path("/tmp/test.ogg")
        assert sig1 == sig2

    def test_different_paths_different_signatures(self):
        from gateway.api_server import _sign_media_path
        sig1 = _sign_media_path("/tmp/a.ogg")
        sig2 = _sign_media_path("/tmp/b.ogg")
        assert sig1 != sig2

    def test_make_media_url_format(self):
        from gateway.api_server import _make_media_url, _sign_media_path
        url = _make_media_url("/tmp/hello.mp3")
        token = _sign_media_path("/tmp/hello.mp3")
        assert url == f"/v1/media/{token}/hello.mp3"


# ── Route existence tests ────────────────────────────────────────────────


class TestRouteExistence:
    def _get_routes(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        return [r.path for r in app.routes]

    def test_upload_route_exists(self):
        assert "/v1/upload" in self._get_routes()

    def test_voice_chat_route_exists(self):
        assert "/v1/chat/voice" in self._get_routes()

    def test_media_route_exists(self):
        routes = self._get_routes()
        assert any("/v1/media" in r for r in routes)

    def test_root_route_exists(self):
        assert "/" in self._get_routes()


# ── Web UI content tests ─────────────────────────────────────────────────


class TestWebUIContent:
    def _get_html(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)
        return client.get("/").text

    def test_glassmorphism_theme(self):
        html = self._get_html()
        assert "--glass" in html
        assert "--accent" in html
        assert "backdrop-filter" in html
        assert "#6c5ce7" in html  # purple accent

    def test_echo_prevention(self):
        html = self._get_html()
        assert "currentTtsAudio" in html
        assert "echoCancellation" in html
        assert "noiseSuppression" in html

    def test_waveform_player(self):
        html = self._get_html()
        assert "voice-bubble" in html
        assert "voice-waveform" in html
        assert "voice-play-btn" in html

    def test_file_type_detection(self):
        html = self._get_html()
        assert "image/" in html
        assert "video/" in html
        assert "audio/" in html
        assert "heic" in html
        assert "docx" in html

    def test_markdown_support(self):
        html = self._get_html()
        assert "marked.parse" in html
        assert "highlight.js" in html

    def test_history_persistence(self):
        html = self._get_html()
        assert "saveHistory" in html
        assert "loadHistory" in html
        assert "localStorage" in html

    def test_no_api_key_in_localstorage(self):
        """API key should NOT be saved to localStorage."""
        html = self._get_html()
        assert "hermes_api_key" not in html

    def test_clean_response_filters(self):
        html = self._get_html()
        assert "cleanResponse" in html
        assert "No home channel" in html
        assert "Rate limited" in html


# ── Adapter host config tests ────────────────────────────────────────────


class TestAdapterHostConfig:
    def test_default_host_localhost(self):
        adapter = _make_adapter()
        assert adapter._host == "127.0.0.1"

    def test_custom_host(self):
        with patch.dict(os.environ, {"API_HOST": "0.0.0.0"}):
            adapter = _make_adapter()
            assert adapter._host == "0.0.0.0"

    def test_has_media_files_dict(self):
        adapter = _make_adapter()
        assert hasattr(adapter, "_media_files")
        assert isinstance(adapter._media_files, dict)

    def test_ssl_disabled_by_default(self):
        adapter = _make_adapter()
        assert adapter._ssl_cert is None
        assert adapter._ssl_key is None

    def test_ssl_enabled_via_env(self):
        with patch.dict(os.environ, {"API_SSL_CERT": "/path/cert.pem", "API_SSL_KEY": "/path/key.pem"}):
            adapter = _make_adapter()
            assert adapter._ssl_cert == "/path/cert.pem"
            assert adapter._ssl_key == "/path/key.pem"


# ── Security fix tests ───────────────────────────────────────────────────


class TestSecurityFixes:
    def test_xss_sanitization_in_html(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)
        html = client.get("/").text
        assert "DOMPurify" in html
        assert "dompurify" in html.lower()

    def test_hmac_secret_not_api_key(self):
        from gateway.api_server import _MEDIA_SECRET
        api_key = os.getenv("API_KEY", "")
        # Media secret should be independently generated, not derived from API_KEY
        if api_key:
            assert _MEDIA_SECRET != api_key

    def test_hmac_secret_is_strong(self):
        from gateway.api_server import _MEDIA_SECRET
        assert len(_MEDIA_SECRET) >= 32  # At least 128 bits

    def test_error_does_not_leak_internals(self):
        adapter = _make_adapter()
        queue = adapter.register_queue("s1")
        session_key = adapter._build_session_key("s1")

        async def failing_parent(event, sk):
            raise RuntimeError("secret database password: abc123")

        event = MessageEvent(
            text="hi",
            source=SessionSource(
                platform=Platform.API, chat_id="s1", user_id="s1", chat_type="channel"
            ),
        )

        import asyncio
        with patch(
            "gateway.platforms.base.BasePlatformAdapter._process_message_background",
            side_effect=failing_parent,
        ):
            asyncio.get_event_loop().run_until_complete(
                adapter._process_message_background(event, session_key)
            )

        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())
        error_msgs = [m for m in messages if m["type"] == "error"]
        assert len(error_msgs) == 1
        assert "secret" not in error_msgs[0]["content"]
        assert "abc123" not in error_msgs[0]["content"]
        assert "internal error" in error_msgs[0]["content"].lower()

    def test_cors_middleware_configured(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        # Check CORS middleware exists in the middleware stack
        middleware_str = str(app.user_middleware)
        assert "CORSMiddleware" in middleware_str

    def test_ws_message_length_limit(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "ws-key"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "ws-key"})
                ws.receive_json()  # auth_ok
                ws.send_json({"message": "x" * 100_001})
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert "too long" in resp["content"].lower()

    def test_upload_streaming_read_rejects_large(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        large_data = b"x" * (26 * 1024 * 1024)
        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            resp = client.post(
                "/v1/upload",
                files={"file": ("big.bin", large_data, "application/octet-stream")},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 413

    def test_no_api_key_in_html_localstorage(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)
        html = client.get("/").text
        assert "hermes_api_key" not in html

    def test_history_no_duplicate_on_load(self):
        """loadHistory should not re-save messages."""
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)
        html = client.get("/").text
        # addUserMessage called from loadHistory should pass save=false
        assert "addUserMessage(m.content, false)" in html


# ── Rate limiting tests ──────────────────────────────────────────────────


class TestRateLimiter:
    def test_rate_limiter_allows_initial_requests(self):
        from gateway.api_server import _RateLimiter
        rl = _RateLimiter()
        # First 10 requests should pass
        for _ in range(10):
            assert rl.check("chat") is True

    def test_rate_limiter_blocks_after_limit(self):
        from gateway.api_server import _RateLimiter
        rl = _RateLimiter()
        # Exhaust all tokens
        for _ in range(10):
            rl.check("chat")
        # Next should be blocked
        assert rl.check("chat") is False

    def test_rate_limiter_unknown_endpoint_allowed(self):
        from gateway.api_server import _RateLimiter
        rl = _RateLimiter()
        assert rl.check("unknown_endpoint") is True

    def test_rate_limiter_refills_over_time(self):
        import time
        from gateway.api_server import _RateLimiter
        rl = _RateLimiter()
        # Exhaust tokens
        for _ in range(10):
            rl.check("chat")
        assert rl.check("chat") is False
        # Manually advance the bucket's last refill time
        rl._buckets["chat"][1] -= 10  # simulate 10 seconds passing
        assert rl.check("chat") is True  # should have refilled some tokens

    def test_chat_endpoint_returns_429_when_limited(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app, _rate_limiter

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

        # Reset rate limiter
        _rate_limiter._buckets.clear()

        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            # Send 10 requests (should pass)
            for i in range(10):
                resp = client.post("/v1/chat", json={"message": f"msg {i}"},
                                   headers={"Authorization": "Bearer test-key"})
                assert resp.status_code == 200

            # 11th should be rate limited
            resp = client.post("/v1/chat", json={"message": "too many"},
                               headers={"Authorization": "Bearer test-key"})
            assert resp.status_code == 429


# ── PWA tests ────────────────────────────────────────────────────────────


class TestPWA:
    def _client(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app
        return TestClient(create_app(_make_adapter()))

    def test_manifest_served(self):
        resp = self._client().get("/manifest.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Hermes Agent"
        assert data["theme_color"] == "#6c5ce7"
        assert data["display"] == "standalone"

    def test_icon_192_served(self):
        resp = self._client().get("/icons/icon-192.png")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]

    def test_icon_512_served(self):
        resp = self._client().get("/icons/icon-512.png")
        assert resp.status_code == 200

    def test_apple_touch_icon_served(self):
        resp = self._client().get("/icons/apple-touch-icon.png")
        assert resp.status_code == 200

    def test_html_has_manifest_link(self):
        html = self._client().get("/").text
        assert 'rel="manifest"' in html
        assert 'apple-mobile-web-app-capable' in html
        assert 'theme-color' in html

    def test_icon_path_traversal_blocked(self):
        resp = self._client().get("/icons/../../etc/passwd")
        assert resp.status_code != 200


# ── Edge case and security tests ─────────────────────────────────────────


class TestEdgeCases:
    """Tests for concurrent access, malformed input, and edge cases."""

    # 1. Concurrent REST with same session_id
    def test_concurrent_same_session_second_request_gets_response(self):
        """Two requests with same session_id should not crash."""
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app, _rate_limiter

        adapter = _make_adapter()
        call_count = 0

        async def fake_handle(sid, msg, user_id=None):
            nonlocal call_count
            call_count += 1
            key = adapter._build_session_key(sid)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": f"reply {call_count}"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)
        _rate_limiter._buckets.clear()

        with patch.dict(os.environ, {"API_KEY": "k"}):
            r1 = client.post("/v1/chat", json={"message": "a", "session_id": "same"},
                             headers={"Authorization": "Bearer k"})
            r2 = client.post("/v1/chat", json={"message": "b", "session_id": "same"},
                             headers={"Authorization": "Bearer k"})
        assert r1.status_code == 200
        assert r2.status_code == 200

    # 2. WS malformed JSON - server should not crash
    def test_ws_malformed_json_does_not_crash_server(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "k"}):
            try:
                with client.websocket_connect("/v1/chat/stream") as ws:
                    ws.send_json({"type": "auth", "token": "k"})
                    ws.receive_json()  # auth_ok
                    ws.send_text("not json at all {{{")
                    # Server may close connection or send error
            except Exception:
                pass  # Connection closed is acceptable
        # Key assertion: server is still running (no crash)
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    # 3. WS auth wrong message type
    def test_ws_auth_wrong_type_closes(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/chat/stream") as ws:
                    ws.send_json({"type": "message", "message": "hi"})
                    ws.receive_json()

    # 4. Rate limiter per-client
    def test_rate_limiter_per_client_independent(self):
        from gateway.api_server import _RateLimiter
        rl = _RateLimiter()
        # Exhaust client A
        for _ in range(10):
            rl.check("chat", "client-A")
        assert rl.check("chat", "client-A") is False
        # Client B should still be allowed
        assert rl.check("chat", "client-B") is True

    # 5. Media token invalid after restart (new secret)
    def test_media_token_tied_to_process_secret(self):
        from gateway.api_server import _sign_media_path, _MEDIA_SECRET
        token = _sign_media_path("/tmp/test.ogg")
        # Token is derived from _MEDIA_SECRET which is random per process
        assert len(token) == 64
        assert isinstance(token, str)

    # 7. CORS preflight
    def test_cors_options_request(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        resp = client.options("/v1/chat")
        # Should not crash - CORS middleware handles it
        assert resp.status_code in (200, 405)

    # 8. API_KEY empty
    def test_api_key_empty_returns_500(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": ""}):
            resp = client.post("/v1/chat", json={"message": "hi"},
                               headers={"Authorization": "Bearer anything"})
            assert resp.status_code == 500

    def test_api_key_not_set_returns_500(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("API_KEY", None)
            resp = client.post("/v1/chat", json={"message": "hi"},
                               headers={"Authorization": "Bearer anything"})
            assert resp.status_code == 500

    # 9. Session ID special chars rejected
    def test_session_id_special_chars_rejected(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app, _rate_limiter

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)
        _rate_limiter._buckets.clear()

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat",
                               json={"message": "hi", "session_id": "../../../etc/passwd"},
                               headers={"Authorization": "Bearer k"})
            assert resp.status_code == 422  # Pydantic validation error

    def test_session_id_too_long_rejected(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat",
                               json={"message": "hi", "session_id": "a" * 100},
                               headers={"Authorization": "Bearer k"})
            assert resp.status_code == 422

    # 12. Mixed media types in response
    @pytest.mark.asyncio
    async def test_mixed_media_response(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app, _rate_limiter

        adapter = _make_adapter()

        async def fake_handle(sid, msg, user_id=None):
            key = adapter._build_session_key(sid)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": "Look at these"})
                await q.put({"type": "image", "url": "http://example.com/img.png"})
                await q.put({"type": "audio", "url": "/v1/media/tok/audio.ogg"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)
        _rate_limiter._buckets.clear()

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "show me"},
                               headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Look at these"
        assert len(data["media"]) == 2
        types = [m["type"] for m in data["media"]]
        assert "image" in types
        assert "audio" in types

    # 15. Upload without file extension
    def test_upload_no_extension(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app

        adapter = _make_adapter()
        app = create_app(adapter)
        client = TestClient(app)

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/upload",
                               files={"file": ("noext", b"binary data", "application/octet-stream")},
                               headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "noext"
        assert "url" in data

    # WS rate limiting
    def test_ws_rate_limited(self):
        from fastapi.testclient import TestClient
        from gateway.api_server import create_app, _rate_limiter

        adapter = _make_adapter()

        async def fake_handle(sid, msg, user_id=None):
            key = adapter._build_session_key(sid)
            q = adapter._response_queues.get(key)
            if q:
                await q.put({"type": "message", "content": "ok"})
                await q.put({"type": "done"})

        adapter.handle_request = fake_handle
        app = create_app(adapter)
        client = TestClient(app)
        _rate_limiter._buckets.clear()

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k"})
                ws.receive_json()  # auth_ok

                # Exhaust rate limit
                for i in range(10):
                    ws.send_json({"message": f"msg {i}"})
                    while True:
                        r = ws.receive_json()
                        if r.get("type") in ("done", "error"):
                            break

                # Next should be rate limited
                ws.send_json({"message": "over limit"})
                r = ws.receive_json()
                assert r["type"] == "error"
                assert "rate" in r["content"].lower() or "limit" in r["content"].lower()


class TestToolsetWiring:
    def test_hermes_api_toolset_exists(self):
        from toolsets import TOOLSETS
        assert "hermes-api" in TOOLSETS

    def test_hermes_gateway_includes_api(self):
        from toolsets import TOOLSETS
        includes = TOOLSETS["hermes-gateway"]["includes"]
        assert "hermes-api" in includes
