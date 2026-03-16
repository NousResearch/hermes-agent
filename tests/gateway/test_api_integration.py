"""Integration tests for the REST/WebSocket API.

Spec-driven tests: each test defines WHAT SHOULD HAPPEN, not what currently happens.
Known bugs are marked with pytest.xfail("reason") so they document the desired
behavior while acknowledging the current limitation.

Tests use real APIPlatformAdapter with real queue management.
Only handle_message is mocked to provide controlled responses.
"""

import asyncio
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api import APIPlatformAdapter
from gateway.platforms.base import MessageType


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Reset rate limiter state between tests."""
    from gateway.api_server import _rate_limiter
    _rate_limiter._buckets.clear()


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_adapter() -> APIPlatformAdapter:
    config = PlatformConfig(enabled=True)
    return APIPlatformAdapter(config)


def _make_app(adapter):
    from gateway.api_server import create_app
    return create_app(adapter)


def _wire_agent(adapter, response_text="Hello!", media=None):
    """Mock handle_message to respond via real queue system."""
    captured_events = []

    async def fake_handle_message(event):
        captured_events.append(event)
        session_key = adapter._build_session_key(event.source.chat_id)
        queue = adapter._response_queues.get(session_key)
        if queue:
            if response_text:
                await queue.put({"type": "message", "content": response_text})
            if media:
                for m in media:
                    await queue.put(m)
            await queue.put({"type": "done"})

    adapter.handle_message = fake_handle_message
    return captured_events


# ═══════════════════════════════════════════════════════════════════════
# SPEC: A user sends a text message and receives a complete response
# ═══════════════════════════════════════════════════════════════════════


class TestChatContract:
    """POST /v1/chat SHOULD return the agent's full response with session ID."""

    def test_should_return_agent_response_with_session_id(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "Hello!")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "Hi", "session_id": "s1"},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200
        assert resp.json()["response"] == "Hello!"
        assert resp.json()["session_id"] == "s1"

    def test_should_include_media_in_response(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        media = [{"type": "image", "url": "/img.png", "caption": "pic"}]
        _wire_agent(adapter, "Here", media=media)
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "show"},
                               headers={"Authorization": "Bearer k"})

        assert len(resp.json()["media"]) == 1
        assert resp.json()["media"][0]["type"] == "image"

    def test_should_generate_unique_session_id_when_not_provided(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            r1 = client.post("/v1/chat", json={"message": "a"}, headers={"Authorization": "Bearer k"})
            r2 = client.post("/v1/chat", json={"message": "b"}, headers={"Authorization": "Bearer k"})

        assert r1.json()["session_id"] != r2.json()["session_id"]

    def test_should_return_empty_response_without_error(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "x"},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200
        assert resp.json()["response"] == ""

    def test_should_cleanup_queue_after_response(self):
        """No orphaned queues after a request completes."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            client.post("/v1/chat", json={"message": "x", "session_id": "c1"},
                        headers={"Authorization": "Bearer k"})

        assert adapter._build_session_key("c1") not in adapter._response_queues


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Sessions SHOULD be isolated from each other
# ═══════════════════════════════════════════════════════════════════════


class TestSessionIsolationContract:

    def test_should_not_mix_responses_between_sessions(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        n = {"c": 0}

        async def handler(event):
            n["c"] += 1
            sk = adapter._build_session_key(event.source.chat_id)
            q = adapter._response_queues.get(sk)
            if q:
                await q.put({"type": "message", "content": f"Reply {n['c']}"})
                await q.put({"type": "done"})

        adapter.handle_message = handler
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            r1 = client.post("/v1/chat", json={"message": "a", "session_id": "s1"},
                             headers={"Authorization": "Bearer k"})
            r2 = client.post("/v1/chat", json={"message": "b", "session_id": "s2"},
                             headers={"Authorization": "Bearer k"})

        assert r1.json()["session_id"] == "s1"
        assert r2.json()["session_id"] == "s2"
        # Each session gets its own response, not mixed
        assert "1" in r1.json()["response"]
        assert "2" in r2.json()["response"]

    def test_concurrent_same_session_should_return_409(self):
        """Second concurrent request on busy session SHOULD get 409 Conflict."""
        from fastapi.testclient import TestClient
        import concurrent.futures
        import time

        adapter = _make_adapter()

        async def slow_handler(event):
            await asyncio.sleep(0.5)  # simulate slow processing
            sk = adapter._build_session_key(event.source.chat_id)
            q = adapter._response_queues.get(sk)
            if q:
                await q.put({"type": "message", "content": "ok"})
                await q.put({"type": "done"})

        adapter.handle_message = slow_handler
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(lambda: client.post("/v1/chat",
                    json={"message": "a", "session_id": "shared"},
                    headers={"Authorization": "Bearer k"}))
                time.sleep(0.05)  # ensure first request registers queue first
                f2 = pool.submit(lambda: client.post("/v1/chat",
                    json={"message": "b", "session_id": "shared"},
                    headers={"Authorization": "Bearer k"}))

                r1 = f1.result(timeout=10)
                r2 = f2.result(timeout=10)

        # One succeeds, the other gets 409
        codes = sorted([r1.status_code, r2.status_code])
        assert codes == [200, 409]


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Voice messages SHOULD be treated as MessageType.VOICE
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceContract:

    def test_voice_endpoint_should_set_message_type_voice(self):
        """Voice input SHOULD be marked as VOICE so auto-TTS triggers."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        events = _wire_agent(adapter, "heard you")
        client = TestClient(_make_app(adapter))

        fake_result = {"success": True, "transcript": "hello",
                       "language": "en", "language_probability": 0.9}

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            resp = client.post("/v1/chat/voice",
                               files={"file": ("v.webm", b"audio", "audio/webm")},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200
        assert events[0].message_type == MessageType.VOICE

    def test_chat_endpoint_should_not_set_voice_type(self):
        """Text chat SHOULD NOT trigger auto-TTS (unless voice mode is 'all')."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        events = _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            client.post("/v1/chat", json={"message": "hi"},
                        headers={"Authorization": "Bearer k"})

        assert events[0].message_type != MessageType.VOICE

    def test_voice_should_forward_transcript_as_message_text(self):
        """Transcribed text SHOULD become the event's text field."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        events = _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        fake_result = {"success": True, "transcript": "what is the weather",
                       "language": "en", "language_probability": 0.95}

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            client.post("/v1/chat/voice",
                        files={"file": ("v.ogg", b"data", "audio/ogg")},
                        headers={"Authorization": "Bearer k"})

        assert events[0].text == "what is the weather"

    def test_voice_transcription_failure_should_return_422(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "should not reach")
        client = TestClient(_make_app(adapter))

        async def fake_to_thread(fn, *args, **kwargs):
            return {"success": False, "error": "STT failed"}

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            resp = client.post("/v1/chat/voice",
                               files={"file": ("v.ogg", b"bad", "audio/ogg")},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
# SPEC: WebSocket SHOULD stream response chunks in real-time
# ═══════════════════════════════════════════════════════════════════════


class TestWebSocketContract:

    def test_should_stream_chunks_then_done(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "streamed reply")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k", "session_id": "ws1"})
                assert ws.receive_json()["type"] == "auth_ok"

                ws.send_json({"message": "hello"})
                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] == "done":
                        break

        assert any(m.get("content") == "streamed reply" for m in messages)
        assert messages[-1]["type"] == "done"
        assert messages[-1]["session_id"] == "ws1"

    def test_should_support_multiple_messages_on_same_connection(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        n = {"c": 0}

        async def handler(event):
            n["c"] += 1
            sk = adapter._build_session_key(event.source.chat_id)
            q = adapter._response_queues.get(sk)
            if q:
                await q.put({"type": "message", "content": f"reply {n['c']}"})
                await q.put({"type": "done"})

        adapter.handle_message = handler
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k"})
                ws.receive_json()

                ws.send_json({"message": "msg1"})
                r1 = []
                while True:
                    m = ws.receive_json()
                    r1.append(m)
                    if m["type"] == "done":
                        break

                ws.send_json({"message": "msg2"})
                r2 = []
                while True:
                    m = ws.receive_json()
                    r2.append(m)
                    if m["type"] == "done":
                        break

        assert any(m.get("content") == "reply 1" for m in r1)
        assert any(m.get("content") == "reply 2" for m in r2)

    def test_should_cleanup_queue_after_ws_message(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k", "session_id": "wsc"})
                ws.receive_json()
                ws.send_json({"message": "test"})
                while ws.receive_json()["type"] != "done":
                    pass

        assert adapter._build_session_key("wsc") not in adapter._response_queues


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Uploaded files SHOULD be downloadable with valid HMAC token
# ═══════════════════════════════════════════════════════════════════════


class TestMediaContract:

    def test_uploaded_file_should_be_downloadable(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            up = client.post("/v1/upload",
                             files={"file": ("doc.txt", b"content", "text/plain")},
                             headers={"Authorization": "Bearer k"})
            url = up.json()["url"]
            down = client.get(url)

        assert down.status_code == 200
        assert down.content == b"content"

    def test_should_sanitize_upload_filename(self):
        """Path traversal in filename SHOULD be stripped."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/upload",
                               files={"file": ("../../etc/passwd", b"x", "text/plain")},
                               headers={"Authorization": "Bearer k"})

        assert resp.json()["filename"] == "passwd"

    def test_forged_token_should_be_rejected(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        resp = client.get("/v1/media/forged_token/secret.txt")
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Auth SHOULD protect all endpoints and leak nothing
# ═══════════════════════════════════════════════════════════════════════


class TestAuthContract:

    def test_wrong_key_should_return_401(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "correct"}):
            resp = client.post("/v1/chat", json={"message": "hi"},
                               headers={"Authorization": "Bearer wrong"})

        assert resp.status_code == 401

    def test_wrong_key_should_not_create_queue(self):
        """Failed auth SHOULD NOT leave any server-side state."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "should not reach")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "correct"}):
            client.post("/v1/chat", json={"message": "hi"},
                        headers={"Authorization": "Bearer wrong"})

        assert len(adapter._response_queues) == 0

    def test_health_should_not_require_auth(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ws_wrong_key_should_close_connection(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "correct"}):
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/chat/stream") as ws:
                    ws.send_json({"type": "auth", "token": "wrong"})
                    ws.receive_json()


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Interrupt SHOULD stop a running agent
# ═══════════════════════════════════════════════════════════════════════


class TestInterruptContract:

    def test_should_interrupt_active_session(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        session_key = adapter._build_session_key("active-1")
        adapter._active_sessions[session_key] = asyncio.Event()

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat/interrupt?session_id=active-1",
                               headers={"Authorization": "Bearer k"})

        assert resp.json()["interrupted"] is True
        assert adapter._active_sessions[session_key].is_set()

    def test_should_report_no_active_session(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat/interrupt?session_id=none",
                               headers={"Authorization": "Bearer k"})

        assert resp.json()["interrupted"] is False


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Rate limiting SHOULD protect the server without leaking resources
# ═══════════════════════════════════════════════════════════════════════


class TestRateLimitContract:

    def test_should_return_429_after_limit(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            for _ in range(10):
                client.post("/v1/chat", json={"message": "x"},
                            headers={"Authorization": "Bearer k"})
            resp = client.post("/v1/chat", json={"message": "over"},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 429

    def test_rate_limit_should_not_leak_queues(self):
        """429 response SHOULD NOT leave orphaned queues."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            for _ in range(11):
                client.post("/v1/chat", json={"message": "x"},
                            headers={"Authorization": "Bearer k"})

        assert len(adapter._response_queues) == 0
