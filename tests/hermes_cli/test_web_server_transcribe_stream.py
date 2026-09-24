"""``/api/audio/transcribe-stream`` — live partial-transcript STT over WebSocket.

Covers the provider-neutral seam contract: the endpoint resolves the active streaming
transcription provider by ``stt.provider`` (registered via the plugin channel), speaks the
``ready`` / ``partial`` / ``final`` / ``error`` / ``unsupported`` wire protocol, and never
binds to a concrete vendor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient

from agent.streaming_transcription_provider import StreamingTranscriptionProvider
from hermes_cli import web_server


@pytest.fixture
def ws_client(monkeypatch, _isolate_hermes_home):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False

    client = TestClient(web_server.app)
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
        if previous_auth_required is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth_required


class _FakeStreamingProvider(StreamingTranscriptionProvider):
    name = "fake_stream"

    def __init__(self):
        self.calls: list[Any] = []

    def start(self, *, language: Optional[str] = None,
              config: Optional[Dict[str, Any]] = None) -> str:
        self.calls.append(("start", config))
        return "key-1"

    def feed(self, stream_key: str, pcm: bytes) -> str:
        self.calls.append(("feed", stream_key, len(pcm)))
        return "partial text"

    def finish(self, stream_key: str) -> str:
        self.calls.append(("finish", stream_key))
        return "full transcript"

    def cancel(self, stream_key: str) -> None:
        self.calls.append(("cancel", stream_key))


@pytest.fixture
def fake_provider():
    provider = _FakeStreamingProvider()
    from agent import streaming_transcription_registry
    streaming_transcription_registry._reset_for_tests()
    streaming_transcription_registry.register_provider(provider)
    yield provider
    streaming_transcription_registry._reset_for_tests()


def _monkey_patch_stt(monkeypatch):
    monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: {"provider": "fake_stream"})
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **k: None)


def _url() -> str:
    return f"/api/audio/transcribe-stream?{urlencode({'token': web_server._SESSION_TOKEN})}"


def test_transcribe_stream_ready_partial_final_protocol(ws_client, fake_provider, monkeypatch):
    _monkey_patch_stt(monkeypatch)
    with ws_client.websocket_connect(_url()) as ws:
        assert ws.receive_json() == {"type": "ready", "sample_rate": 16000}
        ws.send_bytes(b"\x00" * 320)
        assert ws.receive_json() == {"type": "partial", "text": "partial text"}
        ws.send_text("finish")
        assert ws.receive_json() == {"type": "final", "text": "full transcript"}
    assert fake_provider.calls[0][0] == "start"
    assert isinstance(fake_provider.calls[0][1], dict)
    assert "finish" in [c[0] for c in fake_provider.calls]


def test_transcribe_stream_unsupported_without_provider(ws_client, monkeypatch):
    from agent import streaming_transcription_registry
    streaming_transcription_registry._reset_for_tests()
    _monkey_patch_stt(monkeypatch)
    with ws_client.websocket_connect(_url()) as ws:
        assert ws.receive_json() == {"type": "unsupported"}
        ws.send_text("cancel")


def test_transcribe_stream_cancel_aborts_without_final(ws_client, fake_provider, monkeypatch):
    _monkey_patch_stt(monkeypatch)
    with ws_client.websocket_connect(_url()) as ws:
        assert ws.receive_json() == {"type": "ready", "sample_rate": 16000}
        ws.send_text("cancel")
    assert "cancel" in [c[0] for c in fake_provider.calls]
    assert "finish" not in [c[0] for c in fake_provider.calls]