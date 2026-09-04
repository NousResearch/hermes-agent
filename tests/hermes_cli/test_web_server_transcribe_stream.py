"""/api/audio/transcribe-stream — live STT for desktop dictation over WebSocket."""

from __future__ import annotations

import json
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def stream_client(monkeypatch, _isolate_hermes_home):
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


def _url(token: str | None = None) -> str:
    return f"/api/audio/transcribe-stream?{urlencode({'token': token or web_server._SESSION_TOKEN})}"


class _FakeSession:
    """Streaming session contract the endpoint drives (16 kHz s16le PCM in)."""

    def __init__(self):
        self.pushed: list[bytes] = []
        self.finalized = False
        self.partial = ""

    def push_audio(self, chunk: bytes) -> None:
        self.pushed.append(chunk)

    def partial_transcript(self) -> str:
        return self.partial

    def finalize(self):
        self.finalized = True
        text = "".join(c.decode("utf-8", errors="ignore") for c in self.pushed)
        if text == "fail":
            return {"success": False, "transcript": "", "provider": "fake", "error": "boom"}
        return {"success": True, "transcript": f"heard: {text}", "provider": "fake"}


def _patch_streaming(monkeypatch, session: _FakeSession):
    """Route the endpoint's resolution to a fake streaming-capable provider."""
    monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: {})
    monkeypatch.setattr("tools.transcription_tools.is_stt_enabled", lambda cfg: True)
    monkeypatch.setattr("tools.transcription_tools._get_provider", lambda cfg: "fake-stream")
    monkeypatch.setattr("tools.transcription_tools._resolve_stt_language", lambda provider, cfg: "pt")
    monkeypatch.setattr(
        "agent.transcription_registry.open_streaming_session",
        lambda provider, language=None: session if provider == "fake-stream" else None,
    )


def test_streams_pcm_and_returns_final(stream_client, monkeypatch):
    session = _FakeSession()
    _patch_streaming(monkeypatch, session)

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(json.dumps({"sample_rate": 16000}))
        conn.send_bytes(b"ola")
        conn.send_bytes(b" mundo")
        conn.send_text(json.dumps({"eos": True}))

        final = conn.receive_json()
        assert final["type"] == "final"
        assert final["success"] is True
        assert final["transcript"] == "heard: ola mundo"
        assert final["provider"] == "fake"

    assert session.pushed == [b"ola", b" mundo"]
    assert session.finalized is True


def test_rejects_non_16000_sample_rate(stream_client, monkeypatch):
    _patch_streaming(monkeypatch, _FakeSession())

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(json.dumps({"sample_rate": 48000}))
        error = conn.receive_json()
        assert error["type"] == "error"
        assert "16000" in error["message"]


def test_error_envelope_when_provider_has_no_streaming(stream_client, monkeypatch):
    _patch_streaming(monkeypatch, _FakeSession())
    monkeypatch.setattr(
        "agent.transcription_registry.open_streaming_session",
        lambda provider, language=None: None,
    )

    with stream_client.websocket_connect(_url()) as conn:
        error = conn.receive_json()
        assert error["type"] == "error"
        assert "streaming" in error["message"]


def test_error_envelope_when_stt_disabled(stream_client, monkeypatch):
    monkeypatch.setattr("tools.transcription_tools.is_stt_enabled", lambda cfg: False)

    with stream_client.websocket_connect(_url()) as conn:
        error = conn.receive_json()
        assert error["type"] == "error"


def test_failure_envelope_surfaces_provider_error(stream_client, monkeypatch):
    _patch_streaming(monkeypatch, _FakeSession())

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(json.dumps({"sample_rate": 16000}))
        conn.send_bytes(b"fail")
        conn.send_text(json.dumps({"eos": True}))

        error = conn.receive_json()
        assert error["type"] == "error"
        assert error["message"] == "boom"
