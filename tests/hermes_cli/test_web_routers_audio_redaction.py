"""Dashboard audio routes never relay provider credentials to the HTTP/WS client (#37889).

STT/TTS/GPT-Live providers echo the submitted key or bearer in their error text; every
path that turns that text into a response body must pass it through forced redaction.
"""

from __future__ import annotations

import base64
import json
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server

# Built at runtime so no credential-shaped literal sits in the tree.
_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
_BEARER = "Bearer " + "abcdefghijklmnopqrstuvwxyz0123"
_LEAK = f"provider rejected key {_KEY} (Authorization: {_BEARER})"
_AUDIO = {"data_url": "data:audio/webm;base64," + base64.b64encode(b"\x00fake").decode()}


@pytest.fixture
def client(_isolate_hermes_home):
    c = TestClient(web_server.app)
    c.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return c


def _assert_redacted(text: str) -> None:
    assert _KEY not in text
    assert "abcdefghijklmnopqrstuvwxyz0123" not in text


def _raise(*_a, **_k):
    raise RuntimeError(_LEAK)


def test_transcribe_failed_result_is_redacted(client, monkeypatch):
    monkeypatch.setattr("hermes_agent.tools.voice_mode.transcribe_recording",
                        lambda _p: {"success": False, "error": _LEAK})
    resp = client.post("/api/audio/transcribe", json=_AUDIO)
    assert resp.status_code == 400
    _assert_redacted(resp.json()["detail"])
    assert "provider rejected key" in resp.json()["detail"]


def test_transcribe_exception_is_redacted(client, monkeypatch):
    monkeypatch.setattr("hermes_agent.tools.voice_mode.transcribe_recording", _raise)
    resp = client.post("/api/audio/transcribe", json=_AUDIO)
    assert resp.status_code == 500
    _assert_redacted(resp.json()["detail"])


def test_speak_failed_result_is_redacted(client, monkeypatch):
    monkeypatch.setattr("hermes_agent.tools.tts_tool.text_to_speech_tool",
                        lambda _t: json.dumps({"success": False, "error": _LEAK}))
    resp = client.post("/api/audio/speak", json={"text": "hello"})
    assert resp.status_code == 400
    _assert_redacted(resp.json()["detail"])


def test_speak_exception_is_redacted(client, monkeypatch):
    monkeypatch.setattr("hermes_agent.tools.tts_tool.text_to_speech_tool", _raise)
    resp = client.post("/api/audio/speak", json={"text": "hello"})
    assert resp.status_code == 500
    _assert_redacted(resp.json()["detail"])


def test_voice_live_session_vendor_error_is_redacted(client, monkeypatch):
    monkeypatch.setattr("hermes_agent.tools.voice_live.create_webrtc_session", _raise)
    resp = client.post("/api/audio/voice-live/session", json={"sdp": "v=0\r\n"})
    assert resp.status_code == 502
    _assert_redacted(resp.json()["detail"])


@pytest.mark.parametrize("route, target", [
    ("/api/audio/tts-lease", "hermes_agent.tools.tts_tool_lifecycle.acquire_tts_lease"),
    ("/api/audio/stt-lease", "hermes_agent.tools.stt_lease.acquire_stt_lease"),
])
def test_lease_warmup_error_body_is_redacted(client, monkeypatch, route, target):
    monkeypatch.setattr(target, _raise)
    resp = client.post(route, json={"lease": "t", "active": True})
    assert resp.status_code == 200
    assert resp.json()["action"] == "error"
    _assert_redacted(resp.json()["error"])


@pytest.mark.parametrize("route, target", [
    ("/api/audio/tts-lease", "hermes_agent.tools.tts_tool_lifecycle.warm_tts_provider"),
    ("/api/audio/stt-lease", "hermes_agent.tools.stt_lease.warm_stt_provider"),
])
def test_lease_returned_warmup_error_is_redacted(client, monkeypatch, route, target):
    # Warm-up helpers never raise: a load failure comes back as ``error=str(exc)``.
    monkeypatch.setattr(target, lambda *_a, **_k: {"provider": "x", "warmed": False,
                                                    "action": "error", "error": _LEAK})
    resp = client.post(route, json={"lease": "t", "active": True})
    assert resp.status_code == 200
    assert resp.json()["action"] == "error"
    _assert_redacted(resp.json()["error"])
    assert "provider rejected key" in resp.json()["error"]


def test_transcribe_stream_error_frame_is_redacted(client, monkeypatch):
    class _Session:
        set_input_rate = push_audio = cancel = lambda self, *_: None

        def finalize(self):
            return {"success": False, "error": _LEAK}

    monkeypatch.setattr("hermes_agent.tools.transcription_streaming.open_streaming_session",
                        lambda on_partial: _Session())
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    try:
        url = f"/api/audio/transcribe-stream?{urlencode({'token': web_server._SESSION_TOKEN})}"
        with client.websocket_connect(url) as ws:
            ws.send_text(json.dumps({"sample_rate": 16000}))
            ws.send_text(json.dumps({"eos": True}))
            frame = ws.receive_json()
    finally:
        web_server.app.state.auth_required = previous
    assert frame["type"] == "error"
    _assert_redacted(frame["message"])
