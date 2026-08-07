"""/api/audio/speak-stream — desktop streaming TTS over WebSocket."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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


def _url(token: str | None = None, *, audio_protocol: int | None = None) -> str:
    params = {"token": token or web_server._SESSION_TOKEN}
    if audio_protocol is not None:
        params["audio_protocol"] = str(audio_protocol)
    return f"/api/audio/speak-stream?{urlencode(params)}"


class _FakeStreamer:
    sample_rate = 24000
    channels = 1

    def __init__(self, chunks):
        self.chunks = chunks
        self.requests: list[str] = []

    def stream(self, text):
        self.requests.append(text)
        yield from self.chunks


def _patch_provider(monkeypatch, streamer, cap=4000):
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: cap)


def _patch_sync_edge_provider(monkeypatch):
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: None)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "edge")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 5000)

    def fake_tts(*, text, output_path, provider):
        path = Path(output_path)
        path.write_bytes(f"audio:{text}".encode())
        return json.dumps({"success": True, "file_path": str(path), "provider": provider})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)






def test_streams_pcm_frames_then_end(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        start = conn.receive_json()
        assert start == {"type": "start", "sample_rate": 24000, "channels": 1}

        conn.send_text(json.dumps({"text": "Hello there.", "done": True}))
        assert conn.receive_bytes() == b"\x01\x02\x03\x04"
        assert conn.receive_bytes() == b"\x05\x06"
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == ["Hello there."]


def test_sync_edge_streams_first_sentence_before_reply_finishes(stream_client, monkeypatch):
    _patch_sync_edge_provider(monkeypatch)

    with stream_client.websocket_connect(_url(audio_protocol=2)) as conn:
        assert conn.receive_json() == {"type": "start", "encoding": "encoded"}

        conn.send_text(json.dumps({"text": "The first sentence is ready. "}))
        assert conn.receive_bytes() == b"audio:The first sentence is ready."

        conn.send_text(json.dumps({"text": "The second sentence follows.", "done": True}))
        assert conn.receive_bytes() == b"audio:The second sentence follows."
        assert conn.receive_json() == {"type": "end"}


def test_sync_provider_falls_back_for_legacy_desktop_client(stream_client, monkeypatch):
    _patch_sync_edge_provider(monkeypatch)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json() == {"type": "fallback"}


def test_sync_provider_failure_before_audio_requests_fallback(stream_client, monkeypatch):
    _patch_sync_edge_provider(monkeypatch)
    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda **_kwargs: json.dumps({"success": False, "error": "synthetic failure"}),
    )

    with stream_client.websocket_connect(_url(audio_protocol=2)) as conn:
        assert conn.receive_json() == {"type": "start", "encoding": "encoded"}
        conn.send_text(json.dumps({"text": "This will fail.", "done": True}))
        assert conn.receive_json() == {"type": "fallback"}


def test_sync_provider_failure_after_audio_ends_without_replaying(stream_client, monkeypatch):
    _patch_sync_edge_provider(monkeypatch)
    calls = 0

    def fail_second_sentence(*, text, output_path, provider):
        nonlocal calls
        calls += 1
        if calls == 2:
            return json.dumps({"success": False, "error": "synthetic failure"})
        path = Path(output_path)
        path.write_bytes(f"audio:{text}".encode())
        return json.dumps({"success": True, "file_path": str(path), "provider": provider})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fail_second_sentence)

    with stream_client.websocket_connect(_url(audio_protocol=2)) as conn:
        assert conn.receive_json() == {"type": "start", "encoding": "encoded"}
        conn.send_text(json.dumps({"text": "The first sentence works. "}))
        assert conn.receive_bytes() == b"audio:The first sentence works."
        conn.send_text(json.dumps({"text": "The second sentence fails.", "done": True}))
        assert conn.receive_json() == {"type": "end"}








def test_long_text_is_split_across_provider_requests(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer, cap=24)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(
            json.dumps(
                {"text": "First sentence here. Second sentence here. Third one.", "done": True}
            )
        )
        # One PCM frame per split piece, then end.
        frames = 0
        while True:
            message = conn.receive()
            if message.get("bytes") is not None:
                frames += 1
            else:
                assert json.loads(message["text"]) == {"type": "end"}
                break

    assert len(streamer.requests) > 1
    assert frames == len(streamer.requests)
    # Nothing lost in the split: every sentence reached the provider.
    joined = " ".join(streamer.requests)
    for fragment in ("First sentence here.", "Second sentence here.", "Third one."):
        assert fragment in joined


def test_split_text_respects_cap_and_preserves_content():
    text = "Alpha beta. Gamma delta epsilon. Zeta eta theta iota kappa."
    pieces = web_server._split_text_for_speak_stream(text, 30)
    assert pieces
    assert all(len(piece) <= 30 for piece in pieces)
    joined = " ".join(pieces)
    for word in text.replace(".", "").split():
        assert word in joined


