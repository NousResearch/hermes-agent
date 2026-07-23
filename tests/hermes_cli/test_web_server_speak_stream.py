"""/api/audio/speak-stream — desktop streaming TTS over WebSocket."""

from __future__ import annotations

import json
import time
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


def _url(token: str | None = None) -> str:
    return f"/api/audio/speak-stream?{urlencode({'token': token or web_server._SESSION_TOKEN})}"


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


def test_rejects_bad_token(stream_client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with stream_client.websocket_connect(_url(token="wrong")):
            pass
    assert exc.value.code == 4401


def test_fallback_frame_when_no_streaming_provider(stream_client, monkeypatch):
    _patch_provider(monkeypatch, None)
    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json() == {"type": "fallback"}


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


def test_incremental_deltas_are_cut_into_sentences(stream_client, monkeypatch):
    """Text fed across frames is chunked and synthesized while more arrives."""
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "This is the first full"}))
        conn.send_text(json.dumps({"text": " sentence of the reply. And"}))
        # The first sentence is complete — PCM must arrive before `done`.
        assert conn.receive_bytes() == b"\x00\x00"
        conn.send_text(json.dumps({"text": " here is the second one.", "done": True}))
        assert conn.receive_bytes() == b"\x00\x00"
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == [
        "This is the first full sentence of the reply.",
        "And here is the second one.",
    ]


def test_idle_flush_speaks_narration_before_done(stream_client, monkeypatch):
    """A sentence-terminated buffer is spoken while the turn is still busy.

    The desktop keeps one session open for a whole agent turn and only sends
    `done` at the end. Narration like "Let me check." has no trailing
    whitespace, so the sentence cutter never fires on its own — the idle flush
    must speak it during the tool-execution silence, not after the turn.
    """
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "Let me check the config."}))
        # No `done` — PCM must still arrive via the idle flush.
        assert conn.receive_bytes() == b"\x00\x00"
        conn.send_text(json.dumps({"done": True}))
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == ["Let me check the config."]


def test_idle_flush_eventually_speaks_unterminated_text(stream_client, monkeypatch):
    """Text without sentence punctuation is force-flushed after a longer idle."""
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "Checking the wake-word branch now"}))
        assert conn.receive_bytes() == b"\x00\x00"
        conn.send_text(json.dumps({"done": True}))
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == ["Checking the wake-word branch now"]


def test_idle_flush_holds_open_think_block(stream_client, monkeypatch):
    """An unterminated <think> block is never flushed as speech."""
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "<think>secret reasoning."}))
        # Wait past the force-flush window; nothing may be synthesized.
        time.sleep(2.5)
        conn.send_text(json.dumps({"text": "</think>Answer ready.", "done": True}))
        assert conn.receive_bytes() == b"\x00\x00"
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == ["Answer ready."]


def test_stop_frame_cuts_synthesis(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"stop": True}))
        # Socket closes without an "end" frame — barge-in, not completion.
        with pytest.raises(WebSocketDisconnect):
            conn.receive_text()
    assert streamer.requests == []


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


def test_split_text_hard_splits_oversized_sentence():
    pieces = web_server._split_text_for_speak_stream("x" * 100, 30)
    assert all(len(piece) <= 30 for piece in pieces)
    assert sum(len(piece) for piece in pieces) == 100
