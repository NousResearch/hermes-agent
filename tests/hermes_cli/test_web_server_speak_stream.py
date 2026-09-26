"""/api/audio/speak-stream — desktop streaming TTS over WebSocket."""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server
import hermes_cli.web_server_gateway as _web_server_gateway


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


class _RateLearningStreamer(_FakeStreamer):
    """Mimics the OpenAI-compatible streamer: the true PCM rate is only known once
    the endpoint's response headers arrive inside stream()."""

    def stream(self, text):
        self.sample_rate = 44100
        yield from super().stream(text)


def test_start_frame_carries_rate_learned_during_first_stream(stream_client, monkeypatch):
    streamer = _RateLearningStreamer([b"\x01\x02"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(json.dumps({"text": "Hello there.", "done": True}))
        assert conn.receive_json() == {"type": "start", "sample_rate": 44100, "channels": 1}
        assert conn.receive_bytes() == b"\x01\x02"
        assert conn.receive_json() == {"type": "end"}


def test_streams_pcm_frames_then_end(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(json.dumps({"text": "Hello there.", "done": True}))
        start = conn.receive_json()
        assert start == {"type": "start", "sample_rate": 24000, "channels": 1}

        assert conn.receive_bytes() == b"\x01\x02\x03\x04"
        assert conn.receive_bytes() == b"\x05\x06"
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == ["Hello there."]


def test_short_cjk_opener_is_synthesized_alone_with_configured_min_len(stream_client, monkeypatch):
    """speak_stream_ws cuts with the requesting profile's tts.streaming.min_len (#96927): a 7-char
    CJK opener gets its own provider request instead of riding behind the second sentence."""
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {"streaming": {"min_len": 6}})

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(json.dumps({"text": "记得，叫团团. 然后我们再说第二句话，这一句要长一些才行. ", "done": True}))
        # The start frame is deferred until the first PCM chunk (rate learned from the endpoint).
        assert conn.receive_json()["type"] == "start"
        while True:
            message = conn.receive()
            if message.get("bytes") is None:
                assert json.loads(message["text"]) == {"type": "end"}
                break

    assert streamer.requests[0] == "记得，叫团团.", streamer.requests








def test_long_text_is_split_across_provider_requests(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer, cap=24)

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(
            json.dumps(
                {"text": "First sentence here. Second sentence here. Third one.", "done": True}
            )
        )
        assert conn.receive_json()["type"] == "start"
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
    pieces = _web_server_gateway._split_text_for_speak_stream(text, 30)
    assert pieces
    assert all(len(piece) <= 30 for piece in pieces)
    joined = " ".join(pieces)
    for word in text.replace(".", "").split():
        assert word in joined


def test_edge_speak_stream_speaks_each_sentence_instead_of_whole_text_fallback(
    stream_client, monkeypatch, tmp_path
):
    """Edge has no chunked API, but the documented path is per-sentence sync
    synthesis. speak-stream must stream that PCM — not ``type: fallback``,
    which makes Desktop POST the whole reply and wait for it.
    """
    first = "The first spoken sentence is long enough to stand alone."
    second = "The second spoken sentence is also long enough to stand alone."
    reply = f"{first} {second}"
    pcm = [b"\x11\x00\x22\x00", b"\x33\x00\x44\x00"]
    calls: list[str] = []
    pcm_for: dict[str, bytes] = {}

    def fake_tts(*args, **kwargs):
        raw = kwargs.get("text") if "text" in kwargs else (args[0] if args else "")
        text = raw if isinstance(raw, str) else ""
        calls.append(text)
        path = kwargs.get("output_path") or str(tmp_path / f"sent-{len(calls)}.mp3")
        # Edge's sync tool writes MP3 bytes, not a WAV container.
        with open(path, "wb") as fh:
            fh.write(b"ID3not-a-wav")
        pcm_for[path] = pcm[len(calls) - 1]
        return json.dumps({"success": True, "file_path": path, "file_paths": [path]})

    def fake_ffmpeg(_ffmpeg, args, **_kwargs):
        import subprocess

        path = args[args.index("-i") + 1]
        assert "s16le" in args and "pipe:1" in args
        return subprocess.CompletedProcess(args, 0, stdout=pcm_for[path], stderr=b"")

    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: None)
    monkeypatch.setattr(
        "tools.tts_tool._load_tts_config",
        lambda: {"provider": "edge", "streaming": {"min_len": 6}},
    )
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "edge")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 4000)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)
    monkeypatch.setattr("tools.tts_tool_delivery._ffmpeg_run", fake_ffmpeg)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    with stream_client.websocket_connect(_url()) as conn:
        conn.send_text(json.dumps({"text": reply, "done": True}))
        frames = []
        while True:
            message = conn.receive()
            if message.get("bytes") is not None:
                frames.append(message["bytes"])
                continue
            frames.append(json.loads(message["text"]))
            if frames[-1].get("type") in {"end", "fallback"}:
                break

    assert {"type": "fallback"} not in frames
    assert frames[0] == {"type": "start", "sample_rate": 24000, "channels": 1}
    assert frames[-1] == {"type": "end"}
    assert frames[1:-1] == pcm
    assert len(calls) == 2
    assert first in calls[0] and second not in calls[0]
    assert second in calls[1] and first not in calls[1]
    assert reply not in calls


