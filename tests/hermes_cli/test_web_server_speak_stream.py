"""/api/audio/speak-stream — desktop streaming TTS over WebSocket."""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server
import tools.tts_streaming as tts_streaming


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
    monkeypatch.setattr(
        "tools.tts_tool._resolve_max_text_length",
        lambda provider, cfg, **_kwargs: cap,
    )


def _register_provider(monkeypatch, name, stream_impl=None):
    class _RegisteredStreamer(tts_streaming.StreamingTTSProvider):
        instances = []

        def __init__(self, tts_config, section):
            super().__init__(tts_config, section)
            self.requests = []
            type(self).instances.append(self)

        @staticmethod
        def available():
            return True

        def stream(self, text):
            self.requests.append(text)
            if stream_impl is None:
                yield b"\x00\x00"
            else:
                yield from stream_impl(text)

    monkeypatch.setitem(tts_streaming._REGISTRY, name, _RegisteredStreamer)
    return _RegisteredStreamer



def test_fallback_frame_when_no_streaming_provider(stream_client, monkeypatch):
    _patch_provider(monkeypatch, None, cap=1000)
    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json() == {"type": "fallback", "max_text_length": 1000}


def test_explicit_streaming_pin_uses_resolved_cap_not_sync_fallback_cap(
    stream_client,
    monkeypatch,
):
    provider_cls = _register_provider(monkeypatch, "stream-b")
    config = {
        "provider": "sync-a",
        "sync-a": {"max_text_length": 50},
        "stream-b": {"max_text_length": 17},
        "streaming": {"provider": "stream-b"},
    }
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)
    text = "x" * 40

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": text, "done": True}))
        assert [len(conn.receive_bytes()) for _ in range(3)] == [2, 2, 2]
        assert conn.receive_json() == {"type": "end"}

    streamer = provider_cls.instances[0]
    assert [len(request) for request in streamer.requests] == [17, 17, 6]
    assert "".join(streamer.requests) == text


@pytest.mark.parametrize(
    "stream_impl",
    [
        pytest.param(lambda _text: iter(()), id="returns-without-audio"),
        pytest.param(lambda _text: iter((b"", b"")), id="empty-audio-only"),
        pytest.param(
            lambda _text: (_ for _ in ()).throw(RuntimeError("pre-audio failure")),
            id="raises-before-audio",
        ),
    ],
)
def test_pre_audio_provider_failure_or_zero_yield_emits_sync_fallback(
    stream_client,
    monkeypatch,
    stream_impl,
):
    _register_provider(monkeypatch, "stream-b", stream_impl=stream_impl)
    config = {
        "provider": "sync-a",
        "sync-a": {"max_text_length": 50},
        "stream-b": {"max_text_length": 17},
        "streaming": {"provider": "stream-b"},
    }
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "attempted input", "done": True}))
        assert conn.receive_json() == {"type": "fallback", "max_text_length": 50}


def test_auto_streamer_cap_controls_requests_but_pre_audio_fallback_uses_sync_cap(
    stream_client,
    monkeypatch,
):
    provider_cls = _register_provider(
        monkeypatch,
        "stream-b",
        stream_impl=lambda _text: iter(()),
    )
    monkeypatch.setattr(
        tts_streaming,
        "_PROVIDER_PRIORITY",
        ["missing-streamer", "stream-b"],
    )
    config = {
        "provider": "sync-a",
        "sync-a": {"max_text_length": 50},
        "stream-b": {"max_text_length": 17},
        "streaming": {"provider": "auto"},
    }
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)
    text = "x" * 40

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": text, "done": True}))
        assert conn.receive_json() == {"type": "fallback", "max_text_length": 50}

    streamer = provider_cls.instances[0]
    assert streamer.provider_id == "stream-b"
    assert [len(request) for request in streamer.requests] == [17, 17, 6]
    assert "".join(streamer.requests) == text


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


def test_post_audio_provider_failure_filters_empty_chunks_and_ends_without_fallback(
    stream_client,
    monkeypatch,
):
    def _audio_then_error(_text):
        yield b"\x01\x02"
        yield b""
        raise RuntimeError("post-audio failure")

    _register_provider(monkeypatch, "stream-b", stream_impl=_audio_then_error)
    config = {
        "provider": "sync-a",
        "sync-a": {"max_text_length": 50},
        "stream-b": {"max_text_length": 17},
        "streaming": {"provider": "stream-b"},
    }
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "attempted input", "done": True}))
        assert conn.receive_bytes() == b"\x01\x02"
        assert conn.receive_json() == {"type": "end"}


def test_empty_input_ends_without_fallback(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x01\x02"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"done": True}))
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == []


def test_stop_closes_without_stale_terminal_frame(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x01\x02"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"stop": True}))
        with pytest.raises(WebSocketDisconnect):
            conn.receive_json()

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
    text = "Short. " + "x" * 23

    pieces = web_server._split_text_for_speak_stream(text, 10)

    assert pieces == ["Short. ", "x" * 10, "x" * 10, "x" * 3]
    assert all(piece and len(piece) <= 10 for piece in pieces)
    assert "".join(pieces) == text


