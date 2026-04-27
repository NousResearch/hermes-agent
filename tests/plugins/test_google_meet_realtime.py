"""Tests for plugins.google_meet.realtime.openai_client (v2).

Uses a scripted fake WebSocket — no network, no API key required.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


# ---------------------------------------------------------------------------
# Fake WebSocket
# ---------------------------------------------------------------------------


class _FakeWS:
    """Scripted WS: send() records frames, recv() pops a queue."""

    def __init__(self, recv_frames: list):
        self.sent: list[dict] = []
        self._recv_q: list = list(recv_frames)
        self.closed = False

    def send(self, payload):
        # Always accept str payloads — client encodes JSON with json.dumps.
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode()
        self.sent.append(json.loads(payload))

    def recv(self, timeout=None):  # noqa: ARG002
        if not self._recv_q:
            raise RuntimeError("fake ws: no more frames")
        frame = self._recv_q.pop(0)
        if isinstance(frame, dict):
            return json.dumps(frame)
        return frame

    def close(self):
        self.closed = True


def _install_fake_websockets(monkeypatch, fake_ws):
    """Install a fake ``websockets.sync.client`` module in sys.modules."""
    mod_websockets = types.ModuleType("websockets")
    mod_sync = types.ModuleType("websockets.sync")
    mod_sync_client = types.ModuleType("websockets.sync.client")

    captured = {"url": None, "headers": None, "kwargs": None}

    def _connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        captured["headers"] = (
            kwargs.get("additional_headers") or kwargs.get("extra_headers")
        )
        return fake_ws

    mod_sync_client.connect = _connect
    mod_sync.client = mod_sync_client
    mod_websockets.sync = mod_sync

    monkeypatch.setitem(sys.modules, "websockets", mod_websockets)
    monkeypatch.setitem(sys.modules, "websockets.sync", mod_sync)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", mod_sync_client)
    return captured


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


def test_connect_sends_session_update_with_voice_and_instructions(monkeypatch):
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    ws = _FakeWS(recv_frames=[])
    captured = _install_fake_websockets(monkeypatch, ws)

    sess = RealtimeSession(
        api_key="sk-test",
        model="gpt-realtime",
        voice="verse",
        instructions="Be brief.",
    )
    sess.connect()

    # Auth + beta headers set.
    assert captured["url"].startswith("wss://api.openai.com/v1/realtime")
    assert "model=gpt-realtime" in captured["url"]
    headers = captured["headers"] or []
    hdict = dict(headers)
    assert hdict.get("Authorization") == "Bearer sk-test"
    assert hdict.get("OpenAI-Beta") == "realtime=v1"

    # First frame sent must be session.update with the right shape.
    assert len(ws.sent) == 1
    update = ws.sent[0]
    assert update["type"] == "session.update"
    s = update["session"]
    assert s["voice"] == "verse"
    assert s["instructions"] == "Be brief."
    assert set(s["modalities"]) == {"audio", "text"}
    assert s["output_audio_format"] == "pcm16"
    assert s["input_audio_format"] == "pcm16"


# ---------------------------------------------------------------------------
# speak()
# ---------------------------------------------------------------------------


def test_speak_sends_create_and_response_and_writes_audio(monkeypatch, tmp_path):
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    audio_bytes = b"\x01\x02\x03\x04PCM!"
    b64 = base64.b64encode(audio_bytes).decode()

    recv_frames = [
        {"type": "response.created"},
        {"type": "response.audio.delta", "delta": b64},
        {"type": "response.audio.delta", "delta": base64.b64encode(b"more").decode()},
        {"type": "response.done"},
    ]
    ws = _FakeWS(recv_frames=recv_frames)
    _install_fake_websockets(monkeypatch, ws)

    sink = tmp_path / "out.pcm"
    sess = RealtimeSession(api_key="sk-test", audio_sink_path=sink)
    sess.connect()
    result = sess.speak("Hello everyone.")

    # Frames sent after session.update: conversation.item.create then response.create.
    types_sent = [f["type"] for f in ws.sent]
    assert types_sent == ["session.update", "conversation.item.create", "response.create"]

    item = ws.sent[1]["item"]
    assert item["role"] == "user"
    assert item["content"][0]["type"] == "input_text"
    assert item["content"][0]["text"] == "Hello everyone."

    resp = ws.sent[2]["response"]
    assert resp["modalities"] == ["audio"]

    # Audio file got decoded + appended bytes.
    data = sink.read_bytes()
    assert data == audio_bytes + b"more"
    assert result["ok"] is True
    assert result["bytes_written"] == len(audio_bytes) + len(b"more")
    assert result["duration_ms"] >= 0.0


def test_speak_raises_on_error_frame(monkeypatch, tmp_path):
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    ws = _FakeWS(recv_frames=[
        {"type": "response.created"},
        {"type": "error", "error": {"message": "bad juju"}},
    ])
    _install_fake_websockets(monkeypatch, ws)

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=tmp_path / "o.pcm")
    sess.connect()
    with pytest.raises(RuntimeError, match="bad juju"):
        sess.speak("hi")


def test_speak_without_connect_raises(monkeypatch):
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test")
    with pytest.raises(RuntimeError, match="connect"):
        sess.speak("hi")


def test_close_is_idempotent_and_closes_ws(monkeypatch):
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    ws = _FakeWS(recv_frames=[])
    _install_fake_websockets(monkeypatch, ws)

    sess = RealtimeSession(api_key="sk-test")
    sess.connect()
    sess.close()
    assert ws.closed is True
    # Second close is a no-op.
    sess.close()


# ---------------------------------------------------------------------------
# websockets dependency missing
# ---------------------------------------------------------------------------


def test_connect_raises_clean_error_when_websockets_missing(monkeypatch):
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    # Make `import websockets.sync.client` fail.
    monkeypatch.setitem(sys.modules, "websockets", None)
    monkeypatch.setitem(sys.modules, "websockets.sync", None)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", None)

    sess = RealtimeSession(api_key="sk-test")
    with pytest.raises(RuntimeError, match="pip install websockets"):
        sess.connect()


# ---------------------------------------------------------------------------
# RealtimeSpeaker
# ---------------------------------------------------------------------------


class _StubSession:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, timeout=30.0):  # noqa: ARG002
        self.spoken.append(text)
        return {"ok": True, "bytes_written": len(text), "duration_ms": 1.0}


def test_speaker_run_until_stopped_processes_queue(tmp_path):
    from plugins.google_meet.realtime.openai_client import RealtimeSpeaker

    queue = tmp_path / "queue.jsonl"
    processed = tmp_path / "processed.jsonl"
    queue.write_text(
        json.dumps({"id": "a", "text": "hello one"}) + "\n"
        + json.dumps({"id": "b", "text": "hello two"}) + "\n"
    )

    stub = _StubSession()
    speaker = RealtimeSpeaker(stub, queue_path=queue, processed_path=processed)

    # Stop once the queue is empty.
    def _stop():
        return queue.exists() and queue.read_text().strip() == ""

    speaker.run_until_stopped(_stop, poll_interval=0.01)

    assert stub.spoken == ["hello one", "hello two"]

    # Processed file has both entries, in order.
    lines = [json.loads(l) for l in processed.read_text().splitlines() if l.strip()]
    assert [l["id"] for l in lines] == ["a", "b"]
    assert all(l["result"]["ok"] for l in lines)

    # Queue is empty (possibly empty string) after processing.
    assert queue.read_text().strip() == ""


def test_speaker_exits_immediately_when_stop_fn_true(tmp_path):
    from plugins.google_meet.realtime.openai_client import RealtimeSpeaker

    queue = tmp_path / "q.jsonl"
    queue.write_text(json.dumps({"id": "x", "text": "never spoken"}) + "\n")

    stub = _StubSession()
    speaker = RealtimeSpeaker(stub, queue_path=queue)
    speaker.run_until_stopped(lambda: True, poll_interval=0.01)
    assert stub.spoken == []


def test_speaker_drops_line_without_processed_path_when_none(tmp_path):
    from plugins.google_meet.realtime.openai_client import RealtimeSpeaker

    queue = tmp_path / "q.jsonl"
    queue.write_text(json.dumps({"id": "only", "text": "once"}) + "\n")

    stub = _StubSession()
    speaker = RealtimeSpeaker(stub, queue_path=queue, processed_path=None)

    def _stop():
        return queue.read_text().strip() == ""

    speaker.run_until_stopped(_stop, poll_interval=0.01)
    assert stub.spoken == ["once"]
    assert queue.read_text().strip() == ""
