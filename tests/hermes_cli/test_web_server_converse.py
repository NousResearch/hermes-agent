"""/api/audio/converse — off-device realtime voice loop over WebSocket.

Hermetic: no network, no real models, no audio devices. STT, the agent turn and
the streaming TTS provider are all monkeypatched.
"""

from __future__ import annotations

import json
import wave
from urllib.parse import urlencode

import numpy as np
import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def converse_client(monkeypatch, _isolate_hermes_home):
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
    return f"/api/audio/converse?{urlencode({'token': token or web_server._SESSION_TOKEN})}"


def test_create_voice_session_tags_source_voice(monkeypatch):
    """The dashboard voice session is minted with source="voice" so its durable row
    (and every prompt.submit turn that re-binds HERMES_SESSION_SOURCE from it) persists
    as a chat sub-kind labeled "Voice", not under Automations."""
    from hermes_cli.web_routers import _converse_loop

    captured: dict = {}

    def _fake_dispatch(req, transport):
        captured["params"] = req.get("params")
        return {"jsonrpc": "2.0", "id": req.get("id"),
                "result": {"session_id": "voice-sid"}}

    monkeypatch.setattr("tui_gateway.server.dispatch", _fake_dispatch)

    assert _converse_loop.create_voice_session() == "voice-sid"
    assert captured["params"]["source"] == "voice"


class _FakeStreamer:
    sample_rate = 24000
    channels = 1

    def __init__(self, chunks):
        self.chunks = chunks
        self.requests: list[str] = []

    def stream(self, text):
        self.requests.append(text)
        yield from self.chunks


def _patch_converse(monkeypatch, streamer, *, transcript="hello there", deltas=("Hi ", "back.")):
    """Wire fake STT / turn / streaming-TTS so the loop runs end to end offline."""
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 4000)

    import hermes_cli.web_routers._converse_loop as cl

    # STT: any captured WAV transcribes to the fixed transcript once, then "" so
    # the loop doesn't spin firing turns on repeated silence blocks.
    seen = {"n": 0}

    def _fake_transcribe(wav_path, model=None):
        seen["n"] += 1
        return {"success": True, "transcript": transcript if seen["n"] == 1 else ""}

    monkeypatch.setattr("tools.voice_mode.transcribe_recording", _fake_transcribe)

    # Session + turn: skip the real tui_gateway engine — just emit two deltas.
    monkeypatch.setattr(cl, "create_voice_session", lambda model=None: "voice-sid")

    def _fake_run_turn(session_id, text, on_delta, *, interrupted=False, timeout=300.0):
        for d in deltas:
            on_delta(d)
        return None

    monkeypatch.setattr(cl, "run_voice_turn", _fake_run_turn)
    return seen


def _speech_then_silence_pcm(block=480, speech_blocks=16, silence_blocks=60):
    """A canned PCM16 utterance: loud speech blocks then silence to endpoint."""
    speech = (np.ones(block, dtype=np.int16) * 9000)
    silence = np.zeros(block, dtype=np.int16)
    quiet_calib = np.zeros(block, dtype=np.int16)
    frames = [quiet_calib.tobytes() for _ in range(20)]  # calibrate a quiet floor
    frames += [speech.tobytes() for _ in range(speech_blocks)]
    frames += [silence.tobytes() for _ in range(silence_blocks)]
    return frames


def test_unauthorized_connect_closes_4401(converse_client, monkeypatch):
    web_server.app.state.auth_required = True
    with pytest.raises(Exception):
        with converse_client.websocket_connect("/api/audio/converse?token=wrong") as conn:
            conn.receive_json()


def _write_tone_wav(path, *, rate=24000, ms=40, freq=440.0):
    """Write a tiny mono s16 WAV (a short tone) — stands in for a one-shot TTS file."""
    n = int(rate * ms / 1000)
    t = np.arange(n, dtype=np.float64) / rate
    samples = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def test_no_streaming_provider_uses_one_shot_fallback(converse_client, monkeypatch, tmp_path):
    # No chunked API (e.g. edge): the converse loop must fall back to one-shot
    # synthesis + server-side transcode, NOT error out. The client still gets a
    # `ready` frame, PCM bytes, and turn_done.
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: None)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 4000)

    wav = tmp_path / "reply.wav"
    _write_tone_wav(wav)
    # A fresh copy per call so the fallback's unlink can't starve a later sentence.
    call = {"n": 0}

    def _fake_tts(text, *a, **k):
        call["n"] += 1
        dst = tmp_path / f"reply-{call['n']}.wav"
        dst.write_bytes(wav.read_bytes())
        return json.dumps({"success": True, "file_path": str(dst)})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _fake_tts)

    import hermes_cli.web_routers._converse_loop as cl

    seen = {"n": 0}

    def _fake_transcribe(wav_path, model=None):
        seen["n"] += 1
        return {"success": True, "transcript": "hello there" if seen["n"] == 1 else ""}

    monkeypatch.setattr("tools.voice_mode.transcribe_recording", _fake_transcribe)
    monkeypatch.setattr(cl, "create_voice_session", lambda model=None: "voice-sid")

    def _fake_run_turn(session_id, text, on_delta, *, interrupted=False, timeout=300.0):
        on_delta("Sure thing.")
        return None

    monkeypatch.setattr(cl, "run_voice_turn", _fake_run_turn)

    with converse_client.websocket_connect(_url()) as conn:
        conn.send_json({"type": "start"})  # mandatory first frame (config; dashboard auth is the token)
        ready = conn.receive_json()
        assert ready["type"] == "ready"
        assert ready["output"] == {"sample_rate": 24000, "format": "pcm16"}

        for frame in _speech_then_silence_pcm():
            conn.send_bytes(frame)

        pcm = []
        while True:
            msg = conn.receive()
            if msg.get("bytes") is not None:
                pcm.append(msg["bytes"])
                continue
            if json.loads(msg["text"])["type"] == "turn_done":
                break
        conn.send_text(json.dumps({"stop": True}))

    assert pcm  # the fallback produced transcoded PCM
    assert b"".join(pcm)  # non-empty audio


def test_full_turn_transcript_and_pcm(converse_client, monkeypatch):
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_converse(monkeypatch, streamer, transcript="turn it on")

    with converse_client.websocket_connect(_url()) as conn:
        conn.send_json({"type": "start"})
        ready = conn.receive_json()
        assert ready["type"] == "ready"
        assert ready["input"] == {"sample_rate": 16000, "format": "pcm16", "block_ms": 30}
        assert ready["output"] == {"sample_rate": 24000, "format": "pcm16"}

        for frame in _speech_then_silence_pcm():
            conn.send_bytes(frame)

        # Control + audio frames, in order: transcript -> speaking -> PCM -> turn_done.
        got_transcript = None
        pcm = []
        while True:
            msg = conn.receive()
            if msg.get("bytes") is not None:
                pcm.append(msg["bytes"])
                continue
            payload = json.loads(msg["text"])
            if payload["type"] == "transcript":
                got_transcript = payload["text"]
            elif payload["type"] == "turn_done":
                break

        assert got_transcript == "turn it on"
        assert pcm == [b"\x01\x02\x03\x04", b"\x05\x06"]
        assert streamer.requests  # the reply text reached the TTS provider

        conn.send_text(json.dumps({"stop": True}))


def test_reply_is_synthesized_per_sentence(converse_client, monkeypatch):
    # A two-sentence reply must reach the TTS provider as TWO separate stream()
    # calls (one per sentence, spoken as it lands) — not one concatenated blob.
    # This is what makes playback incremental rather than post-turn.
    streamer = _FakeStreamer([b"\x01\x02"])
    _patch_converse(
        monkeypatch, streamer, transcript="tell me two things",
        deltas=("The first light is on. ", "The second light is off."),
    )

    with converse_client.websocket_connect(_url()) as conn:
        conn.send_json({"type": "start"})
        assert conn.receive_json()["type"] == "ready"
        for frame in _speech_then_silence_pcm():
            conn.send_bytes(frame)
        while True:
            msg = conn.receive()
            if msg.get("bytes") is not None:
                continue
            if json.loads(msg["text"])["type"] == "turn_done":
                break
        conn.send_text(json.dumps({"stop": True}))

    assert streamer.requests == ["The first light is on.", "The second light is off."]
