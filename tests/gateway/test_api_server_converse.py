"""GET /v1/audio/converse — API-key-authed realtime voice over WebSocket (aiohttp).

Hermetic: no network, no real models, no audio devices. STT
(``transcribe_recording``), the agent turn (``_run_agent``) and the streaming TTS
provider (``resolve_streaming_provider``) are all monkeypatched, so the whole
loop runs end to end offline over a real aiohttp TestServer.

Auth uses the profile's ``API_SERVER_KEY`` presented one of two ways (never the
``Authorization`` header, never a ``?token=`` param):
  (A) a ``hermes-key.<KEY>`` subprotocol (validated pre-upgrade), or
  (B) a first ``{"type":"auth","key":...}`` frame when no key subprotocol is sent.

Asserts: (1) subprotocol-key accept runs a full turn (transcript + PCM +
turn_done); (2) first-message auth accept runs a full turn; (3) neither provided
→ the socket is closed unauthorized (a bad subprotocol key rejects the upgrade
with 401; a bad/absent first-message auth closes 4401).
"""

import json
import wave

import numpy as np
import pytest
from aiohttp import WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


API_KEY = "-".join(("fixture", "converse", "api", "key", "0123456789"))
VOICE_PROTOCOL = "hermes-voice-v1"


def _key_protocol(key=API_KEY):
    return f"hermes-key.{key}"


class _FakeStreamer:
    sample_rate = 24000
    channels = 1

    def __init__(self, chunks):
        self.chunks = chunks
        self.requests: list[str] = []

    def stream(self, text):
        self.requests.append(text)
        yield from self.chunks


def _adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))


def _app(adapter):
    app = web.Application()
    app.router.add_get("/v1/audio/converse", adapter._handle_converse_ws)
    return app


def _speech_then_silence_pcm(block=480, speech_blocks=16, silence_blocks=60):
    """A canned PCM16 utterance: quiet-floor calibration, loud speech, then silence."""
    speech = (np.ones(block, dtype=np.int16) * 9000)
    silence = np.zeros(block, dtype=np.int16)
    frames = [silence.tobytes() for _ in range(20)]  # calibrate a quiet floor
    frames += [speech.tobytes() for _ in range(speech_blocks)]
    frames += [silence.tobytes() for _ in range(silence_blocks)]
    return frames


def _patch_converse(monkeypatch, streamer, *, transcript="hello there"):
    """Wire fake STT / streaming-TTS so the loop runs end to end offline."""
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 4000)

    # STT: any captured WAV transcribes to the fixed transcript once, then "" so the
    # loop does not spin firing turns on repeated silence blocks.
    seen = {"n": 0}

    def _fake_transcribe(wav_path, model=None):
        seen["n"] += 1
        return {"success": True, "transcript": transcript if seen["n"] == 1 else ""}

    monkeypatch.setattr("tools.voice_mode.transcribe_recording", _fake_transcribe)
    return seen


def _patch_run_agent(adapter, monkeypatch, deltas=("Turn it ", "on.")):
    """Replace the real agent turn with one that emits *deltas* then returns."""
    async def _fake_run_agent(user_message, conversation_history, *,
                              stream_delta_callback=None, session_id=None, **_):
        if stream_delta_callback is not None:
            for d in deltas:
                stream_delta_callback(d)
        return {"final_response": "".join(deltas)}, {"input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)


async def _drive_full_turn(ws, streamer):
    """Feed a canned utterance and assert transcript + PCM + turn_done come back."""
    ready = await ws.receive_json()
    assert ready["type"] == "ready"
    assert ready["input"] == {"sample_rate": 16000, "format": "pcm16", "block_ms": 30}
    assert ready["output"] == {"sample_rate": 24000, "format": "pcm16"}

    for frame in _speech_then_silence_pcm():
        await ws.send_bytes(frame)

    got_transcript = None
    pcm: list[bytes] = []
    while True:
        msg = await ws.receive()
        if msg.type == web.WSMsgType.BINARY:
            pcm.append(msg.data)
            continue
        if msg.type == web.WSMsgType.TEXT:
            payload = json.loads(msg.data)
            if payload["type"] == "transcript":
                got_transcript = payload["text"]
            elif payload["type"] == "turn_done":
                break
            continue
        if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
            break

    assert got_transcript == "turn it on"
    assert pcm == [b"\x01\x02\x03\x04", b"\x05\x06"]
    assert streamer.requests  # the reply text reached the TTS provider


@pytest.mark.asyncio
async def test_subprotocol_key_accept_full_turn(monkeypatch):
    adapter = _adapter()
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_converse(monkeypatch, streamer, transcript="turn it on")
    _patch_run_agent(adapter, monkeypatch)

    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()))
        try:
            # The accepted socket selects ONLY the base protocol — the key is not echoed.
            assert ws.protocol == VOICE_PROTOCOL
            await _drive_full_turn(ws, streamer)
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()


@pytest.mark.asyncio
async def test_turn_runs_under_voice_session_source(monkeypatch):
    # The converse turn must persist as source="voice" (a chat sub-kind), so the
    # agent runs inside session_source_scope("voice"). Record what
    # _session_source_for_agent("api_server") resolves to when _run_agent fires:
    # inside the scope it is "voice"; the platform ("api_server") is unchanged.
    from run_agent import _session_source_for_agent

    adapter = _adapter()
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_converse(monkeypatch, streamer, transcript="turn it on")

    seen: dict = {}

    async def _fake_run_agent(user_message, conversation_history, *,
                              stream_delta_callback=None, session_id=None, **_):
        seen["source"] = _session_source_for_agent("api_server")
        if stream_delta_callback is not None:
            stream_delta_callback("ok.")
        return {"final_response": "ok."}, {"input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)

    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()))
        try:
            ready = await ws.receive_json()
            assert ready["type"] == "ready"
            for frame in _speech_then_silence_pcm():
                await ws.send_bytes(frame)
            while True:
                msg = await ws.receive()
                if msg.type == web.WSMsgType.TEXT:
                    if json.loads(msg.data)["type"] == "turn_done":
                        break
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED,
                                  web.WSMsgType.ERROR):
                    break
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()

    assert seen.get("source") == "voice"


def _write_tone_wav(path, *, rate=24000, ms=40, freq=440.0):
    """Write a tiny mono s16 WAV — stands in for a one-shot TTS output file."""
    n = int(rate * ms / 1000)
    t = np.arange(n, dtype=np.float64) / rate
    samples = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


@pytest.mark.asyncio
async def test_no_streaming_provider_uses_one_shot_fallback(monkeypatch, tmp_path):
    # No chunked API (edge): the converse loop falls back to one-shot synthesis +
    # server-side transcode instead of erroring. The client still gets ready + PCM.
    adapter = _adapter()
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: None)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 4000)

    src = tmp_path / "reply.wav"
    _write_tone_wav(src)
    call = {"n": 0}

    def _fake_tts(text, *a, **k):
        call["n"] += 1
        dst = tmp_path / f"reply-{call['n']}.wav"
        dst.write_bytes(src.read_bytes())
        return json.dumps({"success": True, "file_path": str(dst)})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _fake_tts)

    seen = {"n": 0}

    def _fake_transcribe(wav_path, model=None):
        seen["n"] += 1
        return {"success": True, "transcript": "hello there" if seen["n"] == 1 else ""}

    monkeypatch.setattr("tools.voice_mode.transcribe_recording", _fake_transcribe)
    _patch_run_agent(adapter, monkeypatch, deltas=("Sure ", "thing.",))

    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()))
        try:
            ready = await ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["output"] == {"sample_rate": 24000, "format": "pcm16"}

            for frame in _speech_then_silence_pcm():
                await ws.send_bytes(frame)

            pcm: list[bytes] = []
            while True:
                msg = await ws.receive()
                if msg.type == web.WSMsgType.BINARY:
                    pcm.append(msg.data)
                    continue
                if msg.type == web.WSMsgType.TEXT:
                    if json.loads(msg.data)["type"] == "turn_done":
                        break
                    continue
                if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
            assert pcm and b"".join(pcm)  # fallback produced transcoded PCM
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()


@pytest.mark.asyncio
async def test_first_message_auth_accept_full_turn(monkeypatch):
    adapter = _adapter()
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_converse(monkeypatch, streamer, transcript="turn it on")
    _patch_run_agent(adapter, monkeypatch)

    async with TestClient(TestServer(_app(adapter))) as client:
        # No key subprotocol -> first-message auth path.
        ws = await client.ws_connect("/v1/audio/converse")
        try:
            await ws.send_str(json.dumps({"type": "auth", "key": API_KEY}))
            await _drive_full_turn(ws, streamer)
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()


@pytest.mark.asyncio
async def test_bad_subprotocol_key_rejects_upgrade_401(monkeypatch):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        with pytest.raises(WSServerHandshakeError) as exc:
            await client.ws_connect(
                "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol("nope")))
        assert exc.value.status == 401


@pytest.mark.asyncio
async def test_no_credential_first_frame_closes_unauthorized(monkeypatch):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        # No key subprotocol and a non-auth first frame -> error + close 4401,
        # with no `ready` and no session started.
        ws = await client.ws_connect("/v1/audio/converse")
        try:
            await ws.send_str(json.dumps({"stop": True}))
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            assert json.loads(msg.data) == {"type": "error", "error": "unauthorized"}
            closing = await ws.receive()
            assert closing.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED)
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_origin_guard_exempts_key_bearing_ws_only(monkeypatch):
    """A browser-context client sends an unsuppressible Origin (Electron → ``null``).
    The cors_middleware origin guard must let a key-bearing converse upgrade through
    (the key is explicit, non-ambient auth) while STILL blocking a bare Origin with
    no key subprotocol."""
    from gateway.platforms.api_server import cors_middleware

    adapter = _adapter()  # no cors_origins configured → every browser Origin is "disallowed"
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_converse(monkeypatch, streamer, transcript="turn it on")
    _patch_run_agent(adapter, monkeypatch)

    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/audio/converse", adapter._handle_converse_ws)

    async with TestClient(TestServer(app)) as client:
        # (1) Electron-style: Origin: null + key subprotocol → past the guard, full turn.
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()),
            headers={"Origin": "null"})
        try:
            assert ws.protocol == VOICE_PROTOCOL
            await _drive_full_turn(ws, streamer)
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()

        # (2) Same Origin but NO key subprotocol → origin guard still 403s the upgrade.
        with pytest.raises(WSServerHandshakeError) as exc:
            await client.ws_connect("/v1/audio/converse", headers={"Origin": "null"})
        assert exc.value.status == 403


@pytest.mark.asyncio
async def test_query_token_is_not_a_credential(monkeypatch):
    """A ?token= param is NOT auth: the socket opens (no key subprotocol) but stays
    on the first-message-auth path and closes 4401 without it."""
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(f"/v1/audio/converse?token={API_KEY}")
        try:
            # A ?token= grants nothing; the first non-auth frame is rejected.
            await ws.send_str(json.dumps({"commit": True}))
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            assert json.loads(msg.data) == {"type": "error", "error": "unauthorized"}
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_cookie_is_not_a_credential(monkeypatch):
    """Locks the invariant the Origin exemption depends on: this route has NO ambient
    (cookie/session) auth. A cookie must never authenticate — without a key the socket
    is still rejected. If cookie/session auth is ever added to this handler this test
    breaks loudly; otherwise the Origin exemption would silently become a CSWSH hole."""
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        # A session-looking cookie + no key subprotocol -> still the first-message-auth
        # path; a non-auth first frame is rejected. The cookie grants nothing.
        ws = await client.ws_connect(
            "/v1/audio/converse", headers={"Cookie": "session=pretend-valid"})
        try:
            await ws.send_str(json.dumps({"commit": True}))
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            assert json.loads(msg.data) == {"type": "error", "error": "unauthorized"}
        finally:
            await ws.close()
