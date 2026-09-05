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


def _start_msg(key=None, **cfg):
    """A JSON ``start`` frame — the client's mandatory first frame after connect.

    ``key`` is the start.key auth field (omit on the subprotocol-key path, where it's
    ignored); ``cfg`` carries optional session config (input_rate/output_rate/name/…)."""
    frame = {"type": "start", **cfg}
    if key is not None:
        frame["key"] = key
    return json.dumps(frame)


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
            await ws.send_str(_start_msg())  # subprotocol path: start.key not needed
            await _drive_full_turn(ws, streamer)
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()


@pytest.mark.asyncio
async def test_turn_persists_voice_session_source(monkeypatch):
    # The converse handler must PRE-CREATE the session row as source="voice" (a chat
    # sub-kind the dashboard files under Chats/"Voice", not "Automations"). The agent's
    # later create_session upsert preserves an existing row's source, and the agent
    # PLATFORM stays "api_server" (HA tools). Spy on the pre-create's create_session.
    adapter = _adapter()
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_converse(monkeypatch, streamer, transcript="turn it on")
    _patch_run_agent(adapter, monkeypatch)

    created: dict = {}

    class _SpyDB:
        # No **kwargs on purpose: mirrors the real create_session(session_id, source)
        # call so a regression that passes an unaccepted kwarg (which the real
        # SessionDB rejects with TypeError, then contextlib.suppress swallows) leaves
        # `created` empty and fails this test instead of silently persisting api_server.
        def create_session(self, session_id, source):
            created["id"], created["source"] = session_id, source
            return session_id

    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: _SpyDB())

    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()))
        try:
            await ws.send_str(_start_msg())
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

    assert created.get("source") == "voice"
    assert created.get("id", "").startswith("voice_")


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
            await ws.send_str(_start_msg())
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
async def test_start_key_accept_full_turn(monkeypatch):
    # No subprotocol (an ESP32-style device): start.key authenticates in the single start frame.
    adapter = _adapter()
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_converse(monkeypatch, streamer, transcript="turn it on")
    _patch_run_agent(adapter, monkeypatch)

    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect("/v1/audio/converse")
        try:
            await ws.send_str(_start_msg(key=API_KEY))
            await _drive_full_turn(ws, streamer)
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()


@pytest.mark.asyncio
async def test_start_bad_key_closes_unauthorized(monkeypatch):
    # A start frame with a bad/missing key and no subprotocol -> unauthorized + close 4401.
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect("/v1/audio/converse")
        try:
            await ws.send_str(_start_msg(key="nope"))
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            assert json.loads(msg.data) == {"type": "error", "error": "unauthorized"}
            closing = await ws.receive()
            assert closing.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED)
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_non_start_first_frame_rejected_bad_start(monkeypatch):
    # A valid subprotocol key, but the first frame is not a start frame -> bad_start close 4400.
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()))
        try:
            await ws.send_str(json.dumps({"type": "auth", "key": API_KEY}))  # not a start frame
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            assert json.loads(msg.data) == {"type": "error", "error": "bad_start"}
            closing = await ws.receive()
            assert closing.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED)
        finally:
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
async def test_no_credential_start_frame_closes_unauthorized(monkeypatch):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        # No key subprotocol and a start frame with no key -> error + close 4401,
        # with no `ready` and no session started.
        ws = await client.ws_connect("/v1/audio/converse")
        try:
            await ws.send_str(_start_msg())  # start, but no key and no subprotocol
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
            await ws.send_str(_start_msg())
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
            # A ?token= grants nothing; a start frame with no key is rejected unauthorized.
            await ws.send_str(_start_msg())
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
            await ws.send_str(_start_msg())  # start, but no key: the cookie grants nothing
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            assert json.loads(msg.data) == {"type": "error", "error": "unauthorized"}
        finally:
            await ws.close()


async def _run_prompt_capture_turn(adapter, monkeypatch, *, start_cfg):
    """Drive one full turn, capturing the ephemeral_system_prompt + clean user_message.

    Connects on the subprotocol-key path, sends a ``start`` frame built from *start_cfg*,
    runs the canned utterance to turn_done, and returns the captured dict."""
    streamer = _FakeStreamer([b"\x01\x02"])
    _patch_converse(monkeypatch, streamer, transcript="what time is it")

    seen: dict = {}

    async def _fake_run_agent(user_message, conversation_history, *,
                              ephemeral_system_prompt=None, stream_delta_callback=None,
                              session_id=None, **_):
        seen["prompt"] = ephemeral_system_prompt
        seen["user_message"] = user_message
        if stream_delta_callback is not None:
            stream_delta_callback("It's noon.")
        return {"final_response": "It's noon."}, {"input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)

    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()))
        try:
            await ws.send_str(_start_msg(**start_cfg))
            assert (await ws.receive_json())["type"] == "ready"
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
    return seen


@pytest.mark.asyncio
async def test_turn_uses_voice_system_prompt(monkeypatch):
    # Spoken replies must stay short: converse runs the turn with the name-aware voice
    # ephemeral system prompt, while the user_message (transcript) is passed CLEAN.
    from tools.voice_converse_loop import voice_system_prompt

    adapter = _adapter()
    seen = await _run_prompt_capture_turn(adapter, monkeypatch, start_cfg={})
    # Default name is "Sakura" — the prompt carries the identity + brevity block.
    assert seen["prompt"] == voice_system_prompt("Sakura")
    assert "Sakura" in seen["prompt"]
    assert seen["user_message"] == "what time is it"  # transcript is clean; prompt is separate


@pytest.mark.asyncio
async def test_start_name_injected_into_voice_prompt(monkeypatch):
    # start.name flows into the turn's ephemeral_system_prompt.
    from tools.voice_converse_loop import voice_system_prompt

    adapter = _adapter()
    seen = await _run_prompt_capture_turn(
        adapter, monkeypatch, start_cfg={"name": "Sakura"})
    assert seen["prompt"] == voice_system_prompt("Sakura")
    assert "Sakura" in seen["prompt"]


@pytest.mark.asyncio
async def test_equal_input_output_rates_reported_and_full_turn(monkeypatch):
    # A single-clock device sets input_rate == output_rate: ready reports both, and a full
    # turn still transcribes + returns PCM. output_rate == synth.sample_rate -> no resampling.
    adapter = _adapter()
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    streamer.sample_rate = 16000  # so output_rate=16000 is a no-op resample (bytes unchanged)
    _patch_converse(monkeypatch, streamer, transcript="turn it on")
    _patch_run_agent(adapter, monkeypatch)

    async with TestClient(TestServer(_app(adapter))) as client:
        ws = await client.ws_connect(
            "/v1/audio/converse", protocols=(VOICE_PROTOCOL, _key_protocol()))
        try:
            await ws.send_str(_start_msg(input_rate=16000, output_rate=16000))
            ready = await ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["input"] == {"sample_rate": 16000, "format": "pcm16", "block_ms": 30}
            assert ready["output"] == {"sample_rate": 16000, "format": "pcm16"}

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
            assert pcm == [b"\x01\x02\x03\x04", b"\x05\x06"]  # no-op resample: bytes unchanged
        finally:
            await ws.send_str(json.dumps({"stop": True}))
            await ws.close()
