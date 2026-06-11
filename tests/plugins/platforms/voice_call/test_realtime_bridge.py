"""Realtime bridge: tokens, pacer cadence, barge-in, consult dispatch.

Uses a scripted fake carrier WebSocket and a scripted fake realtime
session — no network, no realtime API keys.
"""

import asyncio
import json
from array import array

import pytest

pytest.importorskip("aiohttp")
from aiohttp import WSMsgType

from plugins.platforms.voice_call import audio
from plugins.platforms.voice_call import runtime as runtime_mod
from plugins.platforms.voice_call.events import CallRecord
from plugins.platforms.voice_call.realtime import bridge as bridge_mod
from plugins.platforms.voice_call.realtime.base import (
    RealtimeEvent,
    RealtimeVoiceSession,
)
from plugins.platforms.voice_call.realtime.bridge import (
    AudioPacer,
    RealtimeBridgeManager,
    RealtimeCallBridge,
)
from plugins.platforms.voice_call.realtime.frames import TelnyxStreamFrameAdapter


class FakeWsMessage:
    def __init__(self, data):
        self.type = WSMsgType.TEXT
        self.data = data


class FakeCarrierWs:
    """Iterates scripted carrier frames; collects what the bridge sends."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []
        self.closed = False
        self._consumed = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._messages:
            return FakeWsMessage(self._messages.pop(0))
        self._consumed.set()
        # Keep the connection "open" until the bridge tears down.
        await asyncio.sleep(3600)
        raise StopAsyncIteration

    async def send_str(self, data):
        self.sent.append(json.loads(data))

    async def close(self):
        self.closed = True


class FakeSession(RealtimeVoiceSession):
    name = "fake"
    input_sample_rate = 24000
    output_sample_rate = 24000

    def __init__(self, scripted_events=None):
        self.received_audio = []
        self.injected_text = []
        self.tool_results = []
        self.cancelled = 0
        self.closed = False
        self._events = asyncio.Queue()
        for event in scripted_events or []:
            self._events.put_nowait(event)

    async def connect(self):
        pass

    async def send_audio(self, pcm16):
        self.received_audio.append(pcm16)

    async def events(self):
        while True:
            event = await self._events.get()
            yield event
            if event.type == "closed":
                return

    def push(self, event):
        self._events.put_nowait(event)

    async def send_tool_result(self, tool_call_id, result):
        self.tool_results.append((tool_call_id, result))

    async def inject_text(self, text):
        self.injected_text.append(text)

    async def cancel_response(self):
        self.cancelled += 1

    async def close(self):
        self.closed = True


def _record(direction="outbound", **metadata):
    record = CallRecord(
        call_id="vc-rt", provider="telnyx", direction=direction,
        from_number="+15555550000", to_number="+15555550001",
        provider_call_id="v3:rt", mode="conversation",
    )
    record.metadata.update(metadata)
    return record


def _media_frame(payload: bytes) -> str:
    import base64

    return json.dumps({
        "event": "media", "stream_id": "s1",
        "media": {"payload": base64.b64encode(payload).decode(),
                  "track": "inbound_track"},
    })


class _FakeRuntime:
    """Just enough of VoiceCallRuntime for the bridge."""

    def __init__(self, make_config, manager=None):
        self.config = make_config()
        self.config.realtime.enabled = True
        self.public_url = "https://hooks.example"
        self.provider = None
        self.manager = manager


@pytest.fixture
def fake_runtime(make_config):
    return _FakeRuntime(make_config)


def _bridge(runtime, record, session):
    return RealtimeCallBridge(
        runtime=runtime, record=record,
        frame_adapter=TelnyxStreamFrameAdapter(), session=session,
    )


# -- token lifecycle -----------------------------------------------------------


def test_tokens_are_one_shot_and_expire(fake_runtime, monkeypatch):
    manager = RealtimeBridgeManager(fake_runtime)
    token = manager.mint_token("vc-1")
    assert manager.consume_token(token) == "vc-1"
    assert manager.consume_token(token) is None  # one-shot
    assert manager.consume_token("never-minted") is None

    stale = manager.mint_token("vc-2")
    manager._tokens[stale] = ("vc-2", 0.0)  # minted long ago
    assert manager.consume_token(stale) is None


def test_prepare_call_attaches_stream_metadata(fake_runtime):
    manager = RealtimeBridgeManager(fake_runtime)
    record = _record()
    manager.prepare_call(record)
    assert record.metadata["realtime"] is True
    url = record.metadata["stream_url"]
    token = record.metadata["stream_auth_token"]
    assert url.startswith("wss://hooks.example/voice/stream/")
    assert url.endswith(token)
    assert manager.consume_token(token) == record.call_id

    notify = _record()
    notify.mode = "notify"
    manager.prepare_call(notify)
    assert "stream_url" not in notify.metadata  # notify keeps carrier TTS


# -- pacer ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pacer_cadence_and_clear():
    sent_at = []
    loop = asyncio.get_running_loop()

    async def send(frame):
        sent_at.append(loop.time())

    pacer = AudioPacer(send)
    task = loop.create_task(pacer.run())
    pacer.push(audio.silence_frame() * 5)  # 5 frames = 100 ms of audio
    await asyncio.sleep(0.15)
    pacer.stop()
    task.cancel()

    assert len(sent_at) == 5
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert all(0.01 < gap < 0.05 for gap in gaps), gaps

    pacer.push(audio.silence_frame() * 10)
    assert pacer.clear() == 10 and pacer.pending == 0


# -- bridge dataflow ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_pumps_carrier_audio_into_session(fake_runtime):
    ulaw = bytes([0x55]) * 160
    ws = FakeCarrierWs([
        json.dumps({"event": "start", "stream_id": "s1",
                    "start": {"call_control_id": "v3:rt"}}),
        _media_frame(ulaw),
    ])
    session = FakeSession()
    bridge = _bridge(fake_runtime, _record(), session)

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.wait_for(ws._consumed.wait(), 2)
    await asyncio.sleep(0.05)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)

    assert len(session.received_audio) == 1
    pcm = session.received_audio[0]
    # 160 µ-law samples @8k → ~480 samples @24k → 960 bytes.
    assert abs(len(pcm) - 960) <= 8
    assert session.closed


@pytest.mark.asyncio
async def test_bridge_sends_model_audio_as_paced_media_frames(fake_runtime):
    ws = FakeCarrierWs([])
    # 40 ms of model audio @24k → 2 carrier frames @8k.
    pcm = array("h", [1000] * (24000 // 25)).tobytes()
    session = FakeSession()
    bridge = _bridge(fake_runtime, _record(), session)

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.sleep(0.02)
    session.push(RealtimeEvent(type="audio", audio=pcm))
    await asyncio.sleep(0.15)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)

    media = [m for m in ws.sent if m.get("event") == "media"]
    assert len(media) == 2
    assert all("payload" in m["media"] for m in media)


@pytest.mark.asyncio
async def test_bridge_barge_in_clears_carrier_queue(fake_runtime):
    ws = FakeCarrierWs([])
    session = FakeSession()
    session.response_active = True  # the bot is mid-utterance
    record = _record()
    bridge = _bridge(fake_runtime, record, session)
    bridge._greeting_until = 0.0

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.sleep(0.02)
    # Queue a lot of audio, then the caller starts talking.
    session.push(RealtimeEvent(
        type="audio", audio=array("h", [500] * 24000).tobytes()))
    await asyncio.sleep(0.03)
    session.push(RealtimeEvent(type="speech_started"))
    await asyncio.sleep(0.05)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)

    assert any(m.get("event") == "clear" for m in ws.sent)
    assert session.cancelled == 1


@pytest.mark.asyncio
async def test_bridge_ignores_caller_speech_when_bot_is_silent(fake_runtime):
    """Caller speaking while the bot is idle is a normal turn — clearing or
    cancelling here used to dump already-generated answers from the pacer
    and spam 'no active response found' errors."""
    ws = FakeCarrierWs([])
    session = FakeSession()  # response_active stays False, pacer empty
    bridge = _bridge(fake_runtime, _record(), session)
    bridge._greeting_until = 0.0

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.sleep(0.02)
    session.push(RealtimeEvent(type="speech_started"))
    await asyncio.sleep(0.05)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)

    assert not any(m.get("event") == "clear" for m in ws.sent)
    assert session.cancelled == 0


@pytest.mark.asyncio
async def test_bridge_greeting_window_suppresses_barge_in(fake_runtime):
    ws = FakeCarrierWs([])
    session = FakeSession()
    record = _record(direction="inbound")
    bridge = _bridge(fake_runtime, record, session)

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.sleep(0.02)
    assert session.injected_text == [fake_runtime.config.inbound_greeting]
    session.push(RealtimeEvent(type="speech_started"))  # within greeting window
    await asyncio.sleep(0.05)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)

    assert not any(m.get("event") == "clear" for m in ws.sent)
    assert session.cancelled == 0


@pytest.mark.asyncio
async def test_bridge_consult_tool_roundtrip(fake_runtime, monkeypatch):
    class FakeLlm:
        def complete(self, messages, **kwargs):
            class R:
                text = "Your next meeting is at three PM."
            assert messages[-1]["content"] == "what's on the calendar?"
            return R()

    monkeypatch.setattr(bridge_mod, "_plugin_llm_factory", lambda: FakeLlm())
    ws = FakeCarrierWs([])
    session = FakeSession()
    bridge = _bridge(fake_runtime, _record(), session)

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.sleep(0.02)
    session.push(RealtimeEvent(
        type="tool_call", tool_call_id="call-1", tool_name="agent_consult",
        tool_args={"question": "what's on the calendar?"},
    ))
    await asyncio.sleep(0.1)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)

    assert session.tool_results == [
        ("call-1", "Your next meeting is at three PM.")
    ]
    # The caller heard a filler while the consult ran.
    assert fake_runtime.config.responder.thinking_phrase in session.injected_text


@pytest.mark.asyncio
async def test_tool_result_deferred_while_response_active(fake_runtime, monkeypatch):
    """A consult result arriving mid-filler waits for response_done —
    response.create during an active response is rejected by the API."""

    class FakeLlm:
        def complete(self, messages, **kwargs):
            class R:
                text = "deferred answer"
            return R()

    monkeypatch.setattr(bridge_mod, "_plugin_llm_factory", lambda: FakeLlm())
    ws = FakeCarrierWs([])
    session = FakeSession()
    session.response_active = True  # filler/utterance still playing
    bridge = _bridge(fake_runtime, _record(), session)

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.sleep(0.02)
    session.push(RealtimeEvent(
        type="tool_call", tool_call_id="call-9", tool_name="agent_consult",
        tool_args={"question": "anything"},
    ))
    await asyncio.sleep(0.1)
    assert session.tool_results == []  # held back

    session.response_active = False
    session.push(RealtimeEvent(type="response_done"))
    await asyncio.sleep(0.05)
    assert session.tool_results == [("call-9", "deferred answer")]

    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)


@pytest.mark.asyncio
async def test_bridge_transcripts_mirror_into_call_record(
    fake_runtime, vc_config, provider, store
):
    from plugins.platforms.voice_call.manager import CallManager

    manager = CallManager(vc_config, provider, store)
    record = _record()
    manager.active[record.call_id] = record
    fake_runtime.manager = manager

    ws = FakeCarrierWs([])
    session = FakeSession()
    bridge = _bridge(fake_runtime, record, session)
    run = asyncio.create_task(bridge.run(ws))
    await asyncio.sleep(0.02)
    session.push(RealtimeEvent(type="transcript", role="user", text="hello"))
    session.push(RealtimeEvent(type="transcript", role="assistant", text="hi there"))
    await asyncio.sleep(0.05)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)

    assert [(t.speaker, t.text) for t in record.transcript] == [
        ("user", "hello"), ("bot", "hi there"),
    ]
    await manager.shutdown()


# -- stream upgrade endpoint --------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_upgrade_rejects_bad_and_reused_tokens(tmp_path, make_config):
    import aiohttp

    cfg = make_config()
    cfg.serve.port = 0
    cfg.realtime.enabled = True
    cfg.realtime.provider = "openai"
    runtime = await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    try:
        assert runtime.bridge_manager is not None
        port = runtime.webhook_server.bound_port
        async with aiohttp.ClientSession() as client:
            async with client.get(
                f"http://127.0.0.1:{port}/voice/stream/not-a-token"
            ) as resp:
                assert resp.status == 403

            # Minted token for a call that doesn't exist → 404, and the
            # token is consumed (second use → 403).
            token = runtime.bridge_manager.mint_token("vc-ghost")
            async with client.get(
                f"http://127.0.0.1:{port}/voice/stream/{token}"
            ) as resp:
                assert resp.status == 404
            async with client.get(
                f"http://127.0.0.1:{port}/voice/stream/{token}"
            ) as resp:
                assert resp.status == 403
    finally:
        await runtime_mod.stop_runtime()


def test_openai_session_update_uses_ga_shape(monkeypatch):
    """Accounts on the GA realtime API reject the 2024 beta shape with
    beta_api_shape_disabled — the session.update must be GA-shaped, and we
    declare audio/pcmu so the model speaks the phone line's codec."""
    from plugins.platforms.voice_call.config import RealtimeConfig
    from plugins.platforms.voice_call.realtime.openai_realtime import (
        OpenAIRealtimeSession,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = OpenAIRealtimeSession(
        RealtimeConfig(enabled=True, provider="openai", model="gpt-realtime-2")
    )
    assert session.audio_wire_format == "ulaw"
    update = session._session_update()
    s = update["session"]
    assert s["type"] == "realtime" and s["model"] == "gpt-realtime-2"
    assert s["output_modalities"] == ["audio"]
    assert s["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert s["audio"]["output"]["format"] == {"type": "audio/pcmu"}
    assert s["audio"]["output"]["voice"] == "marin"
    # Beta-era keys must be absent.
    for beta_key in ("modalities", "voice", "input_audio_format",
                     "output_audio_format", "input_audio_transcription"):
        assert beta_key not in s, beta_key
    assert s["tools"][0]["name"] == "agent_consult"


def test_openai_tool_calls_from_response_done_with_dedupe(monkeypatch):
    """GA delivers function calls inside response.done output items; the
    standalone arguments.done event may also fire — surface each call once."""
    from plugins.platforms.voice_call.config import RealtimeConfig
    from plugins.platforms.voice_call.realtime.openai_realtime import (
        OpenAIRealtimeSession,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = OpenAIRealtimeSession(RealtimeConfig(enabled=True, provider="openai"))

    events = session._translate({
        "type": "response.function_call_arguments.done",
        "call_id": "call_1", "name": "agent_consult",
        "arguments": '{"question": "q1"}',
    })
    assert [e.type for e in events] == ["tool_call"]
    assert events[0].tool_args == {"question": "q1"}

    # Same call inside response.done → deduped; a new one surfaces.
    events = session._translate({
        "type": "response.done",
        "response": {"output": [
            {"type": "function_call", "call_id": "call_1",
             "name": "agent_consult", "arguments": '{"question": "q1"}'},
            {"type": "function_call", "call_id": "call_2",
             "name": "agent_consult", "arguments": '{"question": "q2"}'},
            {"type": "message", "content": []},
        ]},
    })
    assert [e.type for e in events] == ["tool_call", "response_done"]
    assert events[0].tool_call_id == "call_2"


@pytest.mark.asyncio
async def test_bridge_ulaw_passthrough_skips_transcoding(fake_runtime):
    """With a ulaw-native session, carrier frames pass through unmodified."""
    ulaw = bytes(range(160)) + bytes([0x7F]) * 0  # arbitrary µ-law payload
    ws = FakeCarrierWs([
        json.dumps({"event": "start", "stream_id": "s1",
                    "start": {"call_control_id": "v3:rt"}}),
        _media_frame(ulaw),
    ])
    session = FakeSession()
    session.audio_wire_format = "ulaw"
    bridge = _bridge(fake_runtime, _record(), session)

    run = asyncio.create_task(bridge.run(ws))
    await asyncio.wait_for(ws._consumed.wait(), 2)
    await asyncio.sleep(0.02)
    # Inbound: exact bytes, no decode/resample.
    assert session.received_audio == [ulaw]
    # Outbound: model µ-law goes straight to the pacer → carrier frames.
    session.push(RealtimeEvent(type="audio", audio=bytes([0x55]) * 160))
    await asyncio.sleep(0.1)
    session.push(RealtimeEvent(type="closed"))
    await asyncio.wait_for(run, 2)
    import base64 as b64

    media = [m for m in ws.sent if m.get("event") == "media"]
    assert b64.b64decode(media[0]["media"]["payload"]) == bytes([0x55]) * 160


@pytest.mark.asyncio
async def test_manager_skips_carrier_tts_for_realtime_calls(
    vc_config, provider, store
):
    """A realtime call's greeting comes from the model, not carrier TTS."""
    from plugins.platforms.voice_call.events import EventType, NormalizedEvent
    from plugins.platforms.voice_call.manager import CallManager

    manager = CallManager(vc_config, provider, store)
    provider.event_sink = manager.process_event
    manager.prepare_call = lambda record: record.metadata.update(realtime=True)

    record = await manager.initiate_call("+15555550001", message="hi")
    deadline = asyncio.get_running_loop().time() + 1.0
    while not record.answered_at and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.02)
    assert provider.spoken == []        # no carrier TTS
    assert provider.listening == []     # no carrier transcription
    # The initial message stays in metadata for the bridge to inject.
    assert record.metadata["initial_message"] == "hi"
    await manager.shutdown()
