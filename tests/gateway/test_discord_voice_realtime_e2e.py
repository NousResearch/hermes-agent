"""End-to-end proof of the Discord realtime supervisor loop.

A real WebSocket server impersonates xAI's realtime API; everything on the
client side is the production stack: RealtimeVoiceSession (real socket via
the default connector), DiscordMicBridge (48 kHz VC audio in),
MixerPlayoutSink + VoiceMixer (speech out, as discord.py would drain it),
VoiceSupervisorController + DiscordVoiceTurnRunner (consult/steer brain).
Only Discord itself (RTP/opus) and the LLM are faked.

Proves, over a real wire:
1. the session.update sent on connect is the supervisor payload,
2. VC audio fed to the bridge reaches the server as 16 kHz PCM,
3. a consult_hermes call dispatches the task into the gateway pipeline,
4. the instant acknowledgment is a force_message that arrives while the
   Hermes turn is still running (no model round-trip, no waiting),
5. the turn result returns as function_call_output + response.create,
6. streamed speech comes out of the mixer as audible 48 kHz stereo frames.
"""

import asyncio
import base64
import json
import queue
import struct
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

np = pytest.importorskip("numpy")
websockets = pytest.importorskip("websockets")

import os
import sys

_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import realtime_voice as rv  # noqa: E402
import voice_mixer as vm  # noqa: E402

from agent.voice_supervisor import VoiceSupervisorController  # noqa: E402
from gateway.voice_realtime_bridge import DiscordVoiceTurnRunner  # noqa: E402
from tools.voice_realtime import RealtimeVoiceSession  # noqa: E402
from tools.voice_realtime_config import DEFAULT_SUPERVISOR_VOICE, RealtimeConfig  # noqa: E402


def _wait_until(cond, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


class FakeXAIServer:
    """Minimal xAI realtime server: records client events, scripts replies."""

    def __init__(self):
        self.received: "queue.Queue[dict]" = queue.Queue()
        self.events = []  # all client->server events, in order
        self.audio_bytes = 0
        self.loop = None
        self.port = None
        self._conn = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._thread.start()
        assert self._ready.wait(10), "fake xAI server never came up"

    def stop(self):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        from websockets.asyncio.server import serve

        self.loop = asyncio.get_running_loop()
        async with serve(self._handle, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await asyncio.Event().wait()  # runs until loop.stop()

    async def _handle(self, conn):
        self._conn = conn
        try:
            async for raw in conn:
                try:
                    event = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if event.get("type") == "input_audio_buffer.append":
                    self.audio_bytes += len(base64.b64decode(event.get("audio") or ""))
                    continue  # too chatty to record individually
                self.events.append(event)
                self.received.put(event)
        except Exception:
            pass

    # -- test-driven server->client events ----------------------------------

    def send(self, event: dict):
        assert self._conn is not None, "no client connected"
        fut = asyncio.run_coroutine_threadsafe(
            self._conn.send(json.dumps(event)), self.loop
        )
        fut.result(timeout=5)

    def next_event(self, etype: str, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                event = self.received.get(timeout=0.2)
            except queue.Empty:
                continue
            if event.get("type") == etype:
                return event
        raise AssertionError(f"server never received a {etype!r} event")


class StubGatewayRunner:
    """The gateway seam: records consult submissions like run.py would."""

    def __init__(self):
        self.submitted = []

    def _voice_channel_source(self, adapter, guild_id, user_id):
        return SimpleNamespace(guild_id=guild_id, user_id=user_id)

    def _session_key_for_source(self, source):
        return f"agent:main:discord:{source.guild_id}:{source.user_id}"

    async def _handle_voice_channel_input(self, guild_id, user_id, transcript, **kwargs):
        self.submitted.append((guild_id, user_id, transcript))


def _loud_48k_stereo(ms: int) -> bytes:
    n = 48_000 * ms // 1000
    t = np.arange(n, dtype=np.float64)
    mono = (np.sin(2 * np.pi * 220.0 * t / 48_000) * 12000).astype(np.int16)
    return np.repeat(mono, 2).tobytes()


@pytest.fixture
def xai_server():
    server = FakeXAIServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def _fake_creds(monkeypatch):
    import tools.xai_http as xai_http

    monkeypatch.setattr(
        xai_http, "resolve_xai_http_credentials",
        lambda **kw: {"api_key": "xai-test-key", "base_url": "https://api.x.ai/v1"},
    )


@pytest.mark.timeout(60)
def test_full_discord_supervisor_loop(xai_server):
    # ── Assemble the production stack around a real socket ────────────────
    mixer = vm.VoiceMixer(duck_release_ms=40)
    bridge_holder = []

    def mic_factory(on_frame):
        bridge = rv.DiscordMicBridge(on_frame)
        bridge_holder.append(bridge)
        return bridge

    cfg = RealtimeConfig(
        brain="supervisor",
        full_duplex=True,
        idle_pause_seconds=0,
        url=f"ws://127.0.0.1:{xai_server.port}",
    )
    session = RealtimeVoiceSession(
        cfg,
        on_transcript=lambda text: None,
        on_function_call=lambda name, call_id, args: controller.on_function_call(
            name, call_id, args
        ),
        mic_factory=mic_factory,
        playout_sink_factory=lambda: rv.MixerPlayoutSink(lambda: mixer),
        require_local_audio=False,
    )

    gateway = StubGatewayRunner()
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    adapter = SimpleNamespace(
        _voice_sources={5: {"user_id": "1"}},  # joiner ≠ speaker
        _voice_realtime_last_speaker={5: 77},
        _voice_text_channels={5: 900},
        _active_sessions={},
        _pending_messages={},
        interrupt_session_activity=AsyncMock(),
    )
    runner = DiscordVoiceTurnRunner(gateway, adapter, 5, loop)
    controller = VoiceSupervisorController(session, runner, narrate=True)

    # discord.py's sender thread equivalent: drain the mixer continuously.
    mixer_frames = []
    pump_stop = threading.Event()

    def pump():
        while not pump_stop.is_set():
            mixer_frames.append(mixer.read())
            time.sleep(0.005)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()

    try:
        # ── 1. Connect: the supervisor session config travels the wire ────
        session.start()
        session.set_armed(True)
        update = xai_server.next_event("session.update")
        tools = update["session"].get("tools") or []
        assert [t["name"] for t in tools] == ["consult_hermes", "steer_hermes"]
        assert update["session"]["voice"] == DEFAULT_SUPERVISOR_VOICE

        # ── 2. VC audio reaches the server as 16 kHz PCM ──────────────────
        bridge = bridge_holder[0]
        pcm = _loud_48k_stereo(300)
        bridge.feed([pcm])
        expected = len(pcm) // 12 * 2  # 48k stereo -> 16k mono byte ratio
        assert _wait_until(lambda: xai_server.audio_bytes >= expected), (
            f"server got {xai_server.audio_bytes} audio bytes, wanted >= {expected}"
        )

        # ── 3. The model consults Hermes (silently — no audio this turn) ──
        xai_server.send({"type": "response.created", "response": {"id": "r1"}})
        xai_server.send({
            "type": "response.function_call_arguments.done",
            "name": "consult_hermes",
            "call_id": "call-1",
            "arguments": json.dumps({"task": "check disk usage on the server"}),
        })
        xai_server.send({"type": "response.done"})

        # The task lands in the gateway pipeline (Hermes does the work)...
        assert _wait_until(lambda: gateway.submitted)
        assert gateway.submitted[0] == (5, 77, "check disk usage on the server")

        # ── 4. ...and the ack is INSTANT: a force_message (no model turn),
        # sent while the Hermes turn is still running (nothing completed yet).
        ack = xai_server.next_event("conversation.item.create")
        assert ack["item"]["type"] == "force_message"
        assert ack["item"]["content"][0]["text"]
        assert controller.consult_active  # turn not finished when ack went out

        # The ack's speech streams back and is audible through the mixer.
        ack_pcm = struct.pack("<h", 9000) * 4800  # 200ms @ 24k mono
        xai_server.send({"type": "response.created", "response": {"id": "r2"}})
        xai_server.send({
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(ack_pcm).decode(),
        })
        xai_server.send({"type": "response.done"})
        assert _wait_until(lambda: any(
            frame != vm.SILENCE_FRAME and int(
                np.max(np.abs(np.frombuffer(frame, dtype=np.int16)))
            ) > 1000
            for frame in mixer_frames[-100:]
        )), "supervisor speech never became audible through the voice mixer"

        # ── 5. Hermes finishes → result returns to the voice model ────────
        assert controller.on_turn_complete(
            "check disk usage on the server", "Disk is 42% full."
        ) is True
        output = xai_server.next_event("conversation.item.create")
        assert output["item"]["type"] == "function_call_output"
        assert output["item"]["call_id"] == "call-1"
        assert output["item"]["output"] == "Disk is 42% full."
        xai_server.next_event("response.create")  # summary requested
        assert not controller.consult_active
    finally:
        pump_stop.set()
        pump_thread.join(timeout=2)
        session.stop()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        loop.close()
