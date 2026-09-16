"""Tests for the Discord voice-channel realtime (xAI S2S) surface.

Covers the audio bridges (resamplers, mic bridge, mixer playout sink), the
streaming mixer child, continuous receiver draining, the gateway TurnRunner,
and the runner-side controller lifecycle. No network, no audio hardware,
no live Discord connection.
"""

import asyncio
import os
import struct
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# numpy ships only in the optional "voice" extra (CI does not install it).
# Only the resampling/mixing classes need it; everything else (wake gate,
# remaps, controller lifecycle, turn runner, listen loop) must keep running
# in CI, so the skip is per-class rather than module-wide.
try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in CI
    np = None

_needs_numpy = pytest.mark.skipif(np is None, reason="numpy (voice extra) not installed")

# The discord plugin modules import by path (same pattern as the adapter and
# the existing mixer tests).
_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import realtime_voice as rv  # noqa: E402
import voice_mixer as vm  # noqa: E402


# =====================================================================
# Resamplers
# =====================================================================

@_needs_numpy
class TestResamplers:
    def test_downsample_ratio_and_remainder(self):
        # 48kHz stereo s16 → 16kHz mono s16 is 12 bytes in : 2 bytes out.
        pcm = struct.pack("<6h", 100, 100, 100, 100, 100, 100)  # 3 stereo frames
        out, remainder = rv.downsample_48k_stereo_to_16k_mono(pcm)
        assert remainder == b""
        assert len(out) == 2
        assert struct.unpack("<h", out)[0] == 100  # constant in → constant out

        # A trailing partial group is carried, not dropped.
        out2, rem2 = rv.downsample_48k_stereo_to_16k_mono(pcm + b"\x01\x02")
        assert len(out2) == 2
        assert rem2 == b"\x01\x02"

    def test_downsample_short_input_is_all_remainder(self):
        out, remainder = rv.downsample_48k_stereo_to_16k_mono(b"\x01\x02\x03\x04")
        assert out == b""
        assert remainder == b"\x01\x02\x03\x04"

    def test_downsample_mixes_channels(self):
        # L=200, R=0 across 3 frames → mono mean 100.
        pcm = struct.pack("<6h", 200, 0, 200, 0, 200, 0)
        out, _ = rv.downsample_48k_stereo_to_16k_mono(pcm)
        assert struct.unpack("<h", out)[0] == 100

    def test_upsample_linear_interpolation_and_dual_mono(self):
        # 24kHz mono s16 → 48kHz stereo s16 is 1 sample in : 4 samples out.
        pcm = struct.pack("<2h", 1000, -2000)
        out = rv.upsample_24k_mono_to_48k_stereo(pcm)
        assert len(out) == len(pcm) * 4
        samples = struct.unpack("<8h", out)
        # Midpoint between 1000 and -2000 is -500, then dual-mono.
        # Last 48 kHz frame holds the final 24 kHz sample.
        assert samples == (1000, 1000, -500, -500, -2000, -2000, -2000, -2000)
        # Zero-order hold (repeat-4) must not sneak back in.
        assert samples != (1000, 1000, 1000, 1000, -2000, -2000, -2000, -2000)

    def test_upsample_ramp_inserts_midpoints(self):
        pcm = struct.pack("<3h", 0, 10, 20)
        out = rv.upsample_24k_mono_to_48k_stereo(pcm)
        assert len(out) == len(pcm) * 4
        samples = struct.unpack("<12h", out)
        # 48 kHz mono 0, 5, 10, 15, 20, 20 then dual-mono.
        assert samples == (0, 0, 5, 5, 10, 10, 15, 15, 20, 20, 20, 20)

    def test_upsample_trims_odd_trailing_byte(self):
        out = rv.upsample_24k_mono_to_48k_stereo(struct.pack("<h", 7) + b"\x01")
        assert len(out) == 8
        assert struct.unpack("<4h", out) == (7, 7, 7, 7)


# =====================================================================
# DiscordMicBridge
# =====================================================================

@_needs_numpy
class TestMixPcm16:
    def test_single_chunk_passes_through(self):
        pcm = struct.pack("<4h", 1, -2, 3, -4)
        assert rv.mix_pcm16([pcm]) == pcm
        assert rv.mix_pcm16([b"", pcm]) == pcm

    def test_empty_input(self):
        assert rv.mix_pcm16([]) == b""
        assert rv.mix_pcm16([b"", b""]) == b""

    def test_simultaneous_speakers_are_summed_not_concatenated(self):
        a = struct.pack("<3h", 100, 100, 100)
        b = struct.pack("<3h", 20, -20, 20)
        out = rv.mix_pcm16([a, b])
        # Same duration as one speaker (the real-time clock), summed sample-wise.
        assert len(out) == len(a)
        assert struct.unpack("<3h", out) == (120, 80, 120)

    def test_shorter_chunk_aligns_to_the_end_of_the_window(self):
        # Both chunks were drained at the same instant, so a speaker who
        # started mid-window contributes to the *latest* samples.
        a = struct.pack("<4h", 10, 10, 10, 10)
        b = struct.pack("<2h", 5, 5)
        assert struct.unpack("<4h", rv.mix_pcm16([a, b])) == (10, 10, 15, 15)

    def test_sum_saturates_at_int16(self):
        a = struct.pack("<2h", 30000, -30000)
        b = struct.pack("<2h", 30000, -30000)
        assert struct.unpack("<2h", rv.mix_pcm16([a, b])) == (32767, -32768)

    def test_odd_trailing_byte_is_dropped(self):
        a = struct.pack("<2h", 1, 1) + b"\x7f"
        b = struct.pack("<2h", 2, 2)
        assert struct.unpack("<2h", rv.mix_pcm16([a, b])) == (3, 3)


@_needs_numpy
class TestDiscordMicBridge:
    def test_feed_downsamples_and_forwards(self):
        frames = []
        bridge = rv.DiscordMicBridge(frames.append)
        bridge.feed([struct.pack("<6h", 100, 100, 100, 100, 100, 100)])
        assert len(frames) == 1
        assert struct.unpack("<h", frames[0])[0] == 100

    def test_carry_across_feeds(self):
        frames = []
        bridge = rv.DiscordMicBridge(frames.append)
        whole = struct.pack("<6h", 50, 50, 50, 50, 50, 50)
        # Split one 12-byte group across two feeds — nothing emitted first,
        # the carry completes the group on the second call.
        bridge.feed([whole[:8]])
        assert frames == []
        bridge.feed([whole[8:]])
        assert len(frames) == 1
        assert struct.unpack("<h", frames[0])[0] == 50

    def test_two_speakers_in_one_drain_become_one_frame(self):
        frames = []
        bridge = rv.DiscordMicBridge(frames.append)
        a = struct.pack("<6h", 100, 100, 100, 100, 100, 100)
        b = struct.pack("<6h", 20, 20, 20, 20, 20, 20)
        bridge.feed([a, b])
        # One window of real time → one 16 kHz sample, not two back to back.
        assert len(frames) == 1
        assert struct.unpack("<h", frames[0])[0] == 120

    def test_close_stops_forwarding(self):
        frames = []
        bridge = rv.DiscordMicBridge(frames.append)
        bridge.close()
        bridge.feed([struct.pack("<6h", 1, 1, 1, 1, 1, 1)])
        assert frames == []


# =====================================================================
# StreamSpeechChild + mixer integration
# =====================================================================

def _loud_pcm(n_frames: int, value: int = 8000) -> bytes:
    return struct.pack("<h", value) * (n_frames * vm.FRAME_SIZE // 2)


@_needs_numpy
class TestStreamSpeechChild:
    def test_starving_is_not_finished(self):
        child = vm.StreamSpeechChild("s")
        assert child.read_frame() is None
        assert child.finished is False

    def test_frames_flow_and_partial_waits(self):
        child = vm.StreamSpeechChild("s", fade_in_ms=0)
        child.feed(b"\x01\x00" * 10)  # far less than one frame
        assert child.read_frame() is None  # waits for a whole frame
        child.feed(_loud_pcm(1))
        frame = child.read_frame()
        assert frame is not None
        assert len(frame) == vm.FRAME_SIZE // 2  # int16 samples

    def test_end_flushes_partial_then_finishes(self):
        child = vm.StreamSpeechChild("s", fade_in_ms=0)
        child.feed(b"\x01\x00" * 10)
        child.end()
        assert child.read_frame() is not None  # zero-padded final frame
        assert child.read_frame() is None
        assert child.finished is True

    def test_clear_drops_buffer(self):
        child = vm.StreamSpeechChild("s")
        child.feed(_loud_pcm(4))
        child.clear()
        assert child.buffered_bytes == 0
        assert child.read_frame() is None

    def test_feed_after_end_is_ignored(self):
        child = vm.StreamSpeechChild("s")
        child.end()
        child.feed(_loud_pcm(1))
        assert child.finished is True


@_needs_numpy
class TestMixerStreaming:
    def _mixer_with_ambient(self):
        mx = vm.VoiceMixer(ambient_gain=0.2, duck_gain=0.05, duck_release_ms=40)
        mx.set_ambient(vm.synth_ambient_pcm(seconds=0.5))
        return mx

    def test_stream_survives_starvation_and_ducks_by_audibility(self):
        mx = self._mixer_with_ambient()
        child = vm.StreamSpeechChild("rt", fade_in_ms=0)
        mx.attach_speech_stream(child)

        # Starving stream: not audible, ambient at full gain, child kept.
        for _ in range(5):
            mx.read()
        assert mx.speech_active is False
        assert mx._ambient.gain == pytest.approx(0.2)

        # Feed audio → duck engages and the stream is mixed in.
        child.feed(_loud_pcm(3))
        frame = np.frombuffer(mx.read(), dtype=np.int16)
        assert int(np.max(np.abs(frame))) >= 7000
        assert mx.speech_active is True
        assert mx._ambient.gain == pytest.approx(0.05)

        # Drain (2 more frames) then starve → duck releases, child stays.
        mx.read()
        mx.read()
        for _ in range(10):
            mx.read()
        assert mx.speech_active is False
        assert mx._ambient.gain == pytest.approx(0.2)

        # Resumed audio re-ducks — the same child keeps working.
        child.feed(_loud_pcm(1))
        mx.read()
        assert mx.speech_active is True

    def test_attach_is_idempotent(self):
        mx = vm.VoiceMixer()
        child = vm.StreamSpeechChild("rt")
        mx.attach_speech_stream(child)
        mx.attach_speech_stream(child)
        assert mx._speech.count(child) == 1

    def test_finished_stream_is_dropped(self):
        mx = vm.VoiceMixer()
        child = vm.StreamSpeechChild("rt", fade_in_ms=0)
        mx.attach_speech_stream(child)
        child.feed(_loud_pcm(1))
        child.end()
        mx.read()  # plays the last frame
        mx.read()  # sees finished → drops
        assert child not in mx._speech

    def test_one_shot_clips_unaffected(self):
        # Regression guard: play_speech duck semantics survive the
        # audibility-driven rework.
        mx = self._mixer_with_ambient()
        mx.play_speech(_loud_pcm(2), fade_in_ms=0)
        assert mx.speech_active is True
        mx.read()
        assert mx._ambient.gain == pytest.approx(0.05)
        mx.read()
        for _ in range(10):
            mx.read()
        assert mx.speech_active is False
        assert mx._ambient.gain == pytest.approx(0.2)


# =====================================================================
# MixerPlayoutSink
# =====================================================================

@_needs_numpy
class TestMixerPlayoutSink:
    def test_write_upsamples_feeds_and_attaches(self):
        mx = vm.VoiceMixer()
        sink = rv.MixerPlayoutSink(lambda: mx)
        chunk = struct.pack("<h", 5000) * 100  # 24k mono
        sink.write(chunk)
        assert sink.pending() is True
        child = sink._child
        assert child in mx._speech
        assert child.buffered_bytes == len(chunk) * 4

    def test_write_without_mixer_drops(self):
        sink = rv.MixerPlayoutSink(lambda: None)
        sink.write(struct.pack("<h", 1) * 10)
        assert sink.pending() is False

    def test_clear_empties_pending(self):
        mx = vm.VoiceMixer()
        sink = rv.MixerPlayoutSink(lambda: mx)
        sink.write(struct.pack("<h", 1) * 10)
        sink.clear()
        assert sink.pending() is False

    def test_reattaches_after_stop_speech(self):
        mx = vm.VoiceMixer()
        sink = rv.MixerPlayoutSink(lambda: mx)
        sink.write(struct.pack("<h", 1) * 10)
        mx.stop_speech()  # barge: mixer detaches everything
        assert sink._child not in mx._speech
        sink.write(struct.pack("<h", 1) * 10)
        assert sink._child in mx._speech

    def test_close_ends_child_so_mixer_drops_it(self):
        mx = vm.VoiceMixer()
        sink = rv.MixerPlayoutSink(lambda: mx)
        sink.write(struct.pack("<h", 1) * 10)
        child = sink._child
        sink.close()
        assert child.finished is False  # still drains the tail
        while mx._speech:
            mx.read()
        assert child.finished is True


# =====================================================================
# VoiceReceiver.drain_pending
# =====================================================================

def _make_receiver():
    from plugins.platforms.discord.adapter import VoiceReceiver

    vc = MagicMock()
    receiver = VoiceReceiver(vc, allowed_user_ids={"1"})
    return receiver


class TestDrainPending:
    def test_drains_mapped_users_and_clears(self):
        r = _make_receiver()
        r._ssrc_to_user[111] = 1
        r._buffers[111].extend(b"\x01\x02" * 100)
        drained = r.drain_pending()
        assert drained == [(1, b"\x01\x02" * 100)]
        assert len(r._buffers[111]) == 0
        assert r.drain_pending() == []  # nothing left

    def test_unmapped_ssrc_is_kept_but_capped(self):
        r = _make_receiver()
        r._vc.channel = None  # no inference possible
        cap = 2 * r.SAMPLE_RATE * r.CHANNELS * 2
        r._buffers[222].extend(b"\x00" * (cap + 5000))
        assert r.drain_pending() == []
        assert len(r._buffers[222]) == cap  # trimmed to the newest ~2s

    def test_unmapped_becomes_available_once_mapped(self):
        r = _make_receiver()
        r._vc.channel = None
        r._buffers[222].extend(b"\x07\x08" * 10)
        assert r.drain_pending() == []
        r.map_ssrc(222, 42)
        assert r.drain_pending() == [(42, b"\x07\x08" * 10)]


# =====================================================================
# Adapter config gate
# =====================================================================

class TestAdapterRealtimeConfig:
    def _load(self, monkeypatch, cfg):
        from plugins.platforms.discord.adapter import DiscordAdapter

        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: cfg)
        adapter = object.__new__(DiscordAdapter)
        return adapter._load_voice_realtime_config()

    def test_disabled_by_default(self, monkeypatch):
        assert self._load(monkeypatch, {}) is None
        assert self._load(monkeypatch, {"voice": {"realtime": {"enabled": True}}}) is None
        assert self._load(
            monkeypatch, {"voice": {"realtime": {"discord": True}}}
        ) is None  # global enable still required

    def test_enabled_forces_discord_overrides(self, monkeypatch):
        cfg = self._load(monkeypatch, {
            "voice": {"realtime": {
                "enabled": True,
                "discord": True,
                "brain": "supervisor",
                "full_duplex": False,       # forced True for VC
                "idle_pause_seconds": 120,  # forced 0 for VC
            }},
        })
        assert cfg is not None
        assert cfg.supervisor is True
        assert cfg.full_duplex is True
        assert cfg.idle_pause_seconds == 0.0

    def test_omitted_brain_is_the_loaders_default(self, monkeypatch):
        """Discord pins no brain of its own: a config without ``brain`` runs whatever
        ``load_realtime_config`` defaults to (supervisor, hence the mixer requirement at join)."""
        from tools.voice_realtime_config import load_realtime_config

        cfg = self._load(monkeypatch, {"voice": {"realtime": {"enabled": True, "discord": True}}})
        assert cfg is not None
        assert cfg.brain == load_realtime_config({}).brain == "supervisor"


# =====================================================================
# Join-time gate: supervisor speech needs the mixer
# =====================================================================

class _FakeReceiver:
    def __init__(self, vc, allowed_user_ids=None):
        self._running = False

    def start(self):
        pass


def _join_adapter(monkeypatch, *, supervisor, mixer_installs):
    import plugins.platforms.discord.adapter as adapter_module
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(adapter_module, "VoiceReceiver", _FakeReceiver)

    adapter = object.__new__(DiscordAdapter)
    adapter._client = MagicMock()
    adapter._voice_locks = {}
    adapter._voice_clients = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._voice_realtime = {}
    adapter._voice_realtime_mics = {}
    adapter._voice_realtime_last_speaker = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._allowed_user_ids = set()
    adapter._voice_fx_cfg = {"enabled": False}
    adapter._reset_voice_timeout = MagicMock()
    adapter._load_voice_realtime_config = MagicMock(
        return_value=SimpleNamespace(supervisor=supervisor)
    )
    adapter._start_voice_realtime = MagicMock(return_value=True)

    async def _install(guild_id, vc):
        if not mixer_installs:
            raise RuntimeError("no ffmpeg")
        adapter._voice_mixers[guild_id] = MagicMock()

    adapter._install_voice_mixer = _install

    channel = MagicMock()
    channel.guild.id = 5
    channel.connect = AsyncMock(return_value=MagicMock())
    return adapter, channel


class TestJoinRealtimeGate:
    async def _join(self, adapter, channel):
        assert await adapter.join_voice_channel(channel) is True
        task = adapter._voice_listen_tasks.get(5)
        if task is not None:
            await task

    @pytest.mark.asyncio
    async def test_supervisor_without_mixer_stays_classic(self, monkeypatch):
        adapter, channel = _join_adapter(
            monkeypatch, supervisor=True, mixer_installs=False
        )
        await self._join(adapter, channel)
        adapter._start_voice_realtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_supervisor_with_mixer_starts_realtime(self, monkeypatch):
        adapter, channel = _join_adapter(
            monkeypatch, supervisor=True, mixer_installs=True
        )
        await self._join(adapter, channel)
        adapter._start_voice_realtime.assert_called_once()

    @pytest.mark.asyncio
    async def test_ears_brain_needs_no_mixer(self, monkeypatch):
        adapter, channel = _join_adapter(
            monkeypatch, supervisor=False, mixer_installs=False
        )
        await self._join(adapter, channel)
        # Ears never plays server audio, so the missing mixer is irrelevant —
        # and the fx-off config means no mixer install was even attempted.
        adapter._start_voice_realtime.assert_called_once()
        assert adapter._voice_mixers == {}


# =====================================================================
# DiscordVoiceTurnRunner (gateway side)
# =====================================================================

def _make_turn_runner(loop, *, speaker_id: int = 77):
    from gateway.voice_realtime_bridge import DiscordVoiceTurnRunner

    gateway_runner = MagicMock()
    gateway_runner._handle_voice_channel_input = AsyncMock()
    gateway_runner._voice_channel_source = MagicMock(return_value=SimpleNamespace(chat_id="900"))

    # Session key must track the user id the runner asks for (speaker, not joiner).
    def _session_key_for_source(_source):
        user_id = gateway_runner._voice_channel_source.call_args[0][2]
        return f"agent:main:discord:chan9:{user_id}"

    gateway_runner._session_key_for_source = MagicMock(side_effect=_session_key_for_source)
    adapter = SimpleNamespace(
        _voice_sources={5: {"user_id": "1"}},  # joiner ≠ speaker
        _voice_realtime_last_speaker={5: speaker_id},
        _voice_realtime_last_transcribed={},
        _voice_text_channels={5: 900},
        _voice_clients={},
        _voice_realtime={},
        _active_sessions={},
        _pending_messages={},
        interrupt_session_activity=AsyncMock(),
    )
    return DiscordVoiceTurnRunner(gateway_runner, adapter, 5, loop), gateway_runner, adapter


class TestDiscordVoiceTurnRunner:
    @pytest.mark.asyncio
    async def test_submit_attributes_to_actual_speaker_not_joiner(self):
        loop = asyncio.get_running_loop()
        runner, gw, adapter = _make_turn_runner(loop, speaker_id=99)
        assert adapter._voice_sources[5]["user_id"] == "1"  # joiner ≠ speaker
        runner.submit("check the logs")
        for _ in range(50):
            if gw._handle_voice_channel_input.await_count:
                break
            await asyncio.sleep(0.01)
        gw._handle_voice_channel_input.assert_awaited_once_with(
            5, 99, "check the logs", consult=True, adapter=adapter
        )

    @pytest.mark.asyncio
    async def test_submit_without_speaker_context_is_dropped(self):
        loop = asyncio.get_running_loop()
        runner, gw, adapter = _make_turn_runner(loop)
        adapter._voice_realtime_last_speaker = {}
        runner.submit("task")
        await asyncio.sleep(0.05)
        gw._handle_voice_channel_input.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_prefers_last_transcribed_speaker(self):
        loop = asyncio.get_running_loop()
        runner, gw, adapter = _make_turn_runner(loop, speaker_id=99)
        adapter._voice_realtime_last_transcribed = {5: 42}
        runner.submit("task")
        for _ in range(50):
            if gw._handle_voice_channel_input.await_count:
                break
            await asyncio.sleep(0.01)
        gw._handle_voice_channel_input.assert_awaited_once_with(
            5, 42, "task", consult=True, adapter=adapter
        )

    @pytest.mark.asyncio
    async def test_busy_and_queue_follow_latched_speaker_session(self):
        loop = asyncio.get_running_loop()
        runner, _, adapter = _make_turn_runner(loop, speaker_id=77)
        runner.submit("task")  # latch speaker 77
        # Another user talks mid-consult — busy/queue must stay on 77.
        adapter._voice_realtime_last_speaker[5] = 88
        key = "agent:main:discord:chan9:77"
        assert runner.is_busy() is False
        assert runner.is_queue_empty() is True
        adapter._active_sessions[key] = object()
        adapter._pending_messages[key] = object()
        assert runner.is_busy() is True
        assert runner.is_queue_empty() is False

    @pytest.mark.asyncio
    async def test_interrupt_uses_adapter_session_interrupt(self):
        loop = asyncio.get_running_loop()
        runner, _, adapter = _make_turn_runner(loop, speaker_id=77)
        runner.submit("task")  # latch speaker before interrupt
        runner.interrupt()
        for _ in range(50):
            if adapter.interrupt_session_activity.await_count:
                break
            await asyncio.sleep(0.01)
        adapter.interrupt_session_activity.assert_awaited_once_with(
            "agent:main:discord:chan9:77", "900"
        )


# =====================================================================
# GatewayRunner controller lifecycle
# =====================================================================

@pytest.fixture
def bg_loop():
    """A real event loop on a background thread — run_coroutine_threadsafe
    targets from controller/session threads actually execute."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def _make_gateway_runner(session, loop):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    # The bound text channel's source as stored at /voice join (the joiner, user 1 ≠ speaker 77).
    bound_source = SessionSource(
        platform=Platform.DISCORD, chat_id="900", chat_name="#general", chat_type="channel",
        user_id="1", user_name="joiner",
    )
    adapter = SimpleNamespace(
        voice_realtime_session=lambda gid: session if gid == 5 else None,
        _voice_sources={5: bound_source.to_dict()},
        _voice_realtime_last_speaker={5: 77},
        _voice_realtime_last_transcribed={},
        _voice_text_channels={5: 900},
        _voice_clients={},
        _voice_realtime={},
        _active_sessions={},
        _pending_messages={},
        interrupt_session_activity=AsyncMock(),
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_realtime_controllers = {}
    runner._gateway_loop = loop
    return runner, adapter


def _fake_session(alive=True):
    return SimpleNamespace(
        alive=alive,
        last_response_had_audio=True,
        send_function_output=MagicMock(),
        speak_acknowledgment=MagicMock(),
        speak_verbatim=MagicMock(),
    )


def _wait_until(cond, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


class TestControllerLifecycle:
    def test_discord_controller_narrate_defaults_off(self, bg_loop, monkeypatch):
        session = _fake_session()
        runner, _adapter = _make_gateway_runner(session, bg_loop)
        monkeypatch.setattr("gateway.voice_realtime_mixin.read_raw_config", lambda: {})
        ctrl = runner._ensure_voice_realtime_controller(5)
        assert ctrl is not None
        assert ctrl._narrate is False

    def test_discord_controller_narrate_yaml_opt_in(self, bg_loop, monkeypatch):
        session = _fake_session()
        runner, _adapter = _make_gateway_runner(session, bg_loop)
        monkeypatch.setattr(
            "gateway.voice_realtime_mixin.read_raw_config",
            lambda: {"voice": {"realtime": {"narrate_progress": True}}},
        )
        ctrl = runner._ensure_voice_realtime_controller(5)
        assert ctrl is not None
        assert ctrl._narrate is True

    def test_function_call_resolves_against_the_bound_adapter(self, bg_loop):
        """Multiplex: a secondary profile's bot fires the callback bound to
        ITS adapter; the controller must be built from that adapter's
        session, not the default adapter's (which has none for this guild)."""
        import functools

        runner, default_adapter = _make_gateway_runner(None, bg_loop)
        session = _fake_session()
        secondary = SimpleNamespace(
            voice_realtime_session=lambda gid: session if gid == 5 else None,
            _voice_sources={5: {"user_id": "1"}},
            _voice_realtime_last_speaker={5: 77},
            _voice_realtime_last_transcribed={},
            _voice_text_channels={5: 900},
            _voice_clients={},
            _voice_realtime={},
            _active_sessions={},
            _pending_messages={},
            interrupt_session_activity=AsyncMock(),
        )
        submitted = AsyncMock()
        runner._handle_voice_channel_input = submitted
        # What _bind_voice_input_callback installs on the secondary adapter.
        callback = functools.partial(
            runner._handle_voice_channel_function_call, adapter=secondary
        )
        callback(5, "consult_hermes", "call-1", '{"task": "list the repos"}')
        controller = runner._voice_realtime_controllers[(None, 5)]
        assert controller.session is session
        assert _wait_until(lambda: submitted.await_count == 1)
        submitted.assert_awaited_once_with(
            5, 77, "list the repos", consult=True, adapter=secondary
        )
        assert default_adapter.voice_realtime_session(5) is None

    def test_controller_built_reused_and_rebuilt_on_new_session(self, bg_loop):
        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        c1 = runner._ensure_voice_realtime_controller(5)
        assert c1 is not None and c1.session is session
        assert runner._ensure_voice_realtime_controller(5) is c1

        # VC reconnect: adapter now serves a fresh session object.
        session2 = _fake_session()
        adapter.voice_realtime_session = lambda gid: session2
        c2 = runner._ensure_voice_realtime_controller(5)
        assert c2 is not c1 and c2.session is session2

    def test_reconnect_fails_out_in_flight_consult(self, bg_loop):
        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        runner._handle_voice_channel_input = AsyncMock()
        runner._handle_voice_channel_function_call(
            5, "consult_hermes", "call-1", '{"task": "list the repos"}'
        )
        assert runner._voice_realtime_controllers[(None, 5)].consult_active
        session2 = _fake_session()
        adapter.voice_realtime_session = lambda gid: session2
        c2 = runner._ensure_voice_realtime_controller(5)
        session.send_function_output.assert_called()
        assert "dropped" in session.send_function_output.call_args[0][1].lower()
        assert c2.session is session2
        assert not c2.consult_active

    def test_no_session_means_no_controller(self, bg_loop):
        runner, adapter = _make_gateway_runner(None, bg_loop)
        assert runner._ensure_voice_realtime_controller(5) is None
        assert runner._voice_realtime_controllers == {}

    def test_dead_session_controller_is_evicted_and_releases_the_turn(self, bg_loop):
        """The adapter tears down a permanently dead xAI session without telling the gateway
        and no further function call can arrive from it. The controller lookup must treat that
        session as gone: pop the entry, fail the in-flight consult, and stop claiming the
        guild's turns so the classic pipeline's acks and progress bubbles come back."""
        from gateway.config import Platform
        from gateway.platforms.event import MessageType

        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        runner._handle_voice_channel_input = AsyncMock()
        runner._handle_voice_channel_function_call(
            5, "consult_hermes", "call-1", '{"task": "list the repos"}'
        )
        controller = runner._voice_realtime_controllers[(None, 5)]
        assert controller.consult_active
        source = SimpleNamespace(platform=Platform.DISCORD, chat_id="900")
        assert runner._voice_consult_owns_turn(source, MessageType.TEXT, "list the repos")

        session.alive = False

        assert runner._voice_realtime_controller(adapter, 5) is None
        assert (None, 5) not in runner._voice_realtime_controllers
        assert not controller.consult_active
        assert not runner._voice_consult_owns_turn(source, MessageType.TEXT, "list the repos")
        ack_ctx = SimpleNamespace(_voice_ack_guild=(5, None), source=source)
        assert runner._voice_realtime_controller_for_ack(ack_ctx) is None

    def test_live_session_controller_survives_lookup(self, bg_loop):
        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        controller = runner._ensure_voice_realtime_controller(5)
        assert runner._voice_realtime_controller(adapter, 5) is controller
        assert runner._voice_realtime_controllers[(None, 5)] is controller

    def test_function_call_dispatches_consult_as_speaker_not_joiner(self, bg_loop):
        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        assert adapter._voice_sources[5]["user_id"] == "1"
        assert adapter._voice_realtime_last_speaker[5] == 77
        submitted = AsyncMock()
        runner._handle_voice_channel_input = submitted  # instance shadow
        runner._handle_voice_channel_function_call(
            5, "consult_hermes", "call-1", '{"task": "list the repos"}'
        )
        controller = runner._voice_realtime_controllers[(None, 5)]
        assert controller.consult_active is True
        assert _wait_until(lambda: submitted.await_count == 1)
        submitted.assert_awaited_once_with(
            5, 77, "list the repos", consult=True, adapter=adapter
        )

    def test_consult_turn_silences_classic_tts_paths(self, bg_loop):
        """_voice_consult_owns_turn gates BOTH classic TTS paths (base
        adapter auto-TTS + streaming TTS) for consult turns — the leak that
        read the whole Hermes reply over the supervisor's spoken summary."""
        from gateway.config import Platform
        from gateway.platforms.event import MessageType

        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        runner._handle_voice_channel_function_call(
            5, "consult_hermes", "call-1", '{"task": "list the repos"}'
        )
        source = SimpleNamespace(platform=Platform.DISCORD, chat_id="900")

        assert runner._voice_consult_owns_turn(
            source, MessageType.VOICE, "list the repos"
        ) is True
        # Consults are agent TEXT turns, not fake STT.
        assert runner._voice_consult_owns_turn(
            source, MessageType.TEXT, "list the repos"
        ) is True
        # Merged steer text still belongs to the consult.
        assert runner._voice_consult_owns_turn(
            source, MessageType.VOICE, "list the repos\n\nalso sort them"
        ) is True
        # Unrelated utterances, photos, and other chats do not.
        assert runner._voice_consult_owns_turn(
            source, MessageType.VOICE, "what's the weather"
        ) is False
        assert runner._voice_consult_owns_turn(
            source, MessageType.PHOTO, "list the repos"
        ) is False
        assert runner._voice_consult_owns_turn(
            SimpleNamespace(platform=Platform.DISCORD, chat_id="999"),
            MessageType.VOICE, "list the repos",
        ) is False

    def test_supervisor_session_silences_all_classic_voice_replies(self, bg_loop):
        """While a supervisor session is live, no classic voice reply fires
        for its bound chat — typed messages included (grok owns the speaker).
        The ears brain keeps classic replies: it has no other voice."""
        from gateway.config import Platform
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.session import SessionSource

        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        runner._voice_mode = {"discord:900": "all"}
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="900", user_id="77",
            chat_type="channel",
        )
        typed_event = MessageEvent(
            source=source, text="typed message",
            message_type=MessageType.TEXT,
            raw_message=SimpleNamespace(guild_id=5, guild=None),
        )

        adapter.voice_realtime_brain = lambda gid: "supervisor" if gid == 5 else ""
        assert runner._should_send_voice_reply(typed_event, "a reply", []) is False

        adapter.voice_realtime_brain = lambda gid: "ears"
        assert runner._should_send_voice_reply(typed_event, "a reply", []) is True

        # Dead/absent session (brain "") restores classic behavior too.
        adapter.voice_realtime_brain = lambda gid: ""
        assert runner._should_send_voice_reply(typed_event, "a reply", []) is True

    def test_controller_for_event_requires_discord_voice_event(self, bg_loop):
        from gateway.config import Platform
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.session import SessionSource

        session = _fake_session()
        runner, _ = _make_gateway_runner(session, bg_loop)
        controller = runner._ensure_voice_realtime_controller(5)
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="900", user_id="77",
            chat_type="channel",
        )
        voice_event = MessageEvent(
            source=source, text="list the repos",
            message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=5, guild=None),
        )
        assert runner._voice_realtime_controller_for_event(voice_event) is controller

        text_event = MessageEvent(
            source=source, text="hello",
            message_type=MessageType.TEXT,
            raw_message=SimpleNamespace(guild_id=5, guild=None),
        )
        assert runner._voice_realtime_controller_for_event(text_event) is None

        text_event.voice_consult = True
        assert runner._voice_realtime_controller_for_event(text_event) is controller


# =====================================================================
# Realtime listen-loop drain + join greeting
# =====================================================================

def _bare_discord_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._client = None
    adapter._voice_clients = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_receivers = {}
    # Realtime state exactly as DiscordAdapter.__init__ declares it.
    adapter._voice_realtime = {}
    adapter._voice_realtime_mics = {}
    adapter._voice_realtime_last_speaker = {}
    adapter._voice_realtime_last_transcribed = {}
    adapter._voice_wake_last = {}
    adapter._voice_pending_consults = {}
    adapter._voice_rt_join_greeted = set()
    adapter._voice_rt_drain_errors = set()
    adapter._voice_function_call_callback = None
    adapter._is_allowed_user = MagicMock(return_value=True)
    adapter.platform = SimpleNamespace(value="discord")
    return adapter


class TestStartVoiceRealtime:
    @pytest.mark.asyncio
    async def test_rejoin_stops_the_previous_session_first(self, monkeypatch):
        """/voice join after the VoiceClient dropped must replace the guild's
        realtime session, not leave the old one's threads/socket/callbacks
        running beside the new one."""
        import tools.voice_realtime as vr

        adapter = _bare_discord_adapter()
        adapter._voice_mixers = {}
        adapter._voice_fx_cfg = {}
        old = MagicMock(alive=True)
        adapter._voice_realtime[5] = old
        adapter._voice_realtime_mics[5] = MagicMock()
        new = MagicMock()
        monkeypatch.setattr(vr, "RealtimeVoiceSession", MagicMock(return_value=new))

        ok = adapter._start_voice_realtime(5, SimpleNamespace(supervisor=False, brain="ears"))

        assert ok is True
        old.stop.assert_called_once()
        assert adapter._voice_realtime[5] is new
        new.start.assert_called_once()


class TestRealtimeListenLoop:
    @pytest.mark.asyncio
    async def test_drain_error_keeps_loop_alive(self):
        adapter = _bare_discord_adapter()
        receiver = MagicMock()
        receiver._running = True
        receiver.drain_pending.side_effect = RuntimeError("drain boom")
        adapter._voice_receivers[5] = receiver
        adapter._voice_realtime[5] = SimpleNamespace(alive=True)
        adapter._voice_realtime_mics[5] = MagicMock()

        task = asyncio.create_task(adapter._voice_listen_loop(5))
        try:
            await asyncio.sleep(0.16)
            assert receiver.drain_pending.call_count >= 2
            assert 5 in adapter._voice_rt_drain_errors
        finally:
            receiver._running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestJoinGreeting:
    def test_connected_state_schedules_speak_verbatim(self):
        adapter = _bare_discord_adapter()
        session = MagicMock()
        session._cfg = SimpleNamespace(supervisor=True)
        session.speak_verbatim.return_value = True
        adapter._voice_realtime[5] = session
        loop = MagicMock()

        adapter._on_realtime_state(
            5, "connected", "", supervisor=True, loop=loop,
        )
        loop.call_soon_threadsafe.assert_called_once()
        scheduled = loop.call_soon_threadsafe.call_args[0][0]
        scheduled()
        loop.call_later.assert_called_once()
        delay, cb, guild_id, sess = loop.call_later.call_args[0]
        assert delay > 0
        assert guild_id == 5
        assert sess is session
        cb(guild_id, sess)
        session.speak_verbatim.assert_called_once_with(
            "I'm here.", interruptible=True,
        )

    def test_connected_does_not_greet_in_ears_mode(self):
        adapter = _bare_discord_adapter()
        session = MagicMock()
        session._cfg = SimpleNamespace(supervisor=False)
        adapter._voice_realtime[5] = session
        loop = MagicMock()
        adapter._on_realtime_state(
            5, "connected", "", supervisor=False, loop=loop,
        )
        loop.call_later.assert_not_called()
        session.speak_verbatim.assert_not_called()

    def test_speak_verbatim_false_is_warned(self, caplog):
        adapter = _bare_discord_adapter()
        session = MagicMock()
        session._cfg = SimpleNamespace(supervisor=True)
        session.speak_verbatim.return_value = False
        adapter._maybe_speak_join_greeting(5, session)
        session.speak_verbatim.assert_called_once_with(
            "I'm here.", interruptible=True,
        )
        assert any(r.levelname == "WARNING" for r in caplog.records)


# =====================================================================
# Voice session isolation from the bound text channel
# =====================================================================

class TestVoiceSharedSession:
    """Voice turns and supervisor consults run in the bound text channel's own session (upstream
    behavior: type in #general, continue by voice); speaker-ownership predicates resolve the guild
    through that binding."""

    def test_voice_channel_source_is_the_bound_text_chat(self, bg_loop):
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key

        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        voice_source = runner._voice_channel_source(adapter, 5, 77)
        assert voice_source is not None
        assert voice_source.chat_id == "900"
        assert voice_source.platform == Platform.DISCORD
        assert voice_source.user_id == "77"
        typed = SessionSource(
            platform=Platform.DISCORD, chat_id="900", user_id="77", chat_type="channel",
        )
        assert build_session_key(voice_source) == build_session_key(typed)

    def test_voice_channel_source_requires_a_bound_text_channel(self, bg_loop):
        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        adapter._voice_text_channels = {}
        assert runner._voice_channel_source(adapter, 5, 77) is None

    def test_guild_for_chat_resolves_only_the_bound_text_chat(self, bg_loop):
        from gateway.config import Platform
        from gateway.session import SessionSource

        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        bound = SessionSource(
            platform=Platform.DISCORD, chat_id="900", user_id="77", chat_type="channel",
        )
        unrelated = SessionSource(
            platform=Platform.DISCORD, chat_id="999", user_id="77", chat_type="channel",
            guild_id="5",
        )
        assert runner._voice_realtime_guild_for_chat(bound) == 5
        # Guild-scoped but unbound chats must not resolve — that would silence classic TTS there.
        assert runner._voice_realtime_guild_for_chat(unrelated) is None

    def test_supervisor_owns_bound_text_chat(self, bg_loop):
        from gateway.config import Platform
        from gateway.session import SessionSource

        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        adapter.voice_realtime_brain = lambda gid: "supervisor" if gid == 5 else ""
        text_source = SessionSource(
            platform=Platform.DISCORD, chat_id="900", user_id="77",
            chat_type="channel",
        )
        assert runner._voice_supervisor_owns_chat(text_source) is True
        other = SessionSource(
            platform=Platform.DISCORD, chat_id="999", user_id="77",
            chat_type="channel", guild_id="5",
        )
        assert runner._voice_supervisor_owns_chat(other) is False

    def test_consult_owns_turn_in_bound_text_chat(self, bg_loop):
        from gateway.config import Platform
        from gateway.platforms.event import MessageType
        from gateway.session import SessionSource

        session = _fake_session()
        runner, adapter = _make_gateway_runner(session, bg_loop)
        runner._handle_voice_channel_function_call(
            5, "consult_hermes", "call-1", '{"task": "list the repos"}'
        )
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="900", user_id="77", chat_type="channel",
        )
        assert runner._voice_consult_owns_turn(source, MessageType.TEXT, "list the repos") is True
        assert runner._voice_consult_owns_turn(source, MessageType.VOICE, "list the repos") is True
        assert runner._voice_consult_owns_turn(source, MessageType.TEXT, "something else") is False

    def _add_secondary_profile_bot(self, runner, session, *, profile="alt", guild=7, chat=700):
        """A second profile's Discord bot (multiplex): its own adapter, binding and session."""
        from gateway.config import Platform

        alt = SimpleNamespace(
            _owner_profile=profile,
            voice_realtime_session=lambda gid: session if gid == guild else None,
            voice_realtime_brain=lambda gid: "supervisor" if gid == guild else "",
            _voice_sources={}, _voice_realtime_last_speaker={guild: 88},
            _voice_realtime_last_transcribed={}, _voice_text_channels={guild: chat},
            _voice_clients={}, _voice_realtime={}, _active_sessions={}, _pending_messages={},
            interrupt_session_activity=AsyncMock(),
        )
        runner._profile_adapters = {profile: {Platform.DISCORD: alt}}
        return alt

    def test_speaker_ownership_resolves_the_source_profiles_own_bot(self, bg_loop):
        """Under multiplexing the bound chat / supervisor brain live on the SOURCE profile's
        adapter. Resolving through the default adapter would find no binding and let classic TTS
        read the reply over the secondary bot's spoken summary."""
        from gateway.config import Platform
        from gateway.session import SessionSource

        runner, default_adapter = _make_gateway_runner(_fake_session(), bg_loop)
        default_adapter.voice_realtime_brain = lambda gid: ""
        self._add_secondary_profile_bot(runner, _fake_session(), guild=7, chat=700)

        alt_source = SessionSource(
            platform=Platform.DISCORD, chat_id="700", user_id="88", chat_type="channel",
            profile="alt",
        )
        assert runner._voice_realtime_guild_for_chat(alt_source) == 7
        assert runner._voice_supervisor_owns_chat(alt_source) is True
        # The default profile's bot is not bound to #700 — same chat id, no ownership.
        default_source = SessionSource(
            platform=Platform.DISCORD, chat_id="700", user_id="88", chat_type="channel",
        )
        assert runner._voice_realtime_guild_for_chat(default_source) is None
        assert runner._voice_supervisor_owns_chat(default_source) is False

    def test_two_profiles_in_one_guild_get_separate_controllers(self, bg_loop):
        """Controllers are keyed per (profile, guild): two bots in the same guild must never share
        one consult state, and a secondary bot's consult completes on ITS adapter's events."""
        from gateway.config import Platform
        from gateway.platforms.event import MessageType
        from gateway.session import SessionSource

        runner, default_adapter = _make_gateway_runner(_fake_session(), bg_loop)
        alt = self._add_secondary_profile_bot(runner, _fake_session(), guild=5, chat=700)

        default_ctrl = runner._ensure_voice_realtime_controller(5, adapter=default_adapter)
        alt_ctrl = runner._ensure_voice_realtime_controller(5, adapter=alt)
        assert default_ctrl is not None and alt_ctrl is not None
        assert default_ctrl is not alt_ctrl
        assert len(runner._voice_realtime_controllers) == 2

        runner._handle_voice_channel_function_call(
            5, "consult_hermes", "call-alt", '{"task": "check the logs"}', adapter=alt,
        )
        alt_source = SessionSource(
            platform=Platform.DISCORD, chat_id="700", user_id="88", chat_type="channel",
            profile="alt",
        )
        default_source = SessionSource(
            platform=Platform.DISCORD, chat_id="900", user_id="77", chat_type="channel",
        )
        assert runner._voice_consult_owns_turn(alt_source, MessageType.TEXT, "check the logs") is True
        assert runner._voice_consult_owns_turn(default_source, MessageType.TEXT, "check the logs") is False


class TestFeedRealtimePcm:
    def _receiver(self, drained):
        receiver = MagicMock()
        receiver.drain_pending.return_value = drained
        return receiver

    def test_all_speakers_of_a_drain_go_to_the_bridge_as_one_call(self):
        adapter = _bare_discord_adapter()
        bridge = MagicMock()
        adapter._voice_realtime_mics[5] = bridge
        a, b = b"\x01\x00" * 6, b"\x02\x00" * 3
        adapter._feed_realtime_pcm(5, self._receiver([(11, a), (22, b)]))
        bridge.feed.assert_called_once_with([a, b])

    def test_attribution_follows_the_speaker_with_the_most_audio(self):
        adapter = _bare_discord_adapter()
        adapter._voice_realtime_mics[5] = MagicMock()
        adapter._feed_realtime_pcm(
            5, self._receiver([(11, b"\x01\x00" * 2), (22, b"\x02\x00" * 9)]),
        )
        assert adapter._voice_realtime_last_speaker[5] == 22

    def test_disallowed_users_never_reach_the_bridge(self):
        adapter = _bare_discord_adapter()
        bridge = MagicMock()
        adapter._voice_realtime_mics[5] = bridge
        adapter._is_allowed_user = MagicMock(side_effect=lambda uid, **kw: uid == "11")
        adapter._feed_realtime_pcm(
            5, self._receiver([(11, b"\x01\x00" * 6), (99, b"\x09\x00" * 6)]),
        )
        bridge.feed.assert_called_once_with([b"\x01\x00" * 6])
        assert adapter._voice_realtime_last_speaker[5] == 11

    def test_nothing_allowed_means_nothing_fed(self):
        adapter = _bare_discord_adapter()
        bridge = MagicMock()
        adapter._voice_realtime_mics[5] = bridge
        adapter._is_allowed_user = MagicMock(return_value=False)
        adapter._feed_realtime_pcm(5, self._receiver([(99, b"\x09\x00" * 6)]))
        bridge.feed.assert_not_called()
        assert 5 not in adapter._voice_realtime_last_speaker


class TestWakeConsultGate:
    def _session(self, adapter):
        session = MagicMock()
        session.alive = True
        session._cfg = SimpleNamespace(require_wake_name="auto", supervisor=True)
        adapter._voice_realtime[5] = session
        adapter._voice_human_count = MagicMock(return_value=2)
        adapter._voice_function_call_callback = MagicMock()
        return session

    def test_consult_rejected_on_fresh_wake_miss(self):
        adapter = _bare_discord_adapter()
        session = self._session(adapter)
        adapter._voice_wake_last = {5: (False, __import__("time").monotonic(), "chatter")}
        adapter._dispatch_or_gate_function_call(
            5, "consult_hermes", "c1", '{"task":"pwd"}', loop=MagicMock(),
        )
        adapter._voice_function_call_callback.assert_not_called()
        session.send_function_output.assert_called_once()
        kwargs = session.send_function_output.call_args
        assert kwargs.kwargs.get("respond") is False
        session.cancel_response.assert_called_once()

    def test_consult_allowed_on_fresh_wake_hit(self):
        adapter = _bare_discord_adapter()
        self._session(adapter)
        adapter._voice_wake_last = {5: (True, __import__("time").monotonic(), "pwd")}
        adapter._dispatch_or_gate_function_call(
            5, "consult_hermes", "c1", '{"task":"pwd"}', loop=MagicMock(),
        )
        adapter._voice_function_call_callback.assert_called_once_with(
            5, "consult_hermes", "c1", '{"task":"pwd"}',
        )

    def test_consult_allowed_when_solo(self):
        adapter = _bare_discord_adapter()
        self._session(adapter)
        adapter._voice_human_count = MagicMock(return_value=1)
        adapter._dispatch_or_gate_function_call(
            5, "consult_hermes", "c1", '{"task":"pwd"}', loop=MagicMock(),
        )
        adapter._voice_function_call_callback.assert_called_once()

    def test_steer_bypasses_gate(self):
        adapter = _bare_discord_adapter()
        self._session(adapter)
        adapter._voice_wake_last = {5: (False, __import__("time").monotonic(), "chatter")}
        adapter._dispatch_or_gate_function_call(
            5, "steer_hermes", "c1", '{"instruction":"stop"}', loop=MagicMock(),
        )
        adapter._voice_function_call_callback.assert_called_once()
