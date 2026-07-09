"""VoiceSatelliteAdapter integration tests (fake satellite, stubbed STT/TTS)."""

import asyncio
import json
import struct
import math

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from tests.gateway._fake_wyoming_satellite import FakeSatellite
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_mod = load_plugin_adapter("voice_satellite")

RATE = 16000


def make_pcm(seconds, amplitude, rate=RATE):
    n = int(seconds * rate)
    if amplitude == 0:
        return b"\x00\x00" * n
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(n)
    )


def make_config(port, **extra_overrides):
    extra = {
        "satellites": [{"name": "kitchen", "host": "127.0.0.1", "port": port}],
        "endpointing": {
            "silence_threshold": 200,
            "silence_duration": 0.3,
            "min_speech_seconds": 0.2,
            "max_utterance_seconds": 20.0,
        },
    }
    extra.update(extra_overrides)
    return PlatformConfig(enabled=True, extra=extra)


def test_validate_config_requires_connectable_satellites():
    ok = {"satellites": [{"name": "kitchen", "host": "192.168.1.40", "port": 10700}]}
    assert _mod.validate_config(PlatformConfig(extra=ok)) is True
    # port may be omitted (defaults to 10700 at connect time)
    assert _mod.validate_config(
        PlatformConfig(extra={"satellites": [{"host": "pi.local"}]})
    ) is True
    assert _mod.validate_config(PlatformConfig(extra={})) is False
    # entries missing a host would spin the reconnect loop against ""
    assert _mod.validate_config(PlatformConfig(extra={"satellites": [{}]})) is False
    assert _mod.validate_config(
        PlatformConfig(extra={"satellites": [{"host": "  "}]})
    ) is False
    assert _mod.validate_config(
        PlatformConfig(extra={"satellites": [{"host": "pi.local", "port": "not-a-port"}]})
    ) is False


def test_apply_yaml_config_translates_section():
    # The loader binds platform_cfg to the user's block wherever it lives
    # (top-level section OR nested platforms map); the hook must read from
    # the argument, not from yaml_cfg (gateway/config.py ~1261-1273).
    section = {
        "satellites": [{"name": "kitchen", "host": "10.0.0.5", "port": 10700}],
        "tts_sample_rate": 16000,
    }
    extra = _mod._apply_yaml_config({"voice_satellite": section}, section)
    assert extra["satellites"][0]["host"] == "10.0.0.5"
    assert extra["tts_sample_rate"] == 16000
    # nested-only config: top-level key absent, block passed as platform_cfg
    nested = _mod._apply_yaml_config({}, section)
    assert nested["satellites"][0]["host"] == "10.0.0.5"
    # This hook only seeds `extra`; it must not write enablement keys into
    # the user's block (see test_voice_satellite_config.py).
    assert "enabled" not in section
    # absent/empty block -> None, no enablement
    assert _mod._apply_yaml_config({}, {}) is None


def test_register_declares_platform_entry():
    calls = {}

    class Ctx:
        def register_platform(self, **kwargs):
            calls.update(kwargs)

    _mod.register(Ctx())
    assert calls["name"] == "voice_satellite"
    assert callable(calls["adapter_factory"])
    assert callable(calls["check_fn"])
    assert calls["apply_yaml_config_fn"] is _mod._apply_yaml_config
    assert "voice" in calls["platform_hint"].lower() or "aloud" in calls["platform_hint"].lower()
    # is_connected must reflect config presence, NOT dependency presence:
    # the setup wizard probes it with a bare PlatformConfig and would
    # otherwise report the platform "configured" just because wyoming is
    # importable (check_fn's answer).
    assert callable(calls["is_connected"])
    assert calls["is_connected"](PlatformConfig(enabled=True)) is False
    assert calls["is_connected"](
        PlatformConfig(
            enabled=True,
            extra={"satellites": [{"name": "kitchen", "host": "10.0.0.5", "port": 10700}]},
        )
    ) is True


@pytest_asyncio.fixture
async def rig(monkeypatch, tmp_path):
    """Fake satellite + connected adapter with stubbed STT/TTS."""
    sat = FakeSatellite()
    await sat.start()

    import tools.transcription_tools as tt
    import tools.tts_tool as tts

    monkeypatch.setattr(
        tt, "transcribe_audio",
        lambda path, model=None: {"success": True, "transcript": "what time is it"},
    )
    reply_wav = tmp_path / "reply.wav"
    reply_wav.write_bytes(b"RIFFfake")
    monkeypatch.setattr(
        tts, "text_to_speech_tool",
        lambda text, output_path=None: json.dumps(
            {"success": True, "file_path": str(reply_wav)}
        ),
    )
    monkeypatch.setattr(tts, "check_tts_requirements", lambda: True)

    audio_mod = _mod._import_sibling("audio")
    monkeypatch.setattr(
        audio_mod, "transcode_to_pcm", lambda path, rate=22050: b"\x05\x00" * 1000
    )

    adapter = _mod.VoiceSatelliteAdapter(make_config(sat.port))
    dispatched = []

    async def handler(event):
        dispatched.append(event)
        # Simulate the gateway reply path: base auto-TTS then play_tts.
        await adapter.play_tts(
            chat_id=event.source.chat_id, audio_path=str(reply_wav)
        )

    adapter.set_message_handler(handler)
    assert await adapter.connect() is True
    await asyncio.wait_for(sat.run_satellite_received.wait(), timeout=5)
    yield sat, adapter, dispatched
    await adapter.disconnect()
    await sat.stop()


@pytest_asyncio.fixture
async def rig_base_reply(monkeypatch, tmp_path):
    """Fake satellite + adapter whose handler emulates base.py's reply path.

    base.py runs auto-TTS synth -> play_tts(reply audio) -> send(text). This
    fixture's handler reproduces that ordering so tests can assert the reply is
    spoken exactly once.
    """
    sat = FakeSatellite()
    await sat.start()

    import tools.transcription_tools as tt
    import tools.tts_tool as tts

    monkeypatch.setattr(
        tt, "transcribe_audio",
        lambda path, model=None: {"success": True, "transcript": "what time is it"},
    )
    reply_wav = tmp_path / "reply.wav"
    reply_wav.write_bytes(b"RIFFfake")
    monkeypatch.setattr(
        tts, "text_to_speech_tool",
        lambda text, output_path=None: json.dumps(
            {"success": True, "file_path": str(reply_wav)}
        ),
    )
    monkeypatch.setattr(tts, "check_tts_requirements", lambda: True)

    audio_mod = _mod._import_sibling("audio")
    monkeypatch.setattr(
        audio_mod, "transcode_to_pcm", lambda path, rate=22050: b"\x05\x00" * 1000
    )

    adapter = _mod.VoiceSatelliteAdapter(make_config(sat.port))
    dispatched = []

    async def handler(event):
        # Emulate base.py: play the auto-TTS audio, THEN send the text portion.
        await adapter.play_tts(
            chat_id=event.source.chat_id, audio_path=str(reply_wav)
        )
        event.send_result = await adapter.send(
            event.source.chat_id, "the text reply"
        )
        dispatched.append(event)

    adapter.set_message_handler(handler)
    assert await adapter.connect() is True
    await asyncio.wait_for(sat.run_satellite_received.wait(), timeout=5)
    yield sat, adapter, dispatched, reply_wav
    await adapter.disconnect()
    await sat.stop()


@pytest.mark.asyncio
async def test_round_trip_utterance_to_spoken_reply(rig):
    sat, adapter, dispatched = rig
    utterance = make_pcm(0.5, 3000) + make_pcm(0.6, 0)
    await sat.wake_and_stream(utterance)

    await asyncio.wait_for(sat.tts_done.wait(), timeout=10)
    assert len(dispatched) == 1
    event = dispatched[0]
    assert event.text == "what time is it"
    assert event.message_type.value == "voice"
    assert event.source.chat_id == "kitchen"
    assert sat.transcript_received.is_set()  # streaming ended before reply
    assert bytes(sat.play_buffer) == b"\x05\x00" * 1000


@pytest.mark.asyncio
async def test_base_reply_sequence_speaks_reply_exactly_once(rig_base_reply):
    """Emulate base.py exactly: play_tts(reply audio) THEN send(text reply).

    The base reply path plays the auto-TTS audio via play_tts and then calls
    send() with the text portion. A voice-only surface must speak the reply
    exactly ONCE — the send() text follow-up must not re-synthesize/re-speak it.
    """
    sat, adapter, dispatched, reply_wav = rig_base_reply
    utterance = make_pcm(0.5, 3000) + make_pcm(0.6, 0)
    await sat.wake_and_stream(utterance)

    await asyncio.wait_for(sat.tts_done.wait(), timeout=10)
    await asyncio.sleep(0.25)  # let any wrongly-triggered second playback arrive
    assert len(dispatched) == 1
    assert bytes(sat.play_buffer) == b"\x05\x00" * 1000  # exactly ONE copy
    assert dispatched[0].send_result.success is True


@pytest.mark.asyncio
async def test_send_speaks_when_idle_and_noops_mid_turn(rig):
    sat, adapter, dispatched = rig
    # idle announce: speaks through the satellite
    result = await adapter.send("kitchen", "Backup finished.")
    assert result.success is True
    await asyncio.wait_for(sat.tts_done.wait(), timeout=10)
    assert len(sat.play_buffer) > 0

    # mid-turn: text reply is a silent no-op success (play_tts owns audio)
    tm = _mod._import_sibling("turn_machine")
    machine = adapter._machines["kitchen"]
    machine.phase = tm.TurnPhase.THINKING
    sat.play_buffer.clear()
    result = await adapter.send("kitchen", "text reply body")
    assert result.success is True
    await asyncio.sleep(0.2)  # would let any wrongly-started playback arrive
    assert len(sat.play_buffer) == 0
    machine.to_idle()


@pytest.mark.asyncio
async def test_wake_during_turn_sends_rejection_transcript(rig):
    """A wake that arrives mid-turn is rejected with an empty transcript so the
    satellite returns to wake mode instead of streaming mic audio forever."""
    sat, adapter, dispatched = rig
    tm = _mod._import_sibling("turn_machine")
    machine = adapter._machines["kitchen"]
    machine.phase = tm.TurnPhase.THINKING  # a turn is already in flight
    sat.transcript_received.clear()

    await sat.wake_only()

    # The empty rejection transcript ends the satellite's would-be streaming...
    await asyncio.wait_for(sat.transcript_received.wait(), timeout=5)
    # ...without disturbing the in-flight turn.
    assert machine.phase is tm.TurnPhase.THINKING
    machine.to_idle()


def test_reply_timeout_config_wires_through():
    """reply_timeout_seconds in config.extra reaches the adapter knob."""
    adapter = _mod.VoiceSatelliteAdapter(make_config(0, reply_timeout_seconds=0.3))
    assert adapter._reply_timeout == 0.3


@pytest.mark.asyncio
async def test_reply_watchdog_frees_stuck_thinking_machine(rig):
    """A turn whose agent reply never produces audio (no play_tts / send) is
    freed by the watchdog so the next wake works."""
    sat, adapter, dispatched = rig
    tm = _mod._import_sibling("turn_machine")
    adapter._reply_timeout = 0.3

    async def silent_handler(event):
        dispatched.append(event)  # agent produced no spoken/text reply

    adapter.set_message_handler(silent_handler)
    await sat.wake_and_stream(make_pcm(0.5, 3000) + make_pcm(0.6, 0))
    # transcript sent, machine parked in THINKING with the watchdog armed
    await asyncio.wait_for(sat.transcript_received.wait(), timeout=5)
    assert adapter._machines["kitchen"].phase is tm.TurnPhase.THINKING

    await asyncio.sleep(0.8)
    assert adapter._machines["kitchen"].phase is tm.TurnPhase.IDLE


def test_satellite_authorization_is_upstream():
    """The adapter declares upstream authorization: config listing is the grant."""
    adapter = _mod.VoiceSatelliteAdapter(make_config(0))
    assert adapter.authorization_is_upstream is True


def test_inbound_satellite_message_passes_authorization(monkeypatch):
    """An inbound satellite message is authorized with NO env allowlist set.

    Mirrors tests/gateway/test_relay_upstream_authz.py: a bare GatewayRunner
    consults the adapter's ``authorization_is_upstream`` flag (honored at
    gateway/authz_mixin.py:311) rather than default-denying for a missing
    ``VOICE_SATELLITE_ALLOWED_USERS`` allowlist that satellites don't have.
    """
    from unittest.mock import MagicMock

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)

    platform = Platform("voice_satellite")
    runner = object.__new__(GatewayRunner)
    runner.adapters = {platform: _mod.VoiceSatelliteAdapter(make_config(0))}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    src = SessionSource(
        platform=platform,
        user_id="kitchen",
        chat_id="kitchen",
        user_name="kitchen",
        chat_type="dm",
    )
    assert runner._is_user_authorized(src) is True


@pytest.mark.asyncio
async def test_stt_failure_recovers_turn(rig, monkeypatch):
    sat, adapter, dispatched = rig
    import tools.transcription_tools as tt

    def boom(path, model=None):
        raise RuntimeError("stt exploded")

    monkeypatch.setattr(tt, "transcribe_audio", boom)
    await sat.wake_and_stream(make_pcm(0.5, 3000) + make_pcm(0.6, 0))
    # failure path ends satellite streaming with an empty transcript
    await asyncio.wait_for(sat.transcript_received.wait(), timeout=5)
    tm = _mod._import_sibling("turn_machine")
    assert adapter._machines["kitchen"].phase is tm.TurnPhase.IDLE
    assert dispatched == []
