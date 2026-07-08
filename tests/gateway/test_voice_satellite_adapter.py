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


def test_validate_config_requires_satellites():
    assert _mod.validate_config(PlatformConfig(extra={"satellites": [{}]})) is True
    assert _mod.validate_config(PlatformConfig(extra={})) is False


def test_apply_yaml_config_translates_section():
    yaml_cfg = {
        "voice_satellite": {
            "satellites": [{"name": "kitchen", "host": "10.0.0.5", "port": 10700}],
            "tts_sample_rate": 16000,
        }
    }
    platform_cfg = {}
    extra = _mod._apply_yaml_config(yaml_cfg, platform_cfg)
    assert extra["satellites"][0]["host"] == "10.0.0.5"
    assert extra["tts_sample_rate"] == 16000
    # This hook only seeds `extra`; enablement comes from the section's own
    # `enabled: true` key or via a `platforms: voice_satellite: enabled: true`
    # entry (see test_voice_satellite_config.py::test_gateway_config_chain_enables_platform).
    assert "enabled" not in platform_cfg
    # absent/empty section -> None, no enablement
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
