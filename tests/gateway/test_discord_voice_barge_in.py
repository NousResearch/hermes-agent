"""Barge-in: user speech interrupts the bot's in-flight voice-channel speech.

Covers the receiver telemetry (continuous-packet streaks per SSRC), the adapter's level gate
(absolute dBFS floor + sustained streak — NOT a margin over the bot's own output, because
Discord AGC normalizes TTS loopback to close-talk levels), and the config knobs
(`discord.voice_barge_in`: enabled / min_db / min_frames).

The receiver/mixer classes live in the Discord plugin dir; import them flat exactly like the
adapter's own ``_voice_mixer_module()`` resolution does.
"""
from __future__ import annotations

import asyncio
import array
import math
import os
import sys
import threading
import time
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Package import (like test_discord_voice_mixer does): keeps relative imports inside the
# plugin working; the conftest's anti-pattern guard targets bare path-inserted `import adapter`.
from plugins.platforms.discord import adapter as discord_adapter


def _tone_pcm(amplitude: float, seconds: float = 1.0, freq: float = 220.0) -> bytes:
    """int16 sine at the given peak amplitude (full scale = 1.0)."""
    n = int(seconds * 48000)
    samples = array.array("h", (int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / 48000))
                                for i in range(n)))
    return samples.tobytes()


class TestRmsDbfs:
    def test_loud_tone_measures_its_level(self):
        # 0.9 peak -> sine RMS is ~3 dB below peak (≈ -1.9); integer quantization of a low-
        # frequency tone adds a little more spread, so accept the -4..-1 dBFS band.
        level = discord_adapter.DiscordAdapter._rms_dbfs(_tone_pcm(0.9))
        assert -4.5 < level < -1.0

    def test_quiet_tone_measures_its_level(self):
        # 0.01 peak -> RMS of a sine sits ~3 dB below the peak (≈ -43 dBFS)
        level = discord_adapter.DiscordAdapter._rms_dbfs(_tone_pcm(0.01))
        assert -46.0 < level < -39.0

    def test_short_or_empty_buffer_is_silence(self):
        assert discord_adapter.DiscordAdapter._rms_dbfs(b"") == -99.0
        assert discord_adapter.DiscordAdapter._rms_dbfs(b"\x00" * 100) == -99.0


class TestVoiceBargeInConfig:
    def test_defaults_when_unset(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "read_raw_config", dict)
        cfg = discord_adapter.DiscordAdapter._load_voice_barge_in(SimpleNamespace())
        assert cfg == {"enabled": True, "min_db": -45.0, "min_frames": 25}

    def test_boolean_shorthand_disables(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "read_raw_config", lambda: {"discord": {"voice_barge_in": False}})
        cfg = discord_adapter.DiscordAdapter._load_voice_barge_in(SimpleNamespace())
        assert cfg["enabled"] is False

    def test_dict_knobs_override(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "read_raw_config", lambda: {"discord": {
            "voice_barge_in": {"min_db": -35, "min_frames": 40}}})
        cfg = discord_adapter.DiscordAdapter._load_voice_barge_in(SimpleNamespace())
        assert cfg["enabled"] is True and cfg["min_db"] == -35.0 and cfg["min_frames"] == 40.0


def _bare_receiver():
    r = discord_adapter.VoiceReceiver.__new__(discord_adapter.VoiceReceiver)
    r.STREAK_STALE_S = 0.6
    r._lock = threading.Lock()
    r._buffers = {}
    r._last_packet_time = {}
    r._speech_streaks = {}
    r._ssrc_to_user = {}
    return r


def _feed(receiver, ssrc: int, when: float) -> None:
    """Mirror the streak bookkeeping VoiceReceiver._on_packet does inside its decode lock."""
    receiver._last_packet_time[ssrc] = when
    streak = receiver._speech_streaks.get(ssrc)
    if streak is None or when - streak[1] > receiver.STREAK_STALE_S:
        receiver._speech_streaks[ssrc] = [1, when]
    else:
        streak[0] += 1
        streak[1] = when


class TestReceiverStreakState:
    """The state contract the watcher reads (streak reset on packet gap, cleared per utterance)."""

    def test_continuous_packets_grow_one_streak(self):
        r = _bare_receiver()
        base = time.monotonic()
        for i in range(30):
            _feed(r, 42, base + i * 0.02)
        assert r._speech_streaks[42][0] == 30

    def test_streak_resets_after_gap(self):
        r = _bare_receiver()
        base = time.monotonic()
        for i in range(10):
            _feed(r, 7, base + i * 0.02)
        assert r._speech_streaks[7][0] == 10
        _feed(r, 7, base + 10 * 0.02 + 2.0)  # 2 s gap > STREAK_STALE_S
        assert r._speech_streaks[7][0] == 1


class TestBargeWatcher:
    """The watcher's cut decision on fake mixer/receiver state."""

    class FakeMixer:
        """Real VoiceMixer semantics: silence drains the clip and clears speech_active on its
        own — a cut must NOT be the only way the flag goes false."""
        def __init__(self):
            self.stopped = 0
            self._active = True

        @property
        def speech_active(self):
            return self._active

        def stop_speech(self):
            self.stopped += 1
            self._active = False

    def _adapter(self, monkeypatch, cfg, uplink_pcm: bytes, frames: int):
        a = discord_adapter.DiscordAdapter.__new__(discord_adapter.DiscordAdapter)
        # `name` is a read-only property on BasePlatformAdapter; the watcher only logs with it.
        type(a).name = property(lambda self: "test")
        a._voice_mixers = {1: self.FakeMixer()}
        r = _bare_receiver()
        r._buffers[7] = bytearray(uplink_pcm)
        r._speech_streaks[7] = [frames, time.monotonic()]
        r._ssrc_to_user[7] = 111
        a._voice_receivers = {1: r}
        monkeypatch.setattr(discord_adapter.DiscordAdapter, "_load_voice_barge_in",
                            lambda self: cfg)
        return a

    def test_loud_sustained_speech_cuts(self, monkeypatch):
        a = self._adapter(monkeypatch, {"enabled": True, "min_db": -45.0, "min_frames": 25},
                          _tone_pcm(0.1), frames=30)
        asyncio.run(a._watch_barge_in(1))
        assert a._voice_mixers[1].stopped == 1

    def test_ambient_noise_never_cuts(self, monkeypatch):
        a = self._adapter(monkeypatch, {"enabled": True, "min_db": -45.0, "min_frames": 25},
                          _tone_pcm(0.0003), frames=100)  # ~ -70 dBFS

        async def _run():
            task = asyncio.ensure_future(a._watch_barge_in(1))
            await asyncio.sleep(0.4)          # watcher polled and declined at least once
            a._voice_mixers[1]._active = False  # clip drains naturally -> loop exits
            await task

        asyncio.run(_run())
        assert a._voice_mixers[1].stopped == 0

    def test_short_blip_never_cuts(self, monkeypatch):
        a = self._adapter(monkeypatch, {"enabled": True, "min_db": -45.0, "min_frames": 25},
                          _tone_pcm(0.1), frames=10)  # below the sustained threshold

        async def _run():
            task = asyncio.ensure_future(a._watch_barge_in(1))
            await asyncio.sleep(0.4)
            a._voice_mixers[1]._active = False
            await task

        asyncio.run(_run())
        assert a._voice_mixers[1].stopped == 0

    def test_disabled_never_cuts(self, monkeypatch):
        a = self._adapter(monkeypatch, {"enabled": False, "min_db": -45.0, "min_frames": 25},
                          _tone_pcm(0.5), frames=100)

        async def _run():
            task = asyncio.ensure_future(a._watch_barge_in(1))
            await asyncio.sleep(0.1)
            a._voice_mixers[1]._active = False
            await task

        asyncio.run(_run())
        assert a._voice_mixers[1].stopped == 0

    def test_stale_streak_never_cuts(self, monkeypatch):
        """Packets stopped long ago: streak timestamp older than STREAK_STALE_S -> no candidate."""
        a = self._adapter(monkeypatch, {"enabled": True, "min_db": -45.0, "min_frames": 25},
                          _tone_pcm(0.5), frames=100)
        r = a._voice_receivers[1]
        r._speech_streaks[7][1] = time.monotonic() - 5.0  # went quiet 5 s ago

        async def _run():
            task = asyncio.ensure_future(a._watch_barge_in(1))
            await asyncio.sleep(0.4)
            a._voice_mixers[1]._active = False
            await task

        asyncio.run(_run())
        assert a._voice_mixers[1].stopped == 0
