"""Regression tests for mixer speech timeout scoping.

A timed-out ``play_in_voice_channel`` call used to call ``VoiceMixer.stop_speech()``
with no argument, which drops **every** admitted speech clip.  Because ordinary
replies, acks and follow-ups all share one mixer per guild, one slow clip timing
out silenced unrelated, still-playing audio.

The fix gives ``play_speech`` a return handle and lets ``stop_speech`` remove a
single child.  The timeout path now stops only the clip it started; an explicit
``stop_speech()`` with no argument still clears everything on purpose
(barge-in / stop-all semantics).
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import plugins.platforms.discord.adapter as discord_adapter

# voice_mixer lives inside the discord plugin package dir; import by path the
# same way the adapter does.
_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import voice_mixer as vm


def _make_adapter(fx_cfg=None):
    from gateway.config import Platform, PlatformConfig

    from plugins.platforms.discord.adapter import DiscordAdapter

    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._ambient_pcm_cache = None
    adapter._voice_fx_cfg = fx_cfg if fx_cfg is not None else {
        "enabled": True, "ambient_enabled": False, "ambient_path": "",
        "ambient_gain": 0.18, "duck_gain": 0.06, "speech_gain": 1.0,
        "ack_enabled": True, "ack_phrases": ["One moment."],
    }
    return adapter


def _const_pcm(seconds: float, amplitude: int) -> bytes:
    """Frame-aligned constant-amplitude PCM, so reading frames needs no clock."""
    np = pytest.importorskip("numpy")
    frames = max(1, round(seconds * 1000 / vm.FRAME_LENGTH_MS))
    samples = np.full(frames * vm.SAMPLES_PER_FRAME * vm.CHANNELS, amplitude, dtype=np.int16)
    return samples.tobytes()


def _peak(frame: bytes) -> int:
    np = pytest.importorskip("numpy")
    return int(np.max(np.abs(np.frombuffer(frame, dtype=np.int16))))


class TestTimeoutStopsOnlyItsOwnClip:
    @pytest.mark.asyncio
    async def test_timeout_keeps_an_unrelated_clip_playing(self):
        """A clip that times out must not silence a different in-flight clip."""
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        adapter._reset_voice_timeout = MagicMock()
        adapter._playback_timeout_for_audio = AsyncMock(return_value=0.5)

        unrelated = _const_pcm(seconds=2.0, amplitude=2000)
        timed_out = _const_pcm(seconds=5.0, amplitude=9000)
        mixer.play_speech(unrelated)

        # Deterministic clock: wait_start reads 0.0, the first poll reads 1.0 and
        # exceeds the 0.5s timeout, so the loop breaks before any sleep.
        fake_time = MagicMock()
        fake_time.monotonic.side_effect = [0.0, 1.0, 2.0]
        with patch.object(vm, "decode_to_pcm", return_value=timed_out), \
                patch.object(discord_adapter, "time", fake_time):
            ok = await adapter.play_in_voice_channel(111, "/tmp/timed-out.wav")

        assert ok is True
        # The unrelated clip is still admitted and still audible; only the
        # timed-out clip (amplitude 9000) was dropped.
        assert mixer.speech_active is True
        peaks = [_peak(mixer.read()) for _ in range(5)]  # past the 40ms fade-in
        assert peaks, "expected frames"
        assert max(peaks) > 0
        assert max(peaks) < 4000

    @pytest.mark.asyncio
    async def test_timeout_stops_the_handle_it_started_and_returns_true(self):
        """The timeout path passes the handle returned by play_speech (no global stop)."""
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc
        adapter._reset_voice_timeout = MagicMock()
        adapter._playback_timeout_for_audio = AsyncMock(return_value=0.5)

        own_clip = object()

        class _Mixer:
            def __init__(self):
                self.play_speech = MagicMock(return_value=own_clip)
                self.stop_speech = MagicMock()

            @property
            def speech_active(self):
                return True

        mixer = _Mixer()
        adapter._voice_mixers[111] = mixer

        fake_time = MagicMock()
        fake_time.monotonic.side_effect = [0.0, 1.0]
        with patch.object(vm, "decode_to_pcm", return_value=_const_pcm(1.0, 8000)), \
                patch.object(discord_adapter, "time", fake_time):
            ok = await adapter.play_in_voice_channel(111, "/tmp/x.wav")

        assert ok is True
        mixer.stop_speech.assert_called_once_with(own_clip)


class TestMixerStopScope:
    def test_stopping_one_child_leaves_the_other_audible(self):
        mixer = vm.VoiceMixer()
        quiet = mixer.play_speech(_const_pcm(seconds=1.0, amplitude=2000))
        loud = mixer.play_speech(_const_pcm(seconds=1.0, amplitude=9000))
        assert quiet is not None and loud is not None

        mixer.stop_speech(loud)

        assert mixer.speech_active is True
        peaks = [_peak(mixer.read()) for _ in range(5)]
        assert max(peaks) < 4000

    def test_stop_all_without_a_child_still_clears_everything(self):
        mixer = vm.VoiceMixer()
        mixer.play_speech(_const_pcm(seconds=1.0, amplitude=2000))
        mixer.play_speech(_const_pcm(seconds=1.0, amplitude=9000))

        mixer.stop_speech()

        assert mixer.speech_active is False
        assert all(mixer.read() == vm.SILENCE_FRAME for _ in range(3))

    def test_empty_clip_has_no_handle(self):
        mixer = vm.VoiceMixer()
        assert mixer.play_speech(b"") is None
        assert mixer.speech_active is False
