"""Tests for the Discord continuous voice mixer (ambient + ducked speech)
and the verbal-ack-before-tool-calls hook.

The mixer (plugins/platforms/discord/voice_mixer.py) is pure-PCM and has no
discord.py dependency, so its core is tested directly.  The adapter
integration (install on join, play routing, ack) is tested with the standard
``object.__new__(DiscordAdapter)`` helper used elsewhere in the voice suite.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# numpy ships only in the optional "audio-io" extra (not [all,dev]); the mixer
# math needs it, so skip this whole module when it isn't installed.
np = pytest.importorskip("numpy")

# voice_mixer lives inside the discord plugin package dir; import by path the
# same way the adapter does.
_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import voice_mixer as vm


# =====================================================================
# Pure mixer unit tests
# =====================================================================

class TestVoiceMixerCore:

    def test_empty_mixer_returns_silence_frames(self):
        mx = vm.VoiceMixer()
        for _ in range(5):
            frame = mx.read()
            assert len(frame) == vm.FRAME_SIZE
            assert frame == vm.SILENCE_FRAME

    def test_is_opus_false(self):
        # discord.py sends raw PCM when is_opus() is False.
        assert vm.VoiceMixer().is_opus() is False

    def test_ambient_loops_and_is_quiet(self):
        mx = vm.VoiceMixer(ambient_gain=0.2)
        amb = vm.synth_ambient_pcm(seconds=0.5)
        assert len(amb) % vm.FRAME_SIZE == 0  # frame-aligned for seamless loop
        mx.set_ambient(amb)
        peaks = [int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16))))
                 for _ in range(100)]  # 2s >> 0.5s loop
        # Produces audio after the fade-in and stays under the configured gain.
        assert any(p > 0 for p in peaks[10:])
        assert max(peaks) < int(32767 * 0.5)


# =====================================================================
# Adapter integration
# =====================================================================

def _make_adapter(fx_cfg=None):
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import Platform, PlatformConfig
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
        "enabled": True, "ambient_enabled": True, "ambient_path": "",
        "ambient_gain": 0.18, "duck_gain": 0.06, "speech_gain": 1.0,
        "ack_enabled": True, "ack_phrases": ["One moment."],
    }
    return adapter




class TestPlayInVoiceChannelMixerPath:
    @pytest.mark.asyncio
    async def test_routes_through_mixer_when_present(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc

        # speech_active returns True once (so play_speech is observed) then
        # False so the wait loop exits promptly.
        class _Mixer:
            def __init__(self):
                self._polls = 0
                self.play_speech = MagicMock()

            @property
            def speech_active(self):
                self._polls += 1
                return self._polls <= 1

        mixer = _Mixer()
        adapter._voice_mixers[111] = mixer
        adapter._reset_voice_timeout = MagicMock()

        fake_pcm = b"\x00" * vm.FRAME_SIZE
        with patch.object(vm, "decode_to_pcm", return_value=fake_pcm):
            ok = await adapter.play_in_voice_channel(111, "/tmp/x.mp3")
        assert ok is True
        mixer.play_speech.assert_called_once()
        adapter._reset_voice_timeout.assert_called_once_with(111)
        # Legacy path must NOT have been used.
        vc.play.assert_not_called()


class TestLeadSilence:
    """Warm-up lead silence prepended to speech so the first word isn't clipped
    (issue #66827)."""

    def test_bytes_empty_when_unset(self):
        adapter = _make_adapter()  # default cfg has no lead_silence_ms
        assert adapter._lead_silence_bytes() == b""


    def test_bytes_length_matches_ms(self):
        adapter = _make_adapter({"lead_silence_ms": 200})
        lead = adapter._lead_silence_bytes()
        assert lead == b"\x00" * (vm.BYTES_PER_MS * 200)
        assert len(lead) == 200 * 192  # 48kHz stereo s16 -> 192 bytes/ms


class TestPlayAckInVoice:
    @pytest.mark.asyncio
    async def test_noop_when_ack_disabled(self):
        adapter = _make_adapter({"ack_enabled": False})
        adapter._voice_mixers[111] = MagicMock()
        assert await adapter.play_ack_in_voice(111) is False




class TestVoiceFxNumpyGate:
    """voice_fx needs numpy (the optional ``audio-io`` extra), but the ``discord`` extra does not
    install it.  The mixer imports numpy lazily, so the failure lands on discord.py's sender
    thread *after* ``vc.play()`` succeeded: the stream dies, ``speech_active`` never clears, and
    every reply burned the full playback timeout in silence.  The adapter must notice up front
    and use the one-shot path instead."""

    def test_config_disables_mixer_when_numpy_missing(self):
        from plugins.platforms.discord import adapter as adapter_module
        adapter = _make_adapter()
        cfg = {"discord": {"voice_fx": {"enabled": True}}}
        with patch.object(adapter_module, "_numpy_available", return_value=False), \
             patch("hermes_cli.config.read_raw_config", return_value=cfg):
            loaded = adapter._load_voice_fx_config()
        assert loaded["enabled"] is False

    def test_config_keeps_mixer_when_numpy_present(self):
        from plugins.platforms.discord import adapter as adapter_module
        adapter = _make_adapter()
        cfg = {"discord": {"voice_fx": {"enabled": True}}}
        with patch.object(adapter_module, "_numpy_available", return_value=True), \
             patch("hermes_cli.config.read_raw_config", return_value=cfg):
            loaded = adapter._load_voice_fx_config()
        assert loaded["enabled"] is True

    @pytest.mark.asyncio
    async def test_install_skips_mixer_when_numpy_missing(self):
        from plugins.platforms.discord import adapter as adapter_module
        adapter = _make_adapter()
        adapter._get_ambient_pcm = MagicMock(return_value=None)
        vc = MagicMock()
        vc.is_playing.return_value = False
        with patch.object(adapter_module, "_numpy_available", return_value=False):
            await adapter._install_voice_mixer(42, vc)
        vc.play.assert_not_called()
        assert 42 not in adapter._voice_mixers

    @pytest.mark.asyncio
    async def test_stream_error_retires_mixer(self):
        adapter = _make_adapter()
        adapter._get_ambient_pcm = MagicMock(return_value=None)
        vc = MagicMock()
        vc.is_playing.return_value = False
        await adapter._install_voice_mixer(7, vc)
        assert 7 in adapter._voice_mixers
        after = vc.play.call_args.kwargs["after"]
        after(RuntimeError("No module named 'numpy'"))
        assert 7 not in adapter._voice_mixers


class TestDeadMixerFallsBackToOneShot:
    @pytest.mark.asyncio
    async def test_mixer_retired_mid_playback_uses_one_shot(self):
        from plugins.platforms.discord import adapter as adapter_module
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        vc.is_playing.return_value = False
        adapter._voice_clients[3] = vc
        adapter._reset_voice_timeout = MagicMock()

        class _DeadMixer:
            """Reports speech forever but is unregistered on first poll — what a stream error looks like."""

            def __init__(self, owner):
                self._owner = owner
                self.play_speech = MagicMock()

            @property
            def speech_active(self):
                self._owner._voice_mixers.pop(3, None)
                return True

        adapter._voice_mixers[3] = _DeadMixer(adapter)

        played = {}

        class _FakeSource:
            def __init__(self, *a, **k):
                pass

        def _play(source, after=None):
            played["source"] = source
            if after:
                after(None)

        vc.play.side_effect = _play

        async def _timeout(_path):
            return 5.0

        adapter._playback_timeout_for_audio = _timeout
        with patch.object(vm, "decode_to_pcm", return_value=b"\x00" * vm.FRAME_SIZE), \
             patch.object(adapter_module, "resolve_ffmpeg_executable", return_value="/usr/bin/ffmpeg"), \
             patch.object(adapter_module.discord, "FFmpegPCMAudio", _FakeSource), \
             patch.object(adapter_module.discord, "PCMVolumeTransformer", lambda s, volume=1.0: s):
            ok = await adapter.play_in_voice_channel(3, "/tmp/x.mp3")
        assert ok is True
        assert "source" in played, "one-shot playback must run once the mixer is retired"
