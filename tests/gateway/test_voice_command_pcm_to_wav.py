"""Tests for ``VoiceReceiver.pcm_to_wav`` — PCM streams to ffmpeg over stdin.

Extracted byte-verbatim from ``tests/gateway/test_voice_command.py`` (2K-law
shard, issue #79996; see also #78647).  The class body below is unchanged.
"""

import pytest
from unittest.mock import patch


class TestPcmToWav:
    """pcm_to_wav streams PCM through ffmpeg's stdin, not a temp file."""

    def test_pcm_is_piped_to_stdin_not_staged_on_disk(self, tmp_path):
        from plugins.platforms.discord.adapter import VoiceReceiver

        out = tmp_path / "out.wav"
        with patch("plugins.platforms.discord.adapter.subprocess.run") as run:
            VoiceReceiver.pcm_to_wav(b"\x00\x01" * 16, str(out))

        args, kwargs = run.call_args
        cmd = args[0]
        assert kwargs["input"] == b"\x00\x01" * 16, "PCM must be fed via stdin"
        assert "pipe:0" in cmd, "ffmpeg must read the PCM from stdin"
        assert cmd[-1] == str(out), (
            "the WAV must be written to the real path; ffmpeg cannot seek on a "
            "pipe, so a piped WAV gets placeholder RIFF/data sizes"
        )
        assert not any(str(a).endswith(".pcm") for a in cmd), (
            "no temp .pcm file should be staged"
        )

    @pytest.mark.skipif(
        __import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed",
    )
    def test_output_wav_header_reports_true_length(self, tmp_path):
        """A piped-stdout WAV reports 0xFFFFFFFF sizes; the written file must not."""
        import math
        import struct
        import wave

        from plugins.platforms.discord.adapter import VoiceReceiver

        frames = 48000  # 1s @ 48kHz stereo
        pcm = b"".join(
            struct.pack("<hh", v, v)
            for v in (
                int(20000 * math.sin(2 * math.pi * 440 * i / 48000))
                for i in range(frames)
            )
        )
        out = tmp_path / "out.wav"
        VoiceReceiver.pcm_to_wav(pcm, str(out))

        with wave.open(str(out)) as w:
            assert w.getnchannels() == 1
            assert w.getframerate() == 16000
            # 48kHz -> 16kHz is a 3x decimation of a 1s clip.
            assert w.getnframes() == 16000
