"""Tests for gateway/platforms/voice_utils.py — WeChat voice-bubble conversion."""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ogg(tmp_path: Path) -> str:
    """Create a fake OGG file (just a header byte to satisfy file existence)."""
    ogg = tmp_path / "tts_output.ogg"
    ogg.write_bytes(b"OggS\x00\x02" + b"\x00" * 100)
    return str(ogg)


@pytest.fixture
def fake_pcm_mono(tmp_path: Path) -> str:
    """Create a fake 1-second PCM file (24kHz, 16-bit, mono)."""
    sample_rate = 24000
    duration = 1
    num_samples = sample_rate * duration
    pcm = tmp_path / "audio.pcm"
    pcm.write_bytes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))
    return str(pcm)


@pytest.fixture
def fake_silk(tmp_path: Path) -> str:
    """Create a fake SILK file with valid header."""
    silk = tmp_path / "voice.silk"
    silk.write_bytes(b"\x02#!SILK_V3" + b"\x00" * 50)
    return str(silk)


# ---------------------------------------------------------------------------
# Tests: ogg_to_silk
# ---------------------------------------------------------------------------

class TestOggToSilk:
    """Test the OGG → PCM → SILK conversion pipeline."""

    @patch("gateway.platforms.voice_utils.shutil.which")
    @patch("gateway.platforms.voice_utils.pilk")
    def test_converts_ogg_to_silk(self, mock_pilk, mock_which, fake_ogg, tmp_path):
        """Happy path: ffmpeg produces PCM, pilk produces SILK."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        mock_pilk.encode = MagicMock()

        # Mock subprocess.run for ffmpeg
        with patch("gateway.platforms.voice_utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create the expected output files
            original_encode = mock_pilk.encode

            def fake_encode(pcm, silk, pcm_rate=24000):
                Path(silk).write_bytes(b"\x02#!SILK_V3" + b"\x00" * 10)

            mock_pilk.encode.side_effect = fake_encode

            from gateway.platforms.voice_utils import ogg_to_silk
            result = ogg_to_silk(fake_ogg)

        assert result is not None
        assert result.endswith(".silk")
        assert Path(result).is_file()

    @patch("gateway.platforms.voice_utils._HAS_PILK", False)
    def test_returns_none_without_pilk(self, fake_ogg):
        """Returns None when pilk is not installed."""
        from gateway.platforms.voice_utils import ogg_to_silk
        result = ogg_to_silk(fake_ogg)
        assert result is None

    @patch("gateway.platforms.voice_utils.shutil.which")
    def test_returns_none_without_ffmpeg(self, mock_which, fake_ogg):
        """Returns None when ffmpeg is not on PATH."""
        mock_which.return_value = None

        from gateway.platforms.voice_utils import ogg_to_silk
        result = ogg_to_silk(fake_ogg)
        assert result is None

    def test_returns_none_for_missing_file(self):
        """Returns None when source file does not exist."""
        from gateway.platforms.voice_utils import ogg_to_silk
        result = ogg_to_silk("/nonexistent/file.ogg")
        assert result is None

    @patch("gateway.platforms.voice_utils.shutil.which")
    @patch("gateway.platforms.voice_utils.pilk")
    def test_returns_none_on_ffmpeg_failure(self, mock_pilk, mock_which, fake_ogg):
        """Returns None when ffmpeg exits with error."""
        mock_which.return_value = "/usr/bin/ffmpeg"

        with patch("gateway.platforms.voice_utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr=b"Error: invalid input",
            )
            from gateway.platforms.voice_utils import ogg_to_silk
            result = ogg_to_silk(fake_ogg)

        assert result is None


# ---------------------------------------------------------------------------
# Tests: get_audio_duration_s
# ---------------------------------------------------------------------------

class TestGetAudioDurationS:
    """Test duration detection for various audio formats."""

    @patch("gateway.platforms.voice_utils.pilk")
    def test_silk_duration(self, mock_pilk, fake_silk):
        """SILK duration via pilk.get_duration (returns ms)."""
        mock_pilk.get_duration.return_value = 5200

        from gateway.platforms.voice_utils import get_audio_duration_s
        result = get_audio_duration_s(fake_silk)
        assert result == 5  # 5200ms → 5s

    @patch("gateway.platforms.voice_utils.shutil.which")
    def test_ogg_duration_via_ffprobe(self, mock_which, fake_ogg):
        """OGG duration via ffprobe."""
        mock_which.return_value = "/usr/bin/ffprobe"

        with patch("gateway.platforms.voice_utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="3.14\n",
            )
            from gateway.platforms.voice_utils import get_audio_duration_s
            result = get_audio_duration_s(fake_ogg)

        assert result == 3

    def test_returns_0_for_unknown(self):
        """Returns 0 for non-existent file with no ffprobe."""
        from gateway.platforms.voice_utils import get_audio_duration_s
        with patch("gateway.platforms.voice_utils.shutil.which", return_value=None):
            result = get_audio_duration_s("/nonexistent/audio.xyz")
        assert result == 0


# ---------------------------------------------------------------------------
# Tests: cleanup_silk_dir
# ---------------------------------------------------------------------------

class TestCleanupSilkDir:
    """Test temporary directory cleanup."""

    def test_removes_temp_dir(self, tmp_path):
        """cleanup_silk_dir removes the temp directory."""
        silk_dir = tmp_path / "hermes-silk-test"
        silk_dir.mkdir()
        (silk_dir / "voice.silk").write_bytes(b"fake")

        from gateway.platforms.voice_utils import cleanup_silk_dir
        cleanup_silk_dir(str(silk_dir / "voice.silk"))

        assert not silk_dir.exists()

    def test_ignores_none(self):
        """cleanup_silk_dir is safe with None input."""
        from gateway.platforms.voice_utils import cleanup_silk_dir
        cleanup_silk_dir(None)  # should not raise


# ---------------------------------------------------------------------------
# Tests: _has_pilk
# ---------------------------------------------------------------------------

class TestHasPilk:
    """Test pilk availability detection."""

    def test_has_pilk_returns_bool(self):
        """_has_pilk returns a boolean."""
        from gateway.platforms.voice_utils import _has_pilk
        result = _has_pilk()
        assert isinstance(result, bool)
