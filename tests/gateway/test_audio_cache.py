"""
Tests for audio cache utilities in gateway/platforms/base.py.

Covers: get_audio_cache_dir, cache_audio_from_bytes, cleanup_audio_cache.
"""

import os
import time
from pathlib import Path

import pytest

from gateway.platforms.base import (
    cache_audio_from_bytes,
    cleanup_audio_cache,
    get_audio_cache_dir,
)

# ---------------------------------------------------------------------------
# Fixture: redirect AUDIO_CACHE_DIR to a temp directory for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point the module-level AUDIO_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache"
    )


# ---------------------------------------------------------------------------
# TestGetAudioCacheDir
# ---------------------------------------------------------------------------

class TestGetAudioCacheDir:
    def test_creates_directory(self):
        cache_dir = get_audio_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# TestCacheAudioFromBytes
# ---------------------------------------------------------------------------

class TestCacheAudioFromBytes:
    def test_basic_caching(self):
        data = b"fake-ogg-bytes"
        path = cache_audio_from_bytes(data)
        assert os.path.exists(path)
        assert Path(path).read_bytes() == data

    def test_default_extension(self):
        path = cache_audio_from_bytes(b"data")
        assert path.endswith(".ogg")


# ---------------------------------------------------------------------------
# TestCleanupAudioCache
# ---------------------------------------------------------------------------

class TestCleanupAudioCache:
    def test_removes_old_files(self):
        cache_dir = get_audio_cache_dir()
        old_file = cache_dir / "old.ogg"
        old_file.write_text("old")
        # Set modification time to 48 hours ago
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_audio_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self):
        cache_dir = get_audio_cache_dir()
        recent = cache_dir / "recent.ogg"
        recent.write_text("fresh")

        removed = cleanup_audio_cache(max_age_hours=24)
        assert removed == 0
        assert recent.exists()

    def test_tts_voice_memos_survive_the_sweep(self):
        """#100075: TTS-generated files (tts_* prefix) share the audio
        cache with inbound voice notes, but they are user artifacts the
        text_to_speech contract promises persistent — the hourly sweep
        must never delete them, regardless of age."""
        cache_dir = get_audio_cache_dir()
        memo = cache_dir / "tts_20260901_120000_000000.mp3"
        memo.write_text("voice memo")
        memo_mtime = time.time() - 48 * 3600  # two days old — past cutoff
        os.utime(memo, (memo_mtime, memo_mtime))

        removed = cleanup_audio_cache(max_age_hours=24)

        assert memo.exists(), "TTS memo must survive the cache sweep"
        assert removed == 0


# ---------------------------------------------------------------------------
# TestUnifiedMediaCacheCleanup — video + screenshot ride the same shared loop
# ---------------------------------------------------------------------------

class TestUnifiedMediaCacheCleanup:

    def test_cleanup_screenshot_cache_removes_old_files(self, tmp_path, monkeypatch):
        from gateway.platforms.base import (
            cleanup_screenshot_cache,
            get_screenshot_cache_dir,
        )

        monkeypatch.setattr(
            "gateway.platforms.base.SCREENSHOT_CACHE_DIR", tmp_path / "screenshots"
        )
        cache_dir = get_screenshot_cache_dir()
        old_file = cache_dir / "old.png"
        old_file.write_text("old")
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))
        fresh = cache_dir / "fresh.png"
        fresh.write_text("fresh")

        removed = cleanup_screenshot_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()
        assert fresh.exists()

