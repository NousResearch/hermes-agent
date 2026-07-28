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

    def test_returns_existing_directory(self):
        first = get_audio_cache_dir()
        second = get_audio_cache_dir()
        assert first == second
        assert first.exists()


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

    def test_custom_extension(self):
        path = cache_audio_from_bytes(b"data", ext=".mp3")
        assert path.endswith(".mp3")

    def test_unique_filenames(self):
        p1 = cache_audio_from_bytes(b"a")
        p2 = cache_audio_from_bytes(b"b")
        assert p1 != p2

    def test_file_written_inside_cache_dir(self):
        path = cache_audio_from_bytes(b"data")
        cache_dir = get_audio_cache_dir()
        assert Path(path).resolve().is_relative_to(cache_dir.resolve())


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

    def test_returns_removed_count(self):
        cache_dir = get_audio_cache_dir()
        old_time = time.time() - 48 * 3600
        for i in range(3):
            f = cache_dir / f"old_{i}.ogg"
            f.write_text("old")
            os.utime(f, (old_time, old_time))

        removed = cleanup_audio_cache(max_age_hours=24)
        assert removed == 3

    def test_ignores_subdirectories(self):
        cache_dir = get_audio_cache_dir()
        subdir = cache_dir / "subdir"
        subdir.mkdir()
        old_time = time.time() - 48 * 3600
        os.utime(subdir, (old_time, old_time))

        # Should not raise or attempt to remove the directory as a file.
        removed = cleanup_audio_cache(max_age_hours=24)
        assert removed == 0
        assert subdir.exists()

    def test_empty_cache_returns_zero(self):
        removed = cleanup_audio_cache(max_age_hours=24)
        assert removed == 0

# ---------------------------------------------------------------------------
# TestUnifiedMediaCacheCleanup — video + screenshot ride the same shared loop
# ---------------------------------------------------------------------------

class TestUnifiedMediaCacheCleanup:
    def test_cleanup_video_cache_removes_old_files(self, tmp_path, monkeypatch):
        from gateway.platforms.base import cleanup_video_cache, get_video_cache_dir

        monkeypatch.setattr(
            "gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache"
        )
        cache_dir = get_video_cache_dir()
        old_file = cache_dir / "old.mp4"
        old_file.write_text("old")
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))
        fresh = cache_dir / "fresh.mp4"
        fresh.write_text("fresh")

        removed = cleanup_video_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()
        assert fresh.exists()

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

    def test_housekeeping_loop_covers_all_media_caches(self):
        """The housekeeping tick prunes every media cache via one shared loop."""
        import inspect

        from gateway import run as gateway_run

        src = inspect.getsource(gateway_run._start_gateway_housekeeping)
        assert "MEDIA_CACHE_CLEANUPS" in src
        for fn_name in (
            "cleanup_image_cache",
            "cleanup_document_cache",
            "cleanup_audio_cache",
            "cleanup_video_cache",
            "cleanup_screenshot_cache",
        ):
            assert fn_name in src, f"{fn_name} missing from housekeeping loop"
