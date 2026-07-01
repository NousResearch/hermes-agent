"""
Tests for video and screenshot cache cleanup in gateway/platforms/base.py.

Covers: cleanup_video_cache, cleanup_screenshot_cache, get_screenshot_cache_dir.
"""

import os
import time
from pathlib import Path

import pytest

from gateway.platforms.base import (
    cleanup_screenshot_cache,
    cleanup_video_cache,
    get_screenshot_cache_dir,
    get_video_cache_dir,
)


# ---------------------------------------------------------------------------
# Fixtures: redirect cache dirs to temp directories
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_video_cache(tmp_path, monkeypatch):
    """Point VIDEO_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache"
    )


@pytest.fixture(autouse=True)
def _redirect_screenshot_cache(tmp_path, monkeypatch):
    """Point SCREENSHOT_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.SCREENSHOT_CACHE_DIR", tmp_path / "screenshot_cache"
    )


# ---------------------------------------------------------------------------
# TestCleanupVideoCache
# ---------------------------------------------------------------------------

class TestCleanupVideoCache:
    def test_removes_old_files(self):
        cache_dir = get_video_cache_dir()
        old_file = cache_dir / "old_video.mp4"
        old_file.write_bytes(b"\x00" * 64)
        # Set modification time to 48 hours ago
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_video_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self):
        cache_dir = get_video_cache_dir()
        recent = cache_dir / "recent_video.mp4"
        recent.write_bytes(b"\x00" * 64)

        removed = cleanup_video_cache(max_age_hours=24)
        assert removed == 0
        assert recent.exists()

    def test_returns_removed_count(self):
        cache_dir = get_video_cache_dir()
        old_time = time.time() - 48 * 3600
        for i in range(3):
            f = cache_dir / f"old_{i}.mp4"
            f.write_bytes(b"\x00" * 64)
            os.utime(f, (old_time, old_time))

        assert cleanup_video_cache(max_age_hours=24) == 3

    def test_empty_cache_dir(self):
        assert cleanup_video_cache(max_age_hours=24) == 0

    def test_ignores_directories(self):
        cache_dir = get_video_cache_dir()
        subdir = cache_dir / "subdir"
        subdir.mkdir()
        old_mtime = time.time() - 48 * 3600
        os.utime(subdir, (old_mtime, old_mtime))

        removed = cleanup_video_cache(max_age_hours=24)
        assert removed == 0
        assert subdir.exists()


# ---------------------------------------------------------------------------
# TestCleanupScreenshotCache
# ---------------------------------------------------------------------------

class TestCleanupScreenshotCache:
    def test_removes_old_files(self):
        cache_dir = get_screenshot_cache_dir()
        old_file = cache_dir / "browser_screenshot_old.png"
        old_file.write_bytes(b"\x89PNG")
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_screenshot_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self):
        cache_dir = get_screenshot_cache_dir()
        recent = cache_dir / "browser_screenshot_new.png"
        recent.write_bytes(b"\x89PNG")

        removed = cleanup_screenshot_cache(max_age_hours=24)
        assert removed == 0
        assert recent.exists()

    def test_returns_removed_count(self):
        cache_dir = get_screenshot_cache_dir()
        old_time = time.time() - 48 * 3600
        for i in range(4):
            f = cache_dir / f"browser_screenshot_{i}.png"
            f.write_bytes(b"\x89PNG")
            os.utime(f, (old_time, old_time))

        assert cleanup_screenshot_cache(max_age_hours=24) == 4

    def test_empty_cache_dir(self):
        assert cleanup_screenshot_cache(max_age_hours=24) == 0

    def test_ignores_directories(self):
        cache_dir = get_screenshot_cache_dir()
        subdir = cache_dir / "subdir"
        subdir.mkdir()
        old_mtime = time.time() - 48 * 3600
        os.utime(subdir, (old_mtime, old_mtime))

        removed = cleanup_screenshot_cache(max_age_hours=24)
        assert removed == 0
        assert subdir.exists()


# ---------------------------------------------------------------------------
# TestGetScreenshotCacheDir
# ---------------------------------------------------------------------------

class TestGetScreenshotCacheDir:
    def test_creates_directory(self):
        cache_dir = get_screenshot_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_returns_existing_directory(self):
        first = get_screenshot_cache_dir()
        second = get_screenshot_cache_dir()
        assert first == second
        assert first.exists()
