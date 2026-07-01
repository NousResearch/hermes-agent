import os
import time

from gateway.platforms.base import (
    cleanup_screenshot_cache,
    cleanup_video_cache,
    get_video_cache_dir,
)


def test_cleanup_video_cache_removes_only_stale_files(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "videos")
    cache_dir = get_video_cache_dir()
    old_file = cache_dir / "old.mp4"
    fresh_file = cache_dir / "fresh.mp4"
    old_file.write_bytes(b"old")
    fresh_file.write_bytes(b"fresh")
    old_time = time.time() - 48 * 3600
    os.utime(old_file, (old_time, old_time))

    assert cleanup_video_cache(max_age_hours=24) == 1
    assert not old_file.exists()
    assert fresh_file.exists()


def test_cleanup_screenshot_cache_removes_only_stale_files(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.SCREENSHOT_CACHE_DIR", tmp_path / "screenshots")
    cache_dir = tmp_path / "screenshots"
    cache_dir.mkdir()
    old_file = cache_dir / "old.png"
    fresh_file = cache_dir / "fresh.png"
    old_file.write_bytes(b"old")
    fresh_file.write_bytes(b"fresh")
    old_time = time.time() - 48 * 3600
    os.utime(old_file, (old_time, old_time))

    assert cleanup_screenshot_cache(max_age_hours=24) == 1
    assert not old_file.exists()
    assert fresh_file.exists()
