"""Telegram videos over ~10 MB must carry explicit geometry and a thumbnail.

Telegram runs its own video processing only for small uploads: above roughly 10 MB the
``sendVideo`` response comes back as ``width=320 height=320 duration=0`` with no thumbnail, and
clients then draw the message as a square tile whatever the real aspect ratio. Measured on one
6 s 2560x1440 clip: 4.9 MB and 9.8 MB keep ``2560x1440``, duration 7 and a 320x180 thumbnail;
14.8 MB, 19.4 MB and 23.0 MB degrade to the square placeholder, while the same 23.0 MB file sent
with ``width``/``height``/``duration`` plus a JPEG ``thumbnail`` comes back ``2560x1440`` with a
320x180 thumbnail.

``TelegramAdapter.send_video`` therefore probes the local file and forwards that metadata, so
portrait reels and landscape clips stay correct at any size.

These tests confirm:
  1. the probed geometry and thumbnail reach ``bot.send_video``,
  2. a file ffprobe/ffmpeg cannot read still sends, just without metadata,
  3. the temporary thumbnail is removed after the send,
  4. on a real file, the helpers report the true geometry and a same-aspect thumbnail.
"""
import contextlib
import os
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_mod
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_video = AsyncMock(return_value=MagicMock(message_id=7))
    return adapter


@pytest.mark.asyncio
async def test_send_video_forwards_geometry_and_thumbnail(monkeypatch, tmp_path):
    """A video Telegram will not process itself still advertises its real shape."""
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
    monkeypatch.setattr(
        telegram_mod, "_probe_video_geometry",
        lambda _p: {"width": 2560, "height": 1440, "duration": 98})
    monkeypatch.setattr(telegram_mod, "_video_thumbnail_jpeg", lambda _p, _d: str(thumb))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 64)

    adapter = _make_adapter()
    result = await adapter.send_video("123", str(video))

    assert result.success is True
    adapter._bot.send_video.assert_awaited_once()
    kwargs = adapter._bot.send_video.await_args.kwargs
    assert (kwargs["width"], kwargs["height"], kwargs["duration"]) == (2560, 1440, 98)
    assert kwargs["thumbnail"] == str(thumb)
    assert not thumb.exists(), "the temporary thumbnail must not be left behind"


@pytest.mark.asyncio
async def test_send_video_without_probeable_media_omits_metadata(monkeypatch, tmp_path):
    """Unreadable or non-media files must still deliver, with no geometry kwargs."""
    monkeypatch.setattr(telegram_mod, "_probe_video_geometry", lambda _p: {})
    video = tmp_path / "clip.bin"
    video.write_bytes(b"\x00" * 16)

    adapter = _make_adapter()
    result = await adapter.send_video("123", str(video))

    assert result.success is True
    kwargs = adapter._bot.send_video.await_args.kwargs
    assert "width" not in kwargs and "height" not in kwargs
    assert "duration" not in kwargs and "thumbnail" not in kwargs


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed (the CI runner ships without them)",
)
def test_video_helpers_read_real_geometry(tmp_path):
    """On a real 16:9 clip the helpers report the true size and a 16:9 thumbnail."""
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        capture_output=True, check=True)

    geometry = telegram_mod._probe_video_geometry(str(clip))
    assert (geometry["width"], geometry["height"]) == (640, 360)
    assert geometry["duration"] == 2

    thumb = telegram_mod._video_thumbnail_jpeg(str(clip), geometry["duration"])
    assert thumb is not None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", thumb],
            capture_output=True, text=True, check=True).stdout.strip()
        assert probe == "320,180", "thumbnails keep the source aspect ratio"
    finally:
        with contextlib.suppress(OSError):
            os.remove(thumb)
