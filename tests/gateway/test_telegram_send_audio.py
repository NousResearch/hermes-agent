"""``sendAudio`` (.mp3/.m4a) gets the same caption formatting as voice notes, plus the track's own
performer/title/cover tags, so Telegram's music player shows a named track instead of raw markdown."""

import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

_needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed (the CI runner ships without them)",
)


def _cover_bytes(thumbnail) -> bytes:
    """What PTB would upload for ``thumbnail`` (a path or an open handle)."""
    if hasattr(thumbnail, "read"):
        return thumbnail.read()
    with open(thumbnail, "rb") as fh:
        return fh.read()


def _tagged_mp3(tmp_path):
    cover, mp3 = tmp_path / "cover.png", tmp_path / "song.mp3"
    run = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True,
                                    stdin=subprocess.DEVNULL)
    run("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(cover))
    run("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-i", str(cover), "-map", "0:a", "-map", "1:v",
        "-c:v", "mjpeg", "-disposition:v", "attached_pic", "-metadata", "artist=Hermes Band",
        "-metadata", "title=Probe Song", "-id3v2_version", "3", str(mp3))
    return mp3


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_mp3_send_formats_caption_and_forwards_player_tags(tmp_path):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    covers = []
    adapter._bot.send_audio = AsyncMock(
        side_effect=lambda **kw: covers.append(_cover_bytes(kw["thumbnail"])[:2]) or MagicMock(message_id=2))

    result = await adapter.send_voice("123", str(_tagged_mp3(tmp_path)), caption="Here is **your song** (v1.2)!")

    assert result.success is True
    kwargs = adapter._bot.send_audio.await_args.kwargs
    assert kwargs["parse_mode"] == "MarkdownV2"
    assert kwargs["caption"] == "Here is *your song* \\(v1\\.2\\)\\!"
    assert (kwargs["performer"], kwargs["title"]) == ("Hermes Band", "Probe Song")
    assert covers == [b"\xff\xd8"]  # the embedded cover, as a JPEG


@pytest.mark.asyncio
async def test_mp3_caption_entity_rejection_retries_plain(tmp_path):
    """Needs no ffmpeg: an untaggable track still goes out, and the caption retry drops parse_mode."""
    track = tmp_path / "song.mp3"
    track.write_bytes(b"not really an mp3")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_audio = AsyncMock(
        side_effect=[Exception("Bad Request: can't parse entities"), MagicMock(message_id=7)])

    result = await adapter.send_voice("123", str(track), caption="*bold* reply")

    assert result.success is True
    retry = adapter._bot.send_audio.await_args_list[1].kwargs
    assert (retry["parse_mode"], retry["caption"]) == (None, "*bold* reply")


@_needs_ffmpeg
@pytest.mark.asyncio
async def test_mp3_caption_retry_still_carries_the_cover(tmp_path):
    """Every attempt (caption fallback included) uploads the full cover, not a drained handle."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    covers = []

    def _send(**kw):
        covers.append(_cover_bytes(kw["thumbnail"]))
        if len(covers) == 1:
            raise Exception("Bad Request: can't parse entities")
        return MagicMock(message_id=8)

    adapter._bot.send_audio = AsyncMock(side_effect=_send)

    result = await adapter.send_voice("123", str(_tagged_mp3(tmp_path)), caption="*bold* reply")

    assert result.success is True
    assert len(covers) == 2 and covers[0][:2] == b"\xff\xd8" and covers[1] == covers[0]
