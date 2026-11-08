"""``sendAudio`` (.mp3/.m4a) gets the same caption formatting as voice notes, plus the track's own
performer/title/cover tags, so Telegram's music player shows a named track instead of raw markdown."""

import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

def _tagged_mp3(tmp_path):
    cover, mp3 = tmp_path / "cover.png", tmp_path / "song.mp3"
    run = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True,
                                    stdin=subprocess.DEVNULL)
    run("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(cover))
    run("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-i", str(cover), "-map", "0:a", "-map", "1:v",
        "-c:v", "mjpeg", "-disposition:v", "attached_pic", "-metadata", "artist=Hermes Band",
        "-metadata", "title=Probe Song", "-id3v2_version", "3", str(mp3))
    return mp3


@pytest.mark.asyncio
async def test_mp3_send_formats_caption_and_forwards_player_tags(tmp_path):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    covers = []
    adapter._bot.send_audio = AsyncMock(
        side_effect=lambda **kw: covers.append(kw["thumbnail"].read(2)) or MagicMock(message_id=2))

    result = await adapter.send_voice("123", str(_tagged_mp3(tmp_path)), caption="Here is **your song** (v1.2)!")

    assert result.success is True
    kwargs = adapter._bot.send_audio.await_args.kwargs
    assert kwargs["parse_mode"] == "MarkdownV2"
    assert kwargs["caption"] == "Here is *your song* \\(v1\\.2\\)\\!"
    assert (kwargs["performer"], kwargs["title"]) == ("Hermes Band", "Probe Song")
    assert covers == [b"\xff\xd8"]  # the embedded cover, as a JPEG open during the upload


@pytest.mark.asyncio
async def test_mp3_caption_entity_rejection_retries_plain(tmp_path):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_audio = AsyncMock(
        side_effect=[Exception("Bad Request: can't parse entities"), MagicMock(message_id=7)])

    result = await adapter.send_voice("123", str(_tagged_mp3(tmp_path)), caption="*bold* reply")

    assert result.success is True
    retry = adapter._bot.send_audio.await_args_list[1].kwargs
    assert (retry["parse_mode"], retry["caption"]) == (None, "*bold* reply")
