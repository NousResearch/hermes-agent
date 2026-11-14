"""Discord audio delivery: plain audio is an attachment, voice messages are real Ogg/Opus."""
import json
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _ogg_opus(seconds: float, pre_skip: int = 312, last_packet: bytes = b"\xfc" * 8) -> bytes:
    """Minimal Ogg/Opus stream: an OpusHead page and a final page whose granule ends the audio."""
    def page(header_type: int, granule: int, seq: int, packet: bytes) -> bytes:
        return (b"OggS" + bytes([0, header_type]) + granule.to_bytes(8, "little") + b"\x01\x00\x00\x00"
                + seq.to_bytes(4, "little") + b"\x00" * 4 + bytes([1, len(packet)]) + packet)
    head = b"OpusHead" + bytes([1, 1]) + pre_skip.to_bytes(2, "little") + (48000).to_bytes(4, "little") + b"\x00" * 3
    return page(2, 0, 0, head) + page(4, int(seconds * 48000) + pre_skip, 1, last_packet)


class Channel:
    """One text channel: records raw voice-message POSTs and regular attachment sends."""

    id = 222
    guild = None

    def __init__(self):
        self.http = self
        self.voice_posts = []
        self.attachments = []

    def get_channel(self, channel_id):
        return self if channel_id == self.id else None

    async def request(self, route, *, form):
        payload = json.loads(form[0]["value"])
        self.voice_posts.append({"payload": payload, "file": form[1]})
        return {"id": "901"}

    async def send(self, **kwargs):
        files = kwargs.get("files", []) or ([kwargs["file"]] if kwargs.get("file") else [])
        self.attachments.extend(f.filename for f in files)
        return SimpleNamespace(id=902, attachments=files)


def _adapter():
    adapter = DiscordAdapter(PlatformConfig())
    adapter._client = channel = Channel()
    return adapter, channel


@pytest.mark.asyncio
@pytest.mark.parametrize("name, data", [
    ("door_knock.mp3", b"ID3\x04" + b"\x00" * 64),
    ("theme.wav", b"RIFF" + b"\x00" * 64),
    ("ambient.ogg", _ogg_opus(4.0)),
], ids=["mp3", "wav", "ogg-opus"])
async def test_plain_audio_media_is_a_named_attachment_not_a_voice_message(tmp_path, name, data):
    """``is_voice=False`` (an audio MEDIA file without [[audio_as_voice]]) keeps its real name
    and format as a regular attachment; it never becomes a ``voice-message.ogg`` voice note."""
    adapter, channel = _adapter()
    path = tmp_path / name
    path.write_bytes(data)

    result = await adapter.send_voice("222", str(path), is_voice=False)

    assert result.success
    assert channel.attachments == [name]
    assert channel.voice_posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [{"is_voice": True}, {}], ids=["audio_as_voice", "auto-tts"])
async def test_voice_message_carries_ogg_opus_bytes_and_their_real_duration(tmp_path, monkeypatch, flag):
    """Voice intent ([[audio_as_voice]] or an auto-TTS reply with no flag) posts a native voice
    message whose duration is the stream's own length; audio that cannot become Ogg/Opus (no
    ffmpeg) goes out as a playable attachment instead of a mislabelled voice note."""
    adapter, channel = _adapter()
    opus = tmp_path / "reply.ogg"
    opus.write_bytes(_ogg_opus(3.0))

    assert (await adapter.send_voice("222", str(opus), **flag)).success
    [post] = channel.voice_posts
    assert post["payload"]["flags"] == 8192
    assert post["payload"]["attachments"][0]["duration_secs"] == 3.0
    assert post["file"]["value"] == opus.read_bytes()

    monkeypatch.setattr("gateway.platforms.base.transcode_to_ogg_opus", lambda *a, **k: None)
    mp3 = tmp_path / "spoken.mp3"
    mp3.write_bytes(b"ID3\x04" + b"\x00" * 64)
    assert (await adapter.send_voice("222", str(mp3), **flag)).success
    assert len(channel.voice_posts) == 1
    assert channel.attachments == ["spoken.mp3"]


@pytest.mark.asyncio
async def test_non_opus_voice_reply_is_transcoded_with_the_converted_streams_duration(tmp_path, monkeypatch):
    """An mp3 TTS reply becomes a voice message carrying the transcoder's Ogg/Opus bytes and their
    length (read from page headers, not from an "OggS" run inside packet data); the temporary
    transcode is removed."""
    adapter, channel = _adapter()
    mp3 = tmp_path / "tts_reply.mp3"
    mp3.write_bytes(b"ID3\x04" + b"\x00" * 64)
    converted = tmp_path / "voice_transcode_1.ogg"
    opus = _ogg_opus(2.5, last_packet=b"\xfcOggS\x00" + b"\x00" * 16)
    calls = []

    def fake_transcode(path, **kwargs):
        calls.append(path)
        converted.write_bytes(opus)
        return str(converted)

    monkeypatch.setattr("gateway.platforms.base.transcode_to_ogg_opus", fake_transcode)

    assert (await adapter.send_voice("222", str(mp3))).success
    assert calls == [str(mp3)]
    [post] = channel.voice_posts
    assert post["file"]["value"] == opus
    assert post["payload"]["attachments"][0]["duration_secs"] == 2.5
    assert not converted.exists()
