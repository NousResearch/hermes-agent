import asyncio
from unittest.mock import AsyncMock, Mock

from plugins.platforms.matrix.adapter import MatrixAdapter


class _FakeMatrixClient:
    crypto = None

    async def upload_media(self, data, *, mime_type, filename, size):
        self.upload = (data, mime_type, filename, size)
        return "mxc://example.org/audio"

    async def send_message_event(self, room_id, event_type, content):
        self.event = content
        return "$event"


def test_matrix_voice_event_includes_duration_metadata(monkeypatch):
    client = _FakeMatrixClient()
    adapter = object.__new__(MatrixAdapter)
    adapter._client = client
    adapter._encryption = False
    adapter._max_media_bytes = 1024 * 1024
    adapter._apply_relation_metadata = lambda *args, **kwargs: None
    monkeypatch.setattr(
        "plugins.platforms.matrix.adapter._probe_audio_duration_ms",
        lambda data, content_type: 3210,
    )

    result = asyncio.run(
        adapter._upload_and_send(
            "!room:example.org",
            b"ogg",
            "reply.ogg",
            "audio/ogg",
            "m.audio",
            is_voice=True,
        )
    )

    assert result.success is True
    assert client.upload[1] == "audio/ogg"
    assert client.event["info"]["mimetype"] == "audio/ogg"
    assert client.event["info"]["duration"] == 3210
    assert client.event["org.matrix.msc1767.audio"]["duration"] == 3210
    assert client.event["org.matrix.msc3245.voice"] == {}



def test_matrix_send_voice_converts_non_ogg_audio(tmp_path, monkeypatch):
    source = tmp_path / "reply.mp3"
    converted = tmp_path / "reply.ogg"
    source.write_bytes(b"mp3")
    converted.write_bytes(b"ogg")

    convert = Mock(return_value=str(converted))
    monkeypatch.setattr("tools.tts_tool._convert_to_opus", convert)

    adapter = object.__new__(MatrixAdapter)
    adapter._send_local_file = AsyncMock(return_value="sent")

    result = asyncio.run(adapter.send_voice("!room:example.org", str(source)))

    assert result == "sent"
    convert.assert_called_once_with(str(source))
    adapter._send_local_file.assert_awaited_once_with(
        "!room:example.org",
        str(converted),
        "m.audio",
        None,
        None,
        metadata=None,
        is_voice=True,
    )
