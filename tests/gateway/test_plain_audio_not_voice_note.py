"""A plain audio MEDIA file (``is_voice=False``: music, a sound effect) is never sent as a voice note.

The media dispatch routes every audio extension to ``send_voice(..., is_voice=<bool>)``; adapters
whose ``send_voice`` is a native voice-note transport must honour the flag. Each case records the
adapter's own wire boundary.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("is_voice, expected", [(False, "document"), (None, "audio")], ids=["plain", "voice"])
async def test_whatsapp_bridge_sends_plain_audio_as_a_document(tmp_path, is_voice, expected):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_path": str(tmp_path / "wa")}))
    posted = []

    async def post_bridge_message(route, payload, timeout=0):
        posted.append(payload)
        return SimpleNamespace(success=True, message_id=None)

    adapter._post_bridge_message = post_bridge_message
    adapter._bridge_unavailable = AsyncMock(return_value=None)
    path = tmp_path / "door_knock.mp3"
    path.write_bytes(b"ID3\x04" + b"\x00" * 32)

    assert (await adapter.send_voice("15551234567", str(path), is_voice=is_voice)).success
    assert [p["mediaType"] for p in posted] == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_voice, audio_message", [(False, False), (None, True)], ids=["plain", "voice"])
async def test_bluebubbles_sends_plain_audio_without_the_voice_memo_flag(tmp_path, monkeypatch, is_voice, audio_message):
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter

    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={"server_url": "http://localhost:1234", "password": "x"}))
    response = MagicMock()
    response.json.return_value = {"status": 200, "data": {"guid": "g1"}}
    adapter.client = SimpleNamespace(post=AsyncMock(return_value=response))
    monkeypatch.setattr(adapter, "_resolve_chat_guid", AsyncMock(return_value="iMessage;-;+15551234567"))
    path = tmp_path / "door_knock.mp3"
    path.write_bytes(b"ID3\x04" + b"\x00" * 32)

    assert (await adapter.send_voice("+15551234567", str(path), is_voice=is_voice)).success
    data = adapter.client.post.await_args.kwargs["data"]
    assert ("isAudioMessage" in data) is audio_message
    assert data["name"] == "door_knock.mp3"


@pytest.mark.asyncio
async def test_whatsapp_cloud_never_converts_plain_mp3_into_a_voice_bubble(tmp_path):
    from plugins.platforms.whatsapp_cloud.adapter import WhatsAppCloudAdapter

    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._convert_to_opus = AsyncMock(side_effect=AssertionError("plain audio must not become Opus"))
    adapter._send_media_from_path_or_link = AsyncMock(return_value=SimpleNamespace(success=True))
    path = tmp_path / "door_knock.mp3"
    path.write_bytes(b"ID3\x04" + b"\x00" * 32)

    assert (await adapter.send_voice("15551234567", str(path), is_voice=False)).success
    call = adapter._send_media_from_path_or_link.await_args
    assert call.args[1] == str(path)
    assert call.kwargs["mime_type"] == "audio/mpeg"
