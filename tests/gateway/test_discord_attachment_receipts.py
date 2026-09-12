"""Media targets and receipts must describe the attachment, not a fallback notice."""
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


class Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.ids = []
        self.uploads = []
        self.id = 222
        self.http = self

    def get_channel(self, channel_id):
        self.ids.append(channel_id)
        return self if channel_id == self.id else None

    async def fetch_channel(self, channel_id):
        raise RuntimeError(f"10003 Unknown Channel {channel_id}")

    async def request(self, *args, **kwargs):
        raise RuntimeError("native voice transport unavailable")

    async def send(self, **kwargs):
        files = kwargs.get("files", []) or ([kwargs["file"]] if kwargs.get("file") else [])
        if self.fail and files:
            raise RuntimeError("upload rejected")
        self.uploads.extend(files)
        return SimpleNamespace(id=444, attachments=files)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send_document", "send_image_file", "send_video", "send_voice", "send_multiple_images"])
async def test_media_uses_metadata_target_without_losing_direct_channel(tmp_path, method):
    adapter = DiscordAdapter(PlatformConfig())
    adapter._client = client = Client()
    path = tmp_path / "media.bin"
    path.write_bytes(b"attachment bytes")
    media = [(path.as_uri(), "caption")] if method == "send_multiple_images" else str(path)
    for chat_id, metadata in [("111", {"thread_id": "222"}), ("222", None)]:
        client.ids.clear()
        client.uploads.clear()
        result = await getattr(adapter, method)(chat_id, media, metadata=metadata)
        assert result is None or result.success
        assert client.ids == [222]
        assert len(client.uploads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send_document", "send_image_file", "send_video", "send_voice"])
async def test_failed_upload_never_becomes_successful_text_notice(tmp_path, method):
    adapter = DiscordAdapter(PlatformConfig())
    adapter._client = client = Client(fail=True)
    path = tmp_path / "media.bin"
    path.write_bytes(b"attachment bytes")
    result = await getattr(adapter, method)("111", str(path), metadata={"thread_id": "222"})
    assert not result.success
    assert "upload rejected" in result.error
    assert not client.uploads
