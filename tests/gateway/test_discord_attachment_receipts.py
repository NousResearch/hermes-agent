"""Media targets and receipts must describe the attachment, not a fallback notice."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
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


@pytest.mark.asyncio
async def test_image_batch_requires_complete_attachment_receipt(tmp_path):
    adapter = DiscordAdapter(PlatformConfig())
    adapter._client = client = Client()
    path = tmp_path / "media.bin"
    path.write_bytes(b"attachment bytes")

    async def send_without_attachments(**kwargs):
        client.uploads.extend(kwargs["files"])
        return SimpleNamespace(id=444, attachments=[])

    client.send = send_without_attachments
    result = await adapter.send_multiple_images("222", [(f"file://{path}", "caption")])

    assert not result.success
    assert "attached 0 of 1 files" in result.error


@pytest.mark.asyncio
async def test_image_batch_propagates_failed_forum_receipt(tmp_path):
    adapter = DiscordAdapter(PlatformConfig())
    adapter._client = Client()
    adapter._is_forum_parent = lambda _channel: True
    adapter._forum_post_file = AsyncMock(
        return_value=SendResult(success=False, error="Discord forum starter contained no files")
    )
    path = tmp_path / "media.bin"
    path.write_bytes(b"attachment bytes")

    result = await adapter.send_multiple_images("222", [(f"file://{path}", "caption")])

    assert not result.success
    assert result.error == "Discord forum starter contained no files"


@pytest.mark.asyncio
async def test_image_batch_rejects_partial_forum_starter_receipt(tmp_path):
    adapter = DiscordAdapter(PlatformConfig())
    adapter._client = Client()
    adapter._is_forum_parent = lambda _channel: True
    adapter._resolve_channel = AsyncMock(
        return_value=SimpleNamespace(
            id=222,
            create_thread=AsyncMock(
                return_value=SimpleNamespace(
                    id=777,
                    message=SimpleNamespace(id=800, attachments=[SimpleNamespace()]),
                    thread=SimpleNamespace(id=777, send=AsyncMock()),
                )
            ),
        )
    )
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in paths:
        path.write_bytes(b"attachment bytes")

    result = await adapter.send_multiple_images(
        "222", [(f"file://{path}", "caption") for path in paths]
    )

    assert not result.success
    assert "attached 1 of 2 files" in result.error
    assert result.message_id == "800"
    assert result.raw_response == {"thread_id": "777"}
