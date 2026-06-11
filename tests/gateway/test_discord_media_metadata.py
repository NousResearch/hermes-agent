import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def test_discord_media_methods_accept_metadata_kwarg():
    for method_name in ("send_voice", "send_image_file", "send_image", "send_video", "send_document"):
        signature = inspect.signature(getattr(DiscordAdapter, method_name))
        assert "metadata" in signature.parameters, method_name


class _FakeClient:
    def __init__(self, channels):
        self.channels = {int(key): value for key, value in channels.items()}
        # Force send_voice's native Discord voice-message path to fall back to
        # the ordinary file attachment path exercised by other media sends.
        self.http = SimpleNamespace(request=AsyncMock(side_effect=RuntimeError("force file fallback")))
        self.fetch_channel = AsyncMock(side_effect=self._fetch_channel)

    def get_channel(self, channel_id):
        return self.channels.get(int(channel_id))

    async def _fetch_channel(self, channel_id):
        return self.channels.get(int(channel_id))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "path_kwarg", "filename"),
    [
        ("send_document", "file_path", "report.txt"),
        ("send_image_file", "image_path", "image.png"),
        ("send_video", "video_path", "clip.mp4"),
    ],
)
async def test_discord_file_media_methods_send_to_metadata_thread_not_parent_forum(
    tmp_path,
    method_name,
    path_kwarg,
    filename,
):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    media_path = tmp_path / filename
    media_path.write_bytes(b"fixture-bytes")

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    target_thread = SimpleNamespace(id=200, type=11, send=AsyncMock(return_value=SimpleNamespace(id=300)))
    adapter._client = _FakeClient({100: parent_forum, 200: target_thread})

    result = await getattr(adapter, method_name)(
        chat_id="100",
        **{path_kwarg: str(media_path)},
        metadata={"thread_id": "200"},
    )

    assert result.success is True
    parent_forum.create_thread.assert_not_awaited()
    target_thread.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_discord_voice_media_sends_to_metadata_thread_not_parent_forum(tmp_path):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"fixture-audio")

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    target_thread = SimpleNamespace(id=200, type=11, send=AsyncMock(return_value=SimpleNamespace(id=301)))
    adapter._client = _FakeClient({100: parent_forum, 200: target_thread})

    result = await adapter.send_voice(
        chat_id="100",
        audio_path=str(audio_path),
        metadata={"thread_id": "200"},
    )

    assert result.success is True
    parent_forum.create_thread.assert_not_awaited()
    target_thread.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_discord_file_media_preserves_forum_thread_creation_without_metadata(tmp_path):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    media_path = tmp_path / "report.txt"
    media_path.write_bytes(b"fixture-bytes")

    forum_thread = SimpleNamespace(id=201, send=AsyncMock())
    starter_message = SimpleNamespace(id=301)
    parent_forum = SimpleNamespace(
        id=100,
        type=15,
        create_thread=AsyncMock(return_value=SimpleNamespace(thread=forum_thread, message=starter_message)),
    )
    adapter._client = _FakeClient({100: parent_forum})

    result = await adapter.send_document(
        chat_id="100",
        file_path=str(media_path),
        metadata=None,
    )

    assert result.success is True
    assert result.message_id == "301"
    assert result.raw_response == {"thread_id": "201"}
    parent_forum.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_discord_file_media_reports_missing_metadata_thread(tmp_path):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    media_path = tmp_path / "report.txt"
    media_path.write_bytes(b"fixture-bytes")

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    adapter._client = _FakeClient({100: parent_forum})

    result = await adapter.send_document(
        chat_id="100",
        file_path=str(media_path),
        metadata={"thread_id": "200"},
    )

    assert result.success is False
    assert result.error == "Thread 200 not found"
    parent_forum.create_thread.assert_not_awaited()
