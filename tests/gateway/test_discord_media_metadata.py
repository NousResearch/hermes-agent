import inspect
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.discord import adapter as discord_adapter_module
from plugins.platforms.discord.adapter import DiscordAdapter


def test_discord_media_methods_accept_metadata_kwarg():
    for method_name in (
        "send_voice",
        "send_image_file",
        "send_image",
        "send_video",
        "send_document",
        "send_animation",
        "send_multiple_images",
    ):
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


class _FakeAiohttpResponse:
    def __init__(self, data: bytes, content_type: str):
        self.status = 200
        self.headers = {"content-type": content_type}
        self._data = data

    async def read(self):
        return self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeAiohttpSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, **kwargs):
        return self._response

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _install_fake_aiohttp(monkeypatch, data: bytes, content_type: str):
    """Replace ``import aiohttp`` with a stub serving one canned download."""
    session = _FakeAiohttpSession(_FakeAiohttpResponse(data, content_type))
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(
            ClientSession=lambda **kwargs: session,
            ClientTimeout=lambda **kwargs: None,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_trigger", ["aiohttp_import_error", "native_send_failure"])
async def test_discord_send_image_fallbacks_preserve_metadata_thread(monkeypatch, fallback_trigger):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    target_thread = SimpleNamespace(id=200, type=11, send=AsyncMock(return_value=SimpleNamespace(id=300)))
    adapter._client = _FakeClient({100: parent_forum, 200: target_thread})

    # The hermetic test env has no DNS, so the SSRF guard would fail closed
    # and route into the (already metadata-aware) unsafe-URL branch instead
    # of the native download path under test.
    monkeypatch.setattr(discord_adapter_module, "is_safe_url", lambda url: True)

    if fallback_trigger == "aiohttp_import_error":
        monkeypatch.setitem(sys.modules, "aiohttp", None)
    else:
        def _raise_client_session(**kwargs):
            raise RuntimeError("native image send failed")

        monkeypatch.setitem(
            sys.modules,
            "aiohttp",
            SimpleNamespace(
                ClientSession=_raise_client_session,
                ClientTimeout=lambda **kwargs: None,
            ),
        )

    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="900"))

    result = await adapter.send_image(
        chat_id="100",
        image_url="https://cdn.example.com/pic.png",
        caption="hello",
        metadata={"thread_id": "200"},
    )

    assert result.success is True
    adapter.send.assert_awaited_once()
    send_kwargs = adapter.send.await_args.kwargs
    assert send_kwargs["chat_id"] == "100"
    assert "https://cdn.example.com/pic.png" in send_kwargs["content"]
    assert send_kwargs["metadata"] == {"thread_id": "200"}
    parent_forum.create_thread.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "media_kwarg", "media_value"),
    [
        ("send_document", "file_path", "FILE"),
        ("send_image_file", "image_path", "FILE"),
        ("send_video", "video_path", "FILE"),
        ("send_voice", "audio_path", "FILE"),
        ("send_image", "image_url", "https://cdn.example.com/pic.png"),
        ("send_animation", "animation_url", "https://cdn.example.com/anim.gif"),
    ],
)
async def test_discord_media_reports_missing_metadata_thread(
    tmp_path,
    monkeypatch,
    method_name,
    media_kwarg,
    media_value,
):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    adapter._client = _FakeClient({100: parent_forum})

    monkeypatch.setattr(discord_adapter_module, "is_safe_url", lambda url: True)

    if media_value == "FILE":
        media_path = tmp_path / "media.bin"
        media_path.write_bytes(b"fixture-bytes")
        media_value = str(media_path)

    result = await getattr(adapter, method_name)(
        chat_id="100",
        **{media_kwarg: media_value},
        metadata={"thread_id": "200"},
    )

    assert result.success is False
    assert result.error == "Thread 200 not found"
    parent_forum.create_thread.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "url_kwarg", "content_type", "filename"),
    [
        ("send_image", "image_url", "image/png", "image.png"),
        ("send_animation", "animation_url", "image/gif", "animation.gif"),
    ],
)
async def test_discord_url_media_sends_to_metadata_thread_not_parent_forum(
    monkeypatch,
    method_name,
    url_kwarg,
    content_type,
    filename,
):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    target_thread = SimpleNamespace(id=200, type=11, send=AsyncMock(return_value=SimpleNamespace(id=300)))
    adapter._client = _FakeClient({100: parent_forum, 200: target_thread})

    monkeypatch.setattr(discord_adapter_module, "is_safe_url", lambda url: True)
    _install_fake_aiohttp(monkeypatch, b"fixture-media-bytes", content_type)

    result = await getattr(adapter, method_name)(
        chat_id="100",
        **{url_kwarg: "https://cdn.example.com/media.bin"},
        caption="hello",
        metadata={"thread_id": "200"},
    )

    assert result.success is True
    assert result.message_id == "300"
    parent_forum.create_thread.assert_not_awaited()
    target_thread.send.assert_awaited_once()
    send_kwargs = target_thread.send.await_args.kwargs
    assert send_kwargs["content"] == "hello"
    assert send_kwargs["file"].filename == filename


@pytest.mark.asyncio
async def test_discord_send_image_url_creates_forum_post_without_metadata(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    forum_thread = SimpleNamespace(id=201, send=AsyncMock())
    starter_message = SimpleNamespace(id=301)
    parent_forum = SimpleNamespace(
        id=100,
        type=15,
        create_thread=AsyncMock(return_value=SimpleNamespace(thread=forum_thread, message=starter_message)),
    )
    adapter._client = _FakeClient({100: parent_forum})

    monkeypatch.setattr(discord_adapter_module, "is_safe_url", lambda url: True)
    _install_fake_aiohttp(monkeypatch, b"fixture-image-bytes", "image/png")

    result = await adapter.send_image(
        chat_id="100",
        image_url="https://cdn.example.com/pic.png",
        caption="hello",
        metadata=None,
    )

    assert result.success is True
    assert result.message_id == "301"
    assert result.raw_response == {"thread_id": "201"}
    parent_forum.create_thread.assert_awaited_once()
    create_kwargs = parent_forum.create_thread.await_args.kwargs
    assert create_kwargs["file"].filename == "image.png"


@pytest.mark.asyncio
async def test_discord_multiple_images_send_to_metadata_thread_not_parent_forum(tmp_path):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    first = tmp_path / "first.png"
    first.write_bytes(b"fixture-image-1")
    second = tmp_path / "second.png"
    second.write_bytes(b"fixture-image-2")

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    target_thread = SimpleNamespace(id=200, type=11, send=AsyncMock(return_value=SimpleNamespace(id=300)))
    adapter._client = _FakeClient({100: parent_forum, 200: target_thread})

    await adapter.send_multiple_images(
        chat_id="100",
        images=[(first.as_uri(), "first caption"), (second.as_uri(), "")],
        metadata={"thread_id": "200"},
    )

    parent_forum.create_thread.assert_not_awaited()
    target_thread.send.assert_awaited_once()
    send_kwargs = target_thread.send.await_args.kwargs
    assert send_kwargs["content"] == "first caption"
    assert len(send_kwargs["files"]) == 2


@pytest.mark.asyncio
async def test_discord_multiple_images_create_forum_post_without_metadata(tmp_path):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    first = tmp_path / "first.png"
    first.write_bytes(b"fixture-image-1")
    second = tmp_path / "second.png"
    second.write_bytes(b"fixture-image-2")

    forum_thread = SimpleNamespace(id=201, send=AsyncMock())
    starter_message = SimpleNamespace(id=301)
    parent_forum = SimpleNamespace(
        id=100,
        type=15,
        create_thread=AsyncMock(return_value=SimpleNamespace(thread=forum_thread, message=starter_message)),
    )
    adapter._client = _FakeClient({100: parent_forum})

    await adapter.send_multiple_images(
        chat_id="100",
        images=[(first.as_uri(), "first caption"), (second.as_uri(), "")],
        metadata=None,
    )

    parent_forum.create_thread.assert_awaited_once()
    create_kwargs = parent_forum.create_thread.await_args.kwargs
    assert len(create_kwargs["files"]) == 2


@pytest.mark.asyncio
async def test_discord_multiple_images_fallback_preserves_metadata_per_image(tmp_path):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    first = tmp_path / "first.png"
    first.write_bytes(b"fixture-image-1")
    second = tmp_path / "second.png"
    second.write_bytes(b"fixture-image-2")

    parent_forum = SimpleNamespace(id=100, type=15, create_thread=AsyncMock())
    target_thread = SimpleNamespace(
        id=200,
        type=11,
        send=AsyncMock(side_effect=RuntimeError("batched send failed")),
    )
    adapter._client = _FakeClient({100: parent_forum, 200: target_thread})

    # The optimized batch send fails, so the base per-image loop takes over;
    # each image must stay routed to the metadata thread.
    adapter.send_image_file = AsyncMock(return_value=SendResult(success=True, message_id="300"))

    await adapter.send_multiple_images(
        chat_id="100",
        images=[(first.as_uri(), "first caption"), (second.as_uri(), "")],
        metadata={"thread_id": "200"},
    )

    parent_forum.create_thread.assert_not_awaited()
    assert adapter.send_image_file.await_count == 2
    sent_paths = []
    for call in adapter.send_image_file.await_args_list:
        assert call.kwargs["metadata"] == {"thread_id": "200"}
        sent_paths.append(call.kwargs["image_path"])
    assert sent_paths == [str(first), str(second)]
