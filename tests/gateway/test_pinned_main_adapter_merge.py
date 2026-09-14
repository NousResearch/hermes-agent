"""Real adapter dispatch through extracted owners after the pinned main merge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from gateway.native_document_guard import require_native_document
from gateway.platforms.base import SendResult
from tests.gateway.test_native_document_guard import native


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["discord"], indirect=True)
@pytest.mark.parametrize("method", ["send_document", "send_image_file", "send_video"])
async def test_local_media_uses_canonical_owner_and_metadata_recipient(native, method):
    from plugins.platforms.discord.adapter_media import DiscordMediaMixin

    adapter = native.adapter
    channel = adapter._client.get_channel(7)
    adapter._client.get_channel = Mock(return_value=channel)
    assert getattr(type(adapter), method) is getattr(DiscordMediaMixin, method)
    with require_native_document():
        result = await getattr(adapter, method)(
            "123", str(native.path), metadata={"thread_id": "7"},
        )
    assert result.success is True
    adapter._client.get_channel.assert_called_once_with(7)
    assert native.calls[0][0] == native.path.read_bytes()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["discord"], indirect=True)
async def test_batch_media_routes_to_metadata_target(native):
    from plugins.platforms.discord.adapter_media import DiscordMediaMixin

    adapter = native.adapter
    channel = adapter._client.get_channel(7)
    adapter._client.get_channel = Mock(return_value=channel)
    assert type(adapter).send_multiple_images is DiscordMediaMixin.send_multiple_images
    await adapter.send_multiple_images(
        "123", [(native.path.as_uri(), "Image")], metadata={"thread_id": "7"},
    )
    adapter._client.get_channel.assert_called_once_with(7)
    assert native.calls[0][0] == native.path.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["discord"], indirect=True)
async def test_voice_routes_raw_upload_to_metadata_target(native):
    from plugins.platforms.discord.adapter_media import DiscordMediaMixin

    adapter = native.adapter
    channel = adapter._client.get_channel(7)
    channel.id = 7
    adapter._client.get_channel = Mock(return_value=channel)
    request = AsyncMock(return_value={"id": "900"})
    adapter._client.http = SimpleNamespace(request=request)
    assert type(adapter).send_voice is DiscordMediaMixin.send_voice
    result = await adapter.send_voice("123", str(native.path), metadata={"thread_id": "7"})
    assert result.success and result.message_id == "900"
    adapter._client.get_channel.assert_called_once_with(7)
    assert request.await_args.args[0].url.endswith("/channels/7/messages")
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["discord"], indirect=True)
@pytest.mark.parametrize("method", ["send_image", "send_animation"])
async def test_url_failure_keeps_metadata_on_base_fallback(native, monkeypatch, method):
    from plugins.platforms.discord import adapter as facade

    adapter = native.adapter
    channel = adapter._client.get_channel(7)
    adapter._client.get_channel = Mock(return_value=channel)
    monkeypatch.setattr(facade, "is_safe_url", lambda value: True)
    monkeypatch.setattr(facade, "_read_url_image_with_redirect_guard", AsyncMock(side_effect=OSError("download failed")))
    metadata = {"thread_id": "7", "notify": True}
    result = await getattr(adapter, method)("123", "https://example.com/media.png", metadata=metadata)
    assert result.success
    # The Base animation fallback retries through send_image; every attempt must
    # retain the same recipient, including that second media-dispatch path.
    assert adapter._client.get_channel.call_args_list
    assert all(args == call(7) for args in adapter._client.get_channel.call_args_list)
    assert adapter.send.await_args.kwargs["metadata"] == metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["telegram"], indirect=True)
async def test_exhausted_short_flood_send_is_a_definite_refusal(native, monkeypatch):
    from gateway.delivery_ledger import flood_wait_seconds
    from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
    from plugins.platforms.telegram import adapter as facade

    adapter = native.adapter
    sender = AsyncMock(side_effect=RetryAfter(1))
    monkeypatch.setattr(adapter, "_send_chunk_markdown_or_plain", sender)
    sleep = AsyncMock()
    monkeypatch.setattr(facade.asyncio, "sleep", sleep)
    result = await adapter._send_chunk_with_retries(
        "123", "text", 0, None, {}, None, False, (NetworkError, BadRequest, TimedOut),
    )
    assert isinstance(result, SendResult) and result.success is False
    assert result.error.startswith("flood_control:") and flood_wait_seconds(result.error) == 1
    assert result.retryable is False
    assert sender.await_count == 3 and sleep.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["telegram"], indirect=True)
async def test_second_edit_flood_keeps_new_delay(native, monkeypatch):
    from gateway.delivery_ledger import flood_wait_seconds
    from telegram.error import RetryAfter
    from plugins.platforms.telegram import adapter as facade

    adapter = native.adapter
    edit = AsyncMock(side_effect=[RetryAfter(1), RetryAfter(37)])
    monkeypatch.setattr(adapter, "_edit_text", edit)
    sleep = AsyncMock()
    monkeypatch.setattr(facade.asyncio, "sleep", sleep)
    result = await adapter.edit_message("123", "456", "text")
    assert result.success is False and result.error.startswith("flood_control:")
    assert flood_wait_seconds(result.error) == 37
    assert result.retryable is False
    sleep.assert_awaited_once_with(1)
    assert edit.await_count == 2
