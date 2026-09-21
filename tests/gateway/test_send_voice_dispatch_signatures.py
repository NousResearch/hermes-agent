"""Audio media dispatch reaches platform senders with the caller's voice/media distinction."""

from __future__ import annotations

import pytest

@pytest.mark.asyncio
async def test_media_dispatch_delivers_audio_through_mattermost_send_voice(tmp_path) -> None:
    """Behavioural pin for the ``**kwargs`` class fix (line/mattermost/weixin): drive the real
    base dispatch, which calls ``send_voice(..., is_voice=False)`` for an audio-ext MEDIA
    attachment. Before the fix the call raised ``TypeError`` inside ``_send_one`` and the
    attachment was recorded as a failed delivery; now the file reaches the adapter's uploader."""
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.session import Platform, SessionSource
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="mm-token", extra={"url": "https://mm.example"}))
    adapter._send_local_file = AsyncMock(return_value=SendResult(success=True, message_id="post-1"))
    adapter._notify_media_delivery_failure = AsyncMock()
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3")
    event = MessageEvent(text="", source=SessionSource(platform=Platform.MATTERMOST, chat_id="chan-1"))
    results = []

    await adapter._deliver_media_attachments(
        event, [(str(clip), False)], [], force_document_attachments=False, human_delay=0,
        metadata={}, record_delivery=results.append)

    assert [r.success for r in results] == [True]
    adapter._notify_media_delivery_failure.assert_not_awaited()
    args, kwargs = adapter._send_local_file.call_args
    assert args[1] == str(clip)


@pytest.mark.asyncio
async def test_media_dispatch_preserves_audio_kind_through_matrix_send_voice(tmp_path) -> None:
    """The real Matrix sender accepts dispatch's ``is_voice=False`` for audio files."""
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.session import Platform, SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="matrix-token", extra={}))
    adapter._send_local_file = AsyncMock(return_value=SendResult(success=True, message_id="event-1"))
    adapter._notify_media_delivery_failure = AsyncMock()
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3")
    event = MessageEvent(text="", source=SessionSource(platform=Platform.MATRIX, chat_id="!room:example.org"))
    results = []

    await adapter._deliver_media_attachments(
        event, [(str(clip), False)], [], force_document_attachments=False, human_delay=0,
        metadata={}, record_delivery=results.append)

    assert [result.success for result in results] == [True]
    adapter._notify_media_delivery_failure.assert_not_awaited()
    args, kwargs = adapter._send_local_file.call_args
    assert args[1] == str(clip)
    assert args[2] == "m.audio"
    assert kwargs["is_voice"] is False
