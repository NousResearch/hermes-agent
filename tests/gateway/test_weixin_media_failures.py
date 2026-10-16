"""Weixin receives a visible failure note when an attachment cannot be cached."""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms import weixin


@pytest.mark.asyncio
async def test_attachment_only_download_failure_reaches_the_agent():
    adapter = weixin.WeixinAdapter(
        weixin.PlatformConfig(enabled=True, token="test-token", extra={"account_id": "test-account"})
    )
    adapter._poll_session = object()
    adapter._token = ""
    adapter._download_media = AsyncMock(return_value=(None, "application/zip"))
    captured = []
    adapter._enqueue_text_event = captured.append

    await adapter._process_message({
        "from_user_id": "user-1",
        "message_id": "media-1",
        "item_list": [{"type": weixin.ITEM_FILE, "file_item": {"file_name": "archive.zip"}}],
    })

    assert len(captured) == 1
    assert captured[0].text == "[Attachment download failed: file]"


@pytest.mark.asyncio
async def test_plain_text_item_does_not_receive_download_failure_note():
    adapter = weixin.WeixinAdapter(
        weixin.PlatformConfig(enabled=True, token="test-token", extra={"account_id": "test-account"})
    )
    adapter._poll_session = object()
    adapter._token = ""
    adapter._download_media = AsyncMock(return_value=(None, ""))
    captured = []
    adapter._enqueue_text_event = captured.append

    await adapter._process_message({
        "from_user_id": "user-1",
        "message_id": "text-1",
        "item_list": [{"type": weixin.ITEM_TEXT, "text_item": {"text": "hello"}}],
    })

    assert len(captured) == 1
    assert captured[0].text == "hello"
    adapter._download_media.assert_not_called()


@pytest.mark.asyncio
async def test_transcript_only_voice_is_not_reported_as_failed_media():
    adapter = weixin.WeixinAdapter(
        weixin.PlatformConfig(enabled=True, token="test-token", extra={"account_id": "test-account"})
    )
    adapter._poll_session = object()
    adapter._token = ""
    adapter._download_media = AsyncMock(return_value=(None, "audio/silk"))
    captured = []
    adapter._enqueue_text_event = captured.append

    await adapter._process_message({
        "from_user_id": "user-1",
        "message_id": "voice-1",
        "item_list": [{"type": weixin.ITEM_VOICE, "voice_item": {"text": "recognized speech"}}],
    })

    assert len(captured) == 1
    assert captured[0].text == "[Voice transcription provided by Weixin]\nrecognized speech"
    adapter._download_media.assert_not_called()
