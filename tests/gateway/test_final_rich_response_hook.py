"""Tests for BasePlatformAdapter final rich response hook orchestration."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class _RichHookAdapter(BasePlatformAdapter):
    def __init__(self, *, rich_result=None, platform=Platform.FEISHU):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self.rich_result = rich_result
        self.rich_calls = []
        self.sent_text = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        self.sent_text.append((chat_id, content, kwargs))
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def try_send_final_rich_response(self, **kwargs):
        self.rich_calls.append(kwargs)
        return self.rich_result


def _event():
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="chat-1",
        chat_type="dm",
    )
    return MessageEvent(
        text="make image",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _allowed_media_path(tmp_path, monkeypatch, name="image.png"):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


@pytest.mark.asyncio
async def test_final_rich_response_hook_success_skips_legacy_text_and_images(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch)
    adapter = _RichHookAdapter(rich_result=SendResult(success=True, message_id="rich"))
    adapter._message_handler = AsyncMock(return_value=f"正文\nMEDIA:{media_file}")
    adapter.send_multiple_images = AsyncMock()

    await adapter._process_message_background(_event(), build_session_key(_event().source))

    assert len(adapter.rich_calls) == 1
    call = adapter.rich_calls[0]
    assert call["text_content"] == "正文"
    assert call["media_files"] == [(str(media_file), False)]
    assert call["metadata"]["hermes_final_response"] is True
    assert adapter.sent_text == []
    adapter.send_multiple_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_rich_response_hook_none_falls_back_to_legacy_delivery(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch)
    adapter = _RichHookAdapter(rich_result=None)
    adapter._message_handler = AsyncMock(return_value=f"正文\nMEDIA:{media_file}")
    adapter.send_multiple_images = AsyncMock()

    await adapter._process_message_background(_event(), build_session_key(_event().source))

    assert len(adapter.rich_calls) == 1
    assert adapter.sent_text and adapter.sent_text[0][1] == "正文"
    adapter.send_multiple_images.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_rich_response_marker_is_feishu_only(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch)
    adapter = _RichHookAdapter(
        rich_result=SendResult(success=True, message_id="rich"),
        platform=Platform.TELEGRAM,
    )
    adapter._message_handler = AsyncMock(return_value=f"正文\nMEDIA:{media_file}")
    adapter.send_multiple_images = AsyncMock()

    await adapter._process_message_background(_event(), build_session_key(_event().source))

    assert len(adapter.rich_calls) == 1
    assert "hermes_final_response" not in adapter.rich_calls[0]["metadata"]
