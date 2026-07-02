"""Inbound dispatch and dedup tests for WechatyAdapter."""
from __future__ import annotations

import base64
import json
from typing import List
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.wechaty.adapter import WechatyAdapter, _cache_inbound_image


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> WechatyAdapter:
    monkeypatch.setenv("WECHATY_PUPPET", "wechaty-puppet-wechat4u")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return WechatyAdapter(cfg)


def _capture(adapter: WechatyAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def test_is_duplicate_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter._is_duplicate("msg-1") is False
    assert adapter._is_duplicate("msg-1") is True


@pytest.mark.asyncio
async def test_on_inbound_line_ignores_non_message_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter.handle_message = AsyncMock()

    await adapter._on_inbound_line(json.dumps({"type": "scan", "qrcode": "abc"}))
    await adapter._on_inbound_line(
        json.dumps({"type": "login", "userName": "bot", "userId": "id"})
    )
    await adapter._on_inbound_line(json.dumps({"type": "logout", "userName": "bot"}))

    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_dm_text(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_message(
        {
            "messageId": "m-1",
            "chatId": "contact:user-1",
            "chatType": "dm",
            "chatName": "Bob",
            "senderId": "contact:user-1",
            "senderName": "Bob",
            "text": "hello",
            "timestamp": "2026-05-14T19:06:32.000Z",
        }
    )

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "hello"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "m-1"
    assert event.source is not None
    assert event.source.chat_type == "dm"


@pytest.mark.asyncio
async def test_dispatch_image_attachment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    raw = b"\x89PNG\r\n\x1a\n"
    attachment = {
        "kind": "image",
        "encoding": "base64",
        "data": base64.b64encode(raw).decode(),
        "mimeType": "image/png",
    }
    path = _cache_inbound_image(attachment)
    assert path is not None

    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_message(
        {
            "messageId": "m-img",
            "chatId": "contact:user-1",
            "chatType": "dm",
            "chatName": "Bob",
            "senderId": "contact:user-1",
            "text": "",
            "attachment": attachment,
            "timestamp": "2026-05-14T19:06:32.000Z",
        }
    )

    assert len(captured) == 1
    assert captured[0].message_type == MessageType.PHOTO
    assert captured[0].media_urls
