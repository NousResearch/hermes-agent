"""Group-chat mention-gating tests for WechatyAdapter."""
from __future__ import annotations

from typing import List

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent
from plugins.platforms.wechaty.adapter import WechatyAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch, extra: dict | None = None) -> WechatyAdapter:
    monkeypatch.setenv("WECHATY_PUPPET", "wechaty-puppet-wechat4u")
    monkeypatch.delenv("WECHATY_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("WECHATY_MENTION_PATTERNS", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra=extra or {})
    return WechatyAdapter(cfg)


def _group_event(text: str, *, mention_self: bool = False) -> dict:
    return {
        "messageId": f"grp-{abs(hash(text))}",
        "chatId": "room:group-1",
        "chatType": "group",
        "chatName": "Test Group",
        "senderId": "contact:sender-1",
        "senderName": "Alice",
        "text": text,
        "mentionSelf": mention_self,
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _dm_event(text: str) -> dict:
    return {
        "messageId": f"dm-{abs(hash(text))}",
        "chatId": "contact:user-1",
        "chatType": "dm",
        "chatName": "Bob",
        "senderId": "contact:user-1",
        "senderName": "Bob",
        "text": text,
        "mentionSelf": False,
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _capture(adapter: WechatyAdapter, monkeypatch: pytest.MonkeyPatch) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def test_require_mention_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter.require_mention is True


@pytest.mark.asyncio
async def test_group_message_dropped_without_mention(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_message(_group_event("just chatting"))
    assert captured == []


@pytest.mark.asyncio
async def test_group_message_passes_when_mention_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_message(
        _group_event("@hermes what is 2+2?", mention_self=True)
    )
    assert len(captured) == 1
    assert captured[0].text == "@hermes what is 2+2?"


@pytest.mark.asyncio
async def test_group_message_passes_wake_word_and_strips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_message(_group_event("@hermes, summarize this"))
    assert len(captured) == 1
    assert captured[0].text == "summarize this"


@pytest.mark.asyncio
async def test_dm_never_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_message(_dm_event("hello without mention"))
    assert len(captured) == 1
    assert captured[0].text == "hello without mention"
