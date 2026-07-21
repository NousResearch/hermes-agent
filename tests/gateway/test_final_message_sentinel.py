"""Regression tests for Telegram/Discord standalone final-message sentinels."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.final_sentinel import FINAL_MESSAGE_SENTINEL, strip_trailing_final_sentinel
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    _sentinel_sends: list[tuple[str, dict]]

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter(platform: Platform):
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), platform)
    sends: list[tuple[str, dict]] = []

    async def _send_with_retry(chat_id, content, **kwargs):
        sends.append((content, kwargs))
        return SendResult(success=True, message_id=f"m{len(sends)}")

    adapter._send_with_retry = AsyncMock(side_effect=_send_with_retry)
    adapter._sentinel_sends = sends
    return adapter


def _make_event(platform: Platform, text: str = "hello", *, internal: bool = False):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id="42", chat_type="dm"),
        internal=internal,
    )


def _session_key(platform: Platform):
    return build_session_key(SessionSource(platform=platform, chat_id="42", chat_type="dm"))


async def _run_turn(adapter, event, response="main response"):
    async def handler(_event):
        return response

    adapter._message_handler = handler
    await adapter.handle_message(event)
    key = _session_key(event.source.platform)
    for _ in range(100):
        if key not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_telegram_normal_response_gets_standalone_complete_once():
    adapter = _make_adapter(Platform.TELEGRAM)

    await _run_turn(adapter, _make_event(Platform.TELEGRAM), "main response")

    assert [content for content, _ in adapter._sentinel_sends] == [
        "main response",
        FINAL_MESSAGE_SENTINEL,
    ]


@pytest.mark.asyncio
async def test_discord_normal_response_gets_standalone_complete_once():
    adapter = _make_adapter(Platform.DISCORD)

    await _run_turn(adapter, _make_event(Platform.DISCORD), "main response")

    assert [content for content, _ in adapter._sentinel_sends] == [
        "main response",
        FINAL_MESSAGE_SENTINEL,
    ]


@pytest.mark.asyncio
async def test_non_telegram_discord_platform_gets_no_sentinel():
    adapter = _make_adapter(Platform.SLACK)

    await _run_turn(adapter, _make_event(Platform.SLACK), "main response")

    assert [content for content, _ in adapter._sentinel_sends] == ["main response"]


@pytest.mark.asyncio
async def test_slash_command_gets_no_sentinel():
    adapter = _make_adapter(Platform.TELEGRAM)

    await _run_turn(adapter, _make_event(Platform.TELEGRAM, text="/status"), "status response")

    assert [content for content, _ in adapter._sentinel_sends] == ["status response"]


@pytest.mark.asyncio
async def test_internal_event_gets_no_sentinel():
    adapter = _make_adapter(Platform.TELEGRAM)

    await _run_turn(
        adapter,
        _make_event(Platform.TELEGRAM, text="background", internal=True),
        "background response",
    )

    assert [content for content, _ in adapter._sentinel_sends] == ["background response"]


@pytest.mark.asyncio
async def test_model_trailing_complete_is_stripped_and_sent_once_standalone():
    adapter = _make_adapter(Platform.TELEGRAM)

    await _run_turn(adapter, _make_event(Platform.TELEGRAM), "main response\n\nCOMPLETE")

    assert [content for content, _ in adapter._sentinel_sends] == [
        "main response",
        FINAL_MESSAGE_SENTINEL,
    ]


def test_strip_trailing_final_sentinel_detects_only_standalone_marker():
    assert strip_trailing_final_sentinel("body\n\nCOMPLETE") == ("body", True)
    assert strip_trailing_final_sentinel("body COMPLETE") == ("body COMPLETE", False)
