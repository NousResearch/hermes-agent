"""Cross-platform tests for inbound split-message batching."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


class _DummyTask:
    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        return None


def _capture_task(coro):
    coro.close()
    return _DummyTask()


def _make_discord_adapter():
    from gateway.platforms.discord import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True))
    adapter.handle_message = AsyncMock()
    return adapter


def _make_matrix_adapter():
    from gateway.platforms.matrix import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="matrix-token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )
    adapter.handle_message = AsyncMock()
    return adapter


def _make_wecom_adapter():
    from gateway.platforms.wecom import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter.handle_message = AsyncMock()
    return adapter


def _make_event(platform: Platform, *, text: str, reply_to: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=platform,
            chat_id="chat-1",
            chat_name="Test Chat",
            chat_type="group",
            user_id="user-1",
            user_name="User One",
        ),
        message_id=f"msg-{len(text)}",
        reply_to_message_id=reply_to,
    )


@pytest.mark.parametrize(
    ("factory", "module_path", "platform"),
    [
        (_make_discord_adapter, "gateway.platforms.discord", Platform.DISCORD),
        (_make_matrix_adapter, "gateway.platforms.matrix", Platform.MATRIX),
        (_make_wecom_adapter, "gateway.platforms.wecom", Platform.WECOM),
    ],
)
@pytest.mark.asyncio
async def test_short_text_dispatches_immediately(factory, module_path, platform):
    adapter = factory()
    event = _make_event(platform, text="hello")

    with patch(f"{module_path}.asyncio.create_task", side_effect=_capture_task):
        await adapter._handle_text_event(event)

    adapter.handle_message.assert_awaited_once_with(event)
    assert adapter._pending_text_batches == {}


@pytest.mark.parametrize(
    ("factory", "module_path", "platform"),
    [
        (_make_discord_adapter, "gateway.platforms.discord", Platform.DISCORD),
        (_make_matrix_adapter, "gateway.platforms.matrix", Platform.MATRIX),
        (_make_wecom_adapter, "gateway.platforms.wecom", Platform.WECOM),
    ],
)
@pytest.mark.asyncio
async def test_near_limit_text_batches_and_merges_tail(factory, module_path, platform):
    adapter = factory()
    head = _make_event(platform, text="x" * adapter._SPLIT_THRESHOLD)
    tail = _make_event(platform, text="tail")
    head_text = head.text
    tail_text = tail.text

    with patch(f"{module_path}.asyncio.create_task", side_effect=_capture_task):
        await adapter._handle_text_event(head)
        await adapter._handle_text_event(tail)

    adapter.handle_message.assert_not_awaited()
    assert len(adapter._pending_text_batches) == 1
    pending = next(iter(adapter._pending_text_batches.values()))
    assert pending.text == f"{head_text}\n{tail_text}"
    assert getattr(pending, "_last_chunk_len") == len(tail_text)


@pytest.mark.parametrize(
    ("factory", "module_path", "platform", "use_split_delay"),
    [
        (_make_discord_adapter, "gateway.platforms.discord", Platform.DISCORD, True),
        (_make_matrix_adapter, "gateway.platforms.matrix", Platform.MATRIX, True),
        (_make_wecom_adapter, "gateway.platforms.wecom", Platform.WECOM, True),
        (_make_discord_adapter, "gateway.platforms.discord", Platform.DISCORD, False),
        (_make_matrix_adapter, "gateway.platforms.matrix", Platform.MATRIX, False),
        (_make_wecom_adapter, "gateway.platforms.wecom", Platform.WECOM, False),
    ],
)
@pytest.mark.asyncio
async def test_flush_uses_expected_delay(factory, module_path, platform, use_split_delay):
    adapter = factory()
    text = "x" * adapter._SPLIT_THRESHOLD if use_split_delay else "tail"
    event = _make_event(platform, text=text)
    event._last_chunk_len = len(text)  # type: ignore[attr-defined]
    key = adapter._text_batch_key(event)
    adapter._pending_text_batches[key] = event

    observed_delays = []

    async def _sleep(delay):
        observed_delays.append(delay)
        return None

    with patch(f"{module_path}.asyncio.sleep", side_effect=_sleep):
        await adapter._flush_text_batch(key)

    expected = (
        adapter._text_batch_split_delay_seconds
        if use_split_delay
        else adapter._text_batch_delay_seconds
    )
    assert observed_delays == [expected]
    adapter.handle_message.assert_awaited_once()
    assert key not in adapter._pending_text_batches
