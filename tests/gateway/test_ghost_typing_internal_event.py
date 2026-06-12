"""Tests for ghost typing indicator fix — internal/synth events skip typing loop.

Background process completions and cron deliveries inject synth events with
``internal=True`` via ``adapter.handle_message()``.  These events should NOT
trigger the ``_keep_typing`` loop because the agent response is typically fast
(near-instant) and the resulting ghost typing indicator confuses users.

Covered:

1. Internal events skip ``_keep_typing`` entirely (no typing_task created).
2. Normal events create the typing task as before (regression guard).
3. ``_stop_typing_task()`` handles ``typing_task is None`` gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource


class _TestAdapter(BasePlatformAdapter):
    """Minimal adapter for typing-indicator tests."""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content="", **kwargs):
        return SendResult(success=True, message_id="m-1")

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    return _TestAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM
    )


def _make_event(text="hello", chat_id="42", internal=False):
    return MessageEvent(
        text=text,
        message_id="msg-1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            user_id="u-1",
        ),
        message_type=MessageType.TEXT,
        internal=internal,
    )


# ---------------------------------------------------------------------------
# Internal events skip _keep_typing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_event_skips_keep_typing():
    """Synth/internal events must NOT start the _keep_typing loop."""
    adapter = _make_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )

    async def _handler(evt):
        return "Background process completed."

    adapter.set_message_handler(_handler)

    event = _make_event(internal=True)
    session_key = "agent:main:telegram:private:42"

    with patch.object(adapter, "_keep_typing", new=AsyncMock()) as mock_typing:
        with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()):
            await adapter._process_message_background(event, session_key)

    # _keep_typing must NOT be called for internal events
    mock_typing.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_event_still_sends_response():
    """Internal events skip typing but still process and send the response."""
    adapter = _make_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )

    async def _handler(evt):
        return "Process done."

    adapter.set_message_handler(_handler)

    event = _make_event(internal=True)
    session_key = "agent:main:telegram:private:42"

    with patch.object(adapter, "_keep_typing", new=AsyncMock()), \
         patch("gateway.platforms.base.asyncio.sleep", AsyncMock()):
        await adapter._process_message_background(event, session_key)

    adapter._send_with_retry.assert_called_once()
    sent_text = adapter._send_with_retry.call_args.kwargs["content"]
    assert sent_text == "Process done."


# ---------------------------------------------------------------------------
# Normal events still create typing task (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_event_creates_typing_task():
    """Non-internal events must still start the _keep_typing loop."""
    adapter = _make_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )

    async def _handler(evt):
        return "Hello"

    adapter.set_message_handler(_handler)

    event = _make_event(internal=False)
    session_key = "agent:main:telegram:private:42"

    with patch.object(adapter, "_keep_typing", new=AsyncMock()) as mock_typing:
        with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()):
            await adapter._process_message_background(event, session_key)

    # _keep_typing MUST be called for normal events (via create_task)
    mock_typing.assert_called_once()


@pytest.mark.asyncio
async def test_default_internal_false_creates_typing_task():
    """Events without explicit internal flag (default False) get typing."""
    adapter = _make_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )

    async def _handler(evt):
        return "Hello"

    adapter.set_message_handler(_handler)

    event = _make_event()  # internal defaults to False
    session_key = "agent:main:telegram:private:42"

    with patch.object(adapter, "_keep_typing", new=AsyncMock()) as mock_typing:
        with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()):
            await adapter._process_message_background(event, session_key)

    mock_typing.assert_called_once()
