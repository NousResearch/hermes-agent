"""Per-platform typing-indicator toggle (PlatformConfig.typing_indicator).

The "typing…" / "is thinking…" status bubble is driven by the generic
``_keep_typing`` refresh loop that ``_process_message_background`` spawns for
every inbound message on every platform.  ``typing_indicator`` (default True)
gates that spawn: when False, the loop is never started, so ``send_typing``
is never called and no status indicator is shown — while message delivery is
otherwise unchanged.

These are behavioral tests against the real dispatch path, not snapshots.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
)
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter(typing_indicator: bool) -> _StubAdapter:
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=typing_indicator),
        Platform.SLACK,
    )
    # Record send_typing calls without performing any platform I/O.
    adapter.send_typing = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=None)
    # Handler returns immediately; the typing loop only fires if it was spawned.
    adapter._message_handler = AsyncMock(return_value="ok")
    return adapter


def _make_event(chat_id="C123"):
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="dm"),
    )


def _sk(chat_id="C123"):
    return build_session_key(
        SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="dm")
    )


@pytest.mark.asyncio
async def test_typing_indicator_enabled_spawns_refresh_loop():
    """Default (typing_indicator=True): the refresh loop calls send_typing."""
    adapter = _make_adapter(typing_indicator=True)

    # Real handlers take time (tool calls); yield long enough for the spawned
    # refresh loop to fire at least one send_typing before delivery completes.
    async def _slow_handler(_event):
        await asyncio.sleep(0.05)
        return "ok"

    adapter._message_handler = _slow_handler
    event = _make_event()
    adapter._active_sessions[_sk()] = asyncio.Event()

    await adapter._process_message_background(event, _sk())

    assert adapter.send_typing.await_count >= 1


@pytest.mark.asyncio
async def test_typing_indicator_disabled_never_calls_send_typing():
    """typing_indicator=False: the loop is never spawned, send_typing unused."""
    adapter = _make_adapter(typing_indicator=False)
    event = _make_event()
    adapter._active_sessions[_sk()] = asyncio.Event()

    await adapter._process_message_background(event, _sk())

    adapter.send_typing.assert_not_awaited()
    # Delivery still happened — disabling typing must not suppress the reply.
    adapter._send_with_retry.assert_awaited()
