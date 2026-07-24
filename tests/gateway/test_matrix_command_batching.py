"""Regression tests: Matrix long commands must be batched, not dispatched immediately.

Matrix uses ``!command`` aliases that normalize to ``/command``.  Before the
fix, any normalized command (MessageType.COMMAND) was dispatched immediately,
bypassing the text-batching buffer.  Long commands near the client split
threshold should still be batched so that continuation chunks from the Matrix
client's own split arrive as a single aggregated message.

Issue #58559.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


ROOM_ID = "!room:example"
USER_ID = "@user:example"


def _source(message_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id=ROOM_ID,
        chat_name="Element X DM",
        chat_type="dm",
        user_id=USER_ID,
        user_name="User",
        message_id=message_id,
        scope_id="matrix.example",
    )


def _make_adapter(*, split_threshold: int = 3900, batch_delay: float = 0.02):
    """Create a minimal MatrixAdapter for testing command batching."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter.config = SimpleNamespace(
        extra={
            "group_sessions_per_user": True,
            "thread_sessions_per_user": False,
        }
    )
    adapter._text_batch_delay_seconds = batch_delay
    adapter._text_batch_split_delay_seconds = batch_delay
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._SPLIT_THRESHOLD = split_threshold
    return adapter


@pytest.mark.asyncio
async def test_long_matrix_command_is_batched():
    """A Matrix ``!command`` whose normalized body exceeds ``_SPLIT_THRESHOLD``
    must be held in the text batch, not dispatched immediately."""
    adapter = _make_adapter()

    async def resolve_context(room_id, sender, event_id, body, source_content, relates_to):
        return body, True, "dm", None, "User", _source(event_id)

    captured: list[MessageEvent] = []

    async def capture(message: MessageEvent):
        captured.append(message)

    adapter._resolve_message_context = resolve_context
    adapter.handle_message = capture

    # First chunk: long command (>= _SPLIT_THRESHOLD)
    await adapter._handle_text_message(
        ROOM_ID, USER_ID, "$long-1", 0,
        {"body": "!queue " + ("x" * 3900)}, {},
    )
    # Second chunk: continuation from client split
    await adapter._handle_text_message(
        ROOM_ID, USER_ID, "$long-2", 0,
        {"body": "tail from client split"}, {},
    )
    await asyncio.sleep(0.08)

    assert len(captured) == 1, (
        f"Expected 1 batched message, got {len(captured)}: "
        f"{[m.text[:60] for m in captured]}"
    )
    assert "tail from client split" in captured[0].text


@pytest.mark.asyncio
async def test_short_matrix_command_dispatches_immediately():
    """A short Matrix ``!command`` (well below ``_SPLIT_THRESHOLD``) must
    dispatch immediately, not be held in the text batch."""
    adapter = _make_adapter()

    async def resolve_context(room_id, sender, event_id, body, source_content, relates_to):
        return body, True, "dm", None, "User", _source(event_id)

    captured: list[MessageEvent] = []

    async def capture(message: MessageEvent):
        captured.append(message)

    adapter._resolve_message_context = resolve_context
    adapter.handle_message = capture

    # Short command
    await adapter._handle_text_message(
        ROOM_ID, USER_ID, "$short-1", 0,
        {"body": "!stop"}, {},
    )
    # Should dispatch immediately without waiting for batch flush
    assert len(captured) == 1
    assert captured[0].text == "/stop"
    assert captured[0].message_type == MessageType.COMMAND
