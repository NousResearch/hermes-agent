"""Per-sender text batching for the SimpleX adapter.

SimpleX batches a participant's rapid client-side message splits into one
event before dispatch (mirroring Telegram/WhatsApp/Matrix). The batch key
must isolate senders: in a SimpleX *group* ``chat_id`` is the group id while
``source.user_id`` is the member id, so a chat-only key would fuse two
members posting inside the batch window into a single event under the first
sender's identity — dropping the second member's authorship and letting their
text ride the first sender through downstream authorization.

This is the SimpleX sibling of the Matrix fix (#63056); same class of bug,
different platform, so a separate adapter and a separate PR.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_simplex_adapter():
    """Minimal SimplexAdapter wired only for the text-batching path.

    Built with ``object.__new__`` so no daemon/WebSocket state is required —
    the batching helpers only touch the two pending dicts, the flush delay,
    and ``handle_message`` (mirrors ``_make_matrix_adapter`` in
    ``test_text_batching.py``).
    """
    from plugins.platforms.simplex.adapter import SimplexAdapter

    adapter = object.__new__(SimplexAdapter)
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay = 0.1
    adapter.handle_message = AsyncMock()
    return adapter


def _group_event(text: str, user_id: str) -> MessageEvent:
    """A text event in one shared SimpleX group from a given member."""
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform("simplex"),
            chat_id="#team-group",
            chat_type="group",
            user_id=user_id,
        ),
    )


@pytest.mark.asyncio
async def test_different_senders_in_group_not_merged():
    """Two members of one group must not batch together under one identity."""
    adapter = _make_simplex_adapter()

    adapter._enqueue_text_event(_group_event("hi from alice", "alice-member-id"))
    await asyncio.sleep(0.02)
    adapter._enqueue_text_event(_group_event("hi from bob", "bob-member-id"))

    await asyncio.sleep(0.3)

    # Each sender is dispatched as its own event — never fused into one.
    assert adapter.handle_message.call_count == 2
    dispatched = {
        call.args[0].source.user_id: call.args[0].text
        for call in adapter.handle_message.call_args_list
    }
    assert dispatched == {
        "alice-member-id": "hi from alice",
        "bob-member-id": "hi from bob",
    }
    # No single dispatched event carries both senders' text.
    for text in dispatched.values():
        assert not ("alice" in text and "bob" in text)


@pytest.mark.asyncio
async def test_same_sender_split_chunks_still_batch():
    """One member's split chunks in a group still aggregate into one event."""
    adapter = _make_simplex_adapter()

    adapter._enqueue_text_event(_group_event("first part", "alice-member-id"))
    await asyncio.sleep(0.02)
    adapter._enqueue_text_event(_group_event("second part", "alice-member-id"))

    await asyncio.sleep(0.3)

    adapter.handle_message.assert_called_once()
    event = adapter.handle_message.call_args[0][0]
    assert "first part" in event.text
    assert "second part" in event.text
    assert event.source.user_id == "alice-member-id"
