"""Regression tests for structured internal gateway notice metadata."""

import pytest

from gateway.config import Platform
from gateway.run import (
    _internal_notice_metadata,
    _non_conversational_metadata,
    _send_or_update_status_coro,
)


def test_non_conversational_metadata_leaves_bluebubbles_actionable():
    metadata = _non_conversational_metadata(
        {"thread_id": "thread-1"},
        platform=Platform.BLUEBUBBLES,
    )

    assert metadata == {"thread_id": "thread-1"}


def test_internal_notice_metadata_marks_bluebubbles_suppressible():
    metadata = _internal_notice_metadata(
        {"thread_id": "thread-1"},
        platform=Platform.BLUEBUBBLES,
    )

    assert metadata == {
        "thread_id": "thread-1",
        "_interim_send": True,
        "internal_notice": True,
    }


def test_non_conversational_metadata_preserves_discord_marker():
    metadata = _non_conversational_metadata({}, platform=Platform.DISCORD)

    assert metadata == {"non_conversational": True}


def test_internal_notice_metadata_marks_discord_both_ways():
    metadata = _internal_notice_metadata({}, platform=Platform.DISCORD)

    assert metadata == {
        "_interim_send": True,
        "internal_notice": True,
        "non_conversational": True,
    }


@pytest.mark.asyncio
async def test_status_delivery_marks_notice_internal():
    class Adapter:
        def __init__(self):
            self.metadata = None

        async def send(self, chat_id, content, metadata=None):
            self.metadata = metadata
            return "ok"

    adapter = Adapter()
    result = await _send_or_update_status_coro(
        adapter,
        "chat-1",
        "compaction",
        "Compacting context",
        {"thread_id": "thread-1"},
    )

    assert result == "ok"
    assert adapter.metadata == {
        "thread_id": "thread-1",
        "internal_notice": True,
    }
