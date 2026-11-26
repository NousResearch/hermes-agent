"""A synthetic Matrix thread root is only ever an event that can root a thread.

Auto-threading roots a new thread on the triggering event. Synapse refuses an ``m.thread``
whose parent already carries a relation ("Cannot start threads from an event with a
relation", 400), so threading the answer to such an event lost the answer.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    adapter._require_mention = False
    adapter._dm_auto_thread = True
    return adapter


async def _thread_id(relates_to):
    ctx = await _adapter()._resolve_message_context(
        room_id="!dm:ex", sender="@alice:ex", event_id="$ev1", body="hello",
        source_content={"body": "hello", "m.relates_to": relates_to}, relates_to=relates_to)
    assert ctx is not None
    return ctx[3]


@pytest.mark.asyncio
async def test_an_event_with_a_relation_is_not_made_a_thread_root():
    assert await _thread_id({"rel_type": "m.reference", "event_id": "$other"}) is None


@pytest.mark.asyncio
async def test_a_plain_reply_still_roots_its_own_thread():
    """``m.in_reply_to`` is not a relation to Synapse (no ``rel_type``), so it may root one."""
    assert await _thread_id({"m.in_reply_to": {"event_id": "$earlier"}}) == "$ev1"


@pytest.mark.asyncio
async def test_a_real_thread_root_is_kept():
    assert await _thread_id({"rel_type": "m.thread", "event_id": "$root"}) == "$root"
