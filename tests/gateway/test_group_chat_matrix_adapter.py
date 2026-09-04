"""Matrix ingress guarantees used by Group Chat messaging controls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.hosted_room_messaging import messaging_event_id
from tests.gateway.test_matrix import _make_adapter


@pytest.mark.asyncio
async def test_named_two_member_dm_stamps_one_to_one_proof():
    adapter = _make_adapter()
    adapter._joined_rooms = {"!named_dm:ex.org"}
    adapter._dm_rooms = {"!named_dm:ex.org": True}
    adapter._client = MagicMock()
    adapter._client.get_state_event = AsyncMock(
        side_effect=lambda _room_id, event_type: (
            {"name": "Alice & Bot"}
            if event_type == "m.room.name"
            else (_ for _ in ()).throw(Exception("no alias"))
        )
    )
    adapter._client.state_store = MagicMock()
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@bot:ex.org", "@alice:ex.org"]
    )
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()

    identity = await adapter._resolve_room_identity("!named_dm:ex.org")
    context = await adapter._resolve_message_context(
        "!named_dm:ex.org",
        "@alice:ex.org",
        "$event",
        "hello",
        {"body": "hello"},
        {},
    )

    assert identity.chat_type == "dm"
    assert identity.joined_member_count == 2
    assert context is not None
    assert context[-1].is_one_to_one is True


@pytest.mark.asyncio
async def test_room_control_normalizes_and_keeps_matrix_event_id():
    adapter = _make_adapter()
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._require_mention = True
    adapter._free_rooms = set()
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    await adapter._handle_text_message(
        room_id="!room:example.org",
        sender="@alice:example.org",
        event_id="$matrix-command-test",
        event_ts=0.0,
        source_content={"msgtype": "m.text", "body": "!group 1 send hello"},
        relates_to={},
    )

    assert len(captured) == 1
    assert captured[0].text == "/group 1 send hello"
    assert messaging_event_id(captured[0]) == messaging_event_id(captured[0])


def test_picker_literal_at_signs_do_not_emit_matrix_mentions():
    adapter = _make_adapter()
    adapter._allow_room_mentions = True

    content = adapter._build_text_message_content(
        "🟢 1. ＠room\n"
        "🤖 Operator · alice:example.org\n"
        "💬 **＠alice:example.org**\n"
        "• **Operator (`@alice:example.org`):** ＠room status"
    )

    assert "m.mentions" not in content
