"""Tests for Matrix reaction forwarding to agent session.

Covers:
- Non-approval reactions forwarded as synthetic messages
- Approval reactions NOT forwarded
- forward_reactions=False disables forwarding
- Room allowlist respected for group chats
- Self-sender reactions ignored before forwarding
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(*, forward_reactions=True, allowed_rooms=None):
    """Create a MatrixAdapter with test attributes (bypass __init__)."""
    from gateway.platforms.matrix import MatrixAdapter

    adapter = MatrixAdapter.__new__(MatrixAdapter)
    adapter._forward_reactions = forward_reactions
    adapter._allowed_rooms = allowed_rooms or set()
    adapter._approval_prompts_by_event = {}
    adapter._approval_reaction_map = {}
    adapter._approval_prompt_by_session = {}
    adapter._reactions_enabled = True
    adapter._require_mention = True
    adapter._free_rooms = set()
    adapter._allowed_user_ids = set()
    adapter._threads = MagicMock()
    adapter._threads.__contains__ = MagicMock(return_value=False)
    adapter.platform = MagicMock()
    adapter.platform.value = "matrix"
    adapter.config = MagicMock()
    adapter.config.extra = {}
    return adapter


def _make_reaction_event(
    sender="@user:matrix.org",
    event_id="$reaction_event",
    room_id="!room:matrix.org",
    reacts_to="$target_event",
    key="👍",
):
    """Create a mock Matrix reaction event."""
    event = MagicMock()
    event.sender = sender
    event.event_id = event_id
    event.room_id = room_id
    event.content = {
        "m.relates_to": {
            "rel_type": "m.annotation",
            "event_id": reacts_to,
            "key": key,
        }
    }
    return event


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_approval_reaction_forwarded():
    """A non-approval reaction should be forwarded to the agent session."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Test User")
    adapter.build_source = MagicMock(return_value={"chat_id": "!room:matrix.org"})
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event()
    await adapter._on_reaction(event)

    adapter.handle_message.assert_awaited_once()
    msg_event = adapter.handle_message.call_args[0][0]
    assert "👍" in msg_event.text
    assert "[reaction]" in msg_event.text
    assert msg_event.reply_to_message_id == "$target_event"
    assert msg_event.message_type.value == "text"


@pytest.mark.asyncio
async def test_approval_reaction_not_forwarded():
    """A reaction that resolves an approval prompt should NOT be forwarded."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter.handle_message = AsyncMock()

    # Set up an approval prompt for the target event.
    prompt = MagicMock()
    prompt.resolved = False
    prompt.chat_id = "!room:matrix.org"
    prompt.session_key = "test-session"
    adapter._approval_prompts_by_event = {"$target_event": prompt}
    adapter._approval_reaction_map = {"👍": "approve"}
    adapter._allowed_user_ids = {"@user:matrix.org"}

    event = _make_reaction_event()
    with patch("gateway.platforms.matrix.os.getenv", return_value=""):
        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            await adapter._on_reaction(event)

    # handle_message should NOT be called for approval reactions.
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_forwarding_disabled():
    """When forward_reactions=False, non-approval reactions are dropped."""
    adapter = _make_adapter(forward_reactions=False)
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event()
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_allowlist_respected():
    """Reactions from rooms not in the allowlist should be dropped."""
    adapter = _make_adapter(allowed_rooms={"!allowed:matrix.org"})
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event(room_id="!other:matrix.org")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_allowlist_passes():
    """Reactions from allowed rooms should be forwarded."""
    adapter = _make_adapter(allowed_rooms={"!room:matrix.org"})
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = AsyncMock(return_value="Test User")
    adapter.build_source = MagicMock(return_value={"chat_id": "!room:matrix.org"})
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event(room_id="!room:matrix.org")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_self_sender_ignored():
    """Reactions from the bot itself should be ignored."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=True)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event(sender="@bot:matrix.org")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_event_ignored():
    """Duplicate reaction events should be ignored."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=True)
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event()
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_forwarding_error_logged():
    """Errors during forwarding should be logged, not raised."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(side_effect=RuntimeError("test error"))
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event()
    # Should not raise.
    await adapter._on_reaction(event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reaction_uses_dm_detection():
    """DM detection should be called for reaction forwarding."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Test User")
    adapter.build_source = MagicMock(return_value={"chat_id": "!room:matrix.org"})
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event()
    await adapter._on_reaction(event)

    adapter._is_dm_room.assert_awaited_once_with("!room:matrix.org")
    # For DMs, build_source should use chat_type="dm".
    adapter.build_source.assert_called_once()
    call_kwargs = adapter.build_source.call_args[1]
    assert call_kwargs["chat_type"] == "dm"


@pytest.mark.asyncio
async def test_reaction_group_chat_type():
    """Group reactions should use chat_type='group'."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = AsyncMock(return_value="Test User")
    adapter.build_source = MagicMock(return_value={"chat_id": "!room:matrix.org"})
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event()
    await adapter._on_reaction(event)

    call_kwargs = adapter.build_source.call_args[1]
    assert call_kwargs["chat_type"] == "group"
