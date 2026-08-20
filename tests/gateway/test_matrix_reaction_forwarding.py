"""Tests for Matrix reaction forwarding to agent session.

Covers:
- Non-approval reactions forwarded as synthetic messages
- Approval reactions NOT forwarded
- Model-picker / choice-picker reactions NOT forwarded
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
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter.__new__(MatrixAdapter)
    adapter._forward_reactions = forward_reactions
    adapter._allowed_rooms = allowed_rooms or set()
    adapter._approval_prompts_by_event = {}
    adapter._approval_reaction_map = {}
    adapter._approval_prompt_by_session = {}
    adapter._model_picker_prompts_by_event = {}
    adapter._choice_picker_prompts_by_event = {}
    adapter._reactions_enabled = True
    adapter._require_mention = True
    adapter._free_rooms = set()
    adapter._allowed_user_ids = set()
    # Intake-filter mirrors the normal room-message path (bridge/system
    # senders, ignore patterns, room authorization) — default them open.
    adapter._is_system_or_bridge_sender = MagicMock(return_value=False)
    adapter._matches_ignored_user_pattern = MagicMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._client = None
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
    with patch("plugins.platforms.matrix.adapter.os.getenv", return_value=""):
        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            await adapter._on_reaction(event)

    # handle_message should NOT be called for approval reactions.
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_picker_reaction_not_forwarded():
    """A reaction that resolves a model-picker prompt should NOT be forwarded."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter.handle_message = AsyncMock()

    model_prompt = MagicMock()
    model_prompt.resolved = False
    model_prompt.chat_id = "!room:matrix.org"
    model_prompt.expires_at = None
    model_prompt.requester_user_id = None
    model_prompt.choices = {"🤖": ("model-x", "provider-y")}
    model_prompt.on_model_selected = AsyncMock(return_value="switched")
    adapter._model_picker_prompts_by_event = {"$target_event": model_prompt}
    adapter._allowed_user_ids = {"@user:matrix.org"}
    adapter.send = AsyncMock()

    event = _make_reaction_event(key="🤖")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()
    model_prompt.on_model_selected.assert_awaited_once()


@pytest.mark.asyncio
async def test_choice_picker_reaction_not_forwarded():
    """A reaction that resolves a choice-picker prompt should NOT be forwarded."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter.handle_message = AsyncMock()

    choice_prompt = MagicMock()
    choice_prompt.resolved = False
    choice_prompt.chat_id = "!room:matrix.org"
    choice_prompt.expires_at = None
    choice_prompt.requester_user_id = None
    choice_prompt.choices = {"1️⃣": "option-one"}
    choice_prompt.on_choice_selected = AsyncMock(return_value="picked")
    adapter._choice_picker_prompts_by_event = {"$target_event": choice_prompt}
    adapter._allowed_user_ids = {"@user:matrix.org"}
    adapter.send = AsyncMock()

    event = _make_reaction_event(key="1️⃣")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()
    choice_prompt.on_choice_selected.assert_awaited_once()


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
    """Reactions from rooms not authorized by the intake filter should be dropped."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=False)
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event(room_id="!other:matrix.org")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_allowlist_passes():
    """Reactions from allowed rooms should be forwarded."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = AsyncMock(return_value="Test User")
    adapter.build_source = MagicMock(return_value={"chat_id": "!room:matrix.org"})
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event(room_id="!room:matrix.org")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_bridge_sender_reaction_ignored():
    """Reactions relayed by bridge/system senders must not reach the agent."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_system_or_bridge_sender = MagicMock(return_value=True)
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event(sender="@bridge:matrix.org")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignored_sender_pattern_reaction_dropped():
    """Senders matching a configured ignore pattern are dropped before forwarding."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._matches_ignored_user_pattern = MagicMock(return_value=True)
    adapter.handle_message = AsyncMock()

    event = _make_reaction_event(sender="@spam:matrix.org")
    await adapter._on_reaction(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_to_text_fetched_from_target_event():
    """The reacted-to event body is fetched and attached as reply_to_text."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Test User")
    adapter.build_source = MagicMock(return_value={"chat_id": "!room:matrix.org"})
    adapter.handle_message = AsyncMock()

    target_event = MagicMock()
    target_event.content = {"body": "the digest summary"}
    adapter._client = MagicMock()
    adapter._client.get_event = AsyncMock(return_value=target_event)

    event = _make_reaction_event()
    await adapter._on_reaction(event)

    adapter._client.get_event.assert_awaited_once_with("!room:matrix.org", "$target_event")
    msg_event = adapter.handle_message.call_args[0][0]
    assert msg_event.reply_to_text == "the digest summary"
    assert msg_event.reply_to_message_id == "$target_event"


@pytest.mark.asyncio
async def test_reply_to_text_fetch_failure_degrades_gracefully():
    """When the target event can't be fetched, forwarding still happens."""
    adapter = _make_adapter()
    adapter._is_self_sender = MagicMock(return_value=False)
    adapter._is_duplicate_event = MagicMock(return_value=False)
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Test User")
    adapter.build_source = MagicMock(return_value={"chat_id": "!room:matrix.org"})
    adapter.handle_message = AsyncMock()

    adapter._client = MagicMock()
    adapter._client.get_event = AsyncMock(side_effect=RuntimeError("history redacted"))

    event = _make_reaction_event()
    await adapter._on_reaction(event)

    adapter.handle_message.assert_awaited_once()
    msg_event = adapter.handle_message.call_args[0][0]
    assert msg_event.reply_to_text is None


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
