"""Gateway boundary tests for RAM-only volatile user context."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.turn_context import VOLATILE_USER_CONTEXT_REPLAY_ID_KEY
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _build_replay_entry
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )


def test_replay_entry_keeps_only_coordinate_free_platform_identity():
    entry = _build_replay_entry(
        "user",
        "where am I?",
        {
            "role": "user",
            "content": "where am I?",
            "message_id": "telegram-42",
        },
    )
    assert entry[VOLATILE_USER_CONTEXT_REPLAY_ID_KEY] == "telegram-42"
    assert "latitude" not in repr(entry).lower()


@pytest.mark.asyncio
async def test_recursive_queued_turn_resolves_its_ref_at_its_own_boundary():
    runner = object.__new__(GatewayRunner)
    source = _source()
    adapter = SimpleNamespace(
        _active_sessions={},
        send_typing=AsyncMock(),
    )
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="next-session-key")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value="find coffee nearby"
    )
    runner._reply_anchor_for_event = MagicMock(return_value="reply-2")
    runner._resolve_event_volatile_user_context = MagicMock(
        return_value="[Background Telegram location context]\nLatitude: 1\nLongitude: 2"
    )
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={"final_response": "next", "messages": [], "history_offset": 2}
    )

    opaque_ref = object()
    pending_event = MessageEvent(
        text="find coffee nearby",
        message_type=MessageType.TEXT,
        source=source,
        message_id="telegram-2",
        ephemeral_context_ref=opaque_ref,
    )
    turn_ctx = TurnContext(
        source=source,
        session_id="session-1",
        session_key="session-key",
        run_generation=3,
        history=[{"role": "user", "content": "first"}],
        context_prompt="context",
        _interrupt_depth=0,
        _status_thread_metadata=None,
    )

    await runner._run_agent_queued_followup(
        turn_ctx,
        adapter,
        pending_event.text,
        pending_event,
        "old response",
        {"interrupted": True, "messages": turn_ctx.history, "history_offset": 1},
        None,
    )

    runner._resolve_event_volatile_user_context.assert_called_once_with(pending_event)
    kwargs = runner._run_agent.await_args.kwargs
    assert kwargs["inbound_message_id"] == "telegram-2"
    assert kwargs["volatile_user_context"].endswith("Longitude: 2")
    assert kwargs["message"] == "find coffee nearby"
