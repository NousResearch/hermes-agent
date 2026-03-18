"""Tests for gateway approval staleness fix (#1888).

When a pending dangerous-command approval exists, sending a message that is NOT
an approval/denial keyword should discard the stale approval so that a later
"yes" for something unrelated does not accidentally approve the command.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:c1:u1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._prefill_messages = None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Sure, here you go.",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="tester",
            chat_type="dm",
        ),
        message_id="m1",
    )


def _session_key(runner):
    """Return the session key the runner would compute for our test source."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    return runner._session_key_for_source(source)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unrelated_message_clears_pending_approval():
    """Sending a non-approval message discards the stale pending approval."""
    runner = _make_runner()
    key = _session_key(runner)
    runner._pending_approvals[key] = {
        "command": "rm -rf /",
        "pattern_key": "recursive_delete",
        "description": "Recursive delete",
    }

    # User sends an unrelated message (not yes/no/show)
    await runner._handle_message(_make_event("Can you help me with something else?"))

    # The pending approval should have been discarded
    assert key not in runner._pending_approvals


@pytest.mark.asyncio
async def test_yes_still_approves_when_pending():
    """Normal flow: 'yes' still works when a pending approval exists."""
    runner = _make_runner()
    key = _session_key(runner)
    runner._pending_approvals[key] = {
        "command": "echo hello",
        "pattern_key": "some_pattern",
        "description": "Some command",
    }

    with patch("tools.terminal_tool.terminal_tool", return_value="hello"):
        result = await runner._handle_message(_make_event("yes"))

    assert "approved" in result.lower() or "hello" in result
    assert key not in runner._pending_approvals


@pytest.mark.asyncio
async def test_no_still_denies_when_pending():
    """Normal flow: 'no' correctly denies and clears the pending approval."""
    runner = _make_runner()
    key = _session_key(runner)
    runner._pending_approvals[key] = {
        "command": "rm -rf /",
        "pattern_key": "recursive_delete",
        "description": "Recursive delete",
    }

    result = await runner._handle_message(_make_event("no"))

    assert "denied" in result.lower()
    assert key not in runner._pending_approvals


@pytest.mark.asyncio
async def test_show_preserves_pending_approval():
    """'show' displays the command without consuming the approval."""
    runner = _make_runner()
    key = _session_key(runner)
    runner._pending_approvals[key] = {
        "command": "rm -rf /important",
        "pattern_key": "recursive_delete",
        "description": "Recursive delete",
    }

    result = await runner._handle_message(_make_event("show"))

    assert "rm -rf /important" in result
    # Approval should still be pending after "show"
    assert key in runner._pending_approvals


@pytest.mark.asyncio
async def test_sequential_unrelated_then_yes_is_safe():
    """After an unrelated message clears the approval, a later 'yes' is safe."""
    runner = _make_runner()
    key = _session_key(runner)
    runner._pending_approvals[key] = {
        "command": "rm -rf /",
        "pattern_key": "recursive_delete",
        "description": "Recursive delete",
    }

    # Step 1: send unrelated message — clears stale approval
    await runner._handle_message(_make_event("What's the weather?"))
    assert key not in runner._pending_approvals

    # Step 2: send "yes" — should go through normal processing, not trigger approval
    result = await runner._handle_message(_make_event("yes"))

    # "yes" should NOT have triggered approval execution (no pending approval)
    # It should have gone through normal agent processing
    assert "approved" not in (result or "").lower() or result is None
