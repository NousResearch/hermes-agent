"""Tests for the session race guard that prevents concurrent agent runs.

The sentinel-based guard ensures that when _handle_message passes the
"is an agent already running?" check and proceeds to the slow async
setup path (vision enrichment, STT, hooks, session hygiene), a second
message for the same session is correctly recognized as "already running"
and routed through the interrupt/queue path instead of spawning a
duplicate agent.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, build_session_key


class _FakeAdapter:
    """Minimal adapter stub for testing."""

    def __init__(self):
        self._pending_messages = {}

    async def send(self, chat_id, text, **kwargs):
        pass


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    return runner


def _make_event(text="hello", chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


# ------------------------------------------------------------------
# Test 1: Sentinel is placed before _handle_message_with_agent runs
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sentinel_placed_before_agent_setup():
    """After passing the 'not running' guard, the sentinel must be
    written into _running_agents *before* any await, so that a
    concurrent message sees the session as occupied."""
    runner = _make_runner()
    event = _make_event()
    session_key = build_session_key(event.source)

    # Patch _handle_message_with_agent to capture state at entry
    sentinel_was_set = False

    async def mock_inner(self_inner, ev, src, qk):
        nonlocal sentinel_was_set
        sentinel_was_set = runner._running_agents.get(qk) is _AGENT_PENDING_SENTINEL
        return "ok"

    with patch.object(GatewayRunner, "_handle_message_with_agent", mock_inner):
        await runner._handle_message(event)

    assert sentinel_was_set, (
        "Sentinel must be in _running_agents when _handle_message_with_agent starts"
    )


# ------------------------------------------------------------------
# Test 2: Sentinel is cleaned up after _handle_message_with_agent
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sentinel_cleaned_up_after_handler_returns():
    """If _handle_message_with_agent returns normally, the sentinel
    must be removed so the session is not permanently locked."""
    runner = _make_runner()
    event = _make_event()
    session_key = build_session_key(event.source)

    async def mock_inner(self_inner, ev, src, qk):
        return "ok"

    with patch.object(GatewayRunner, "_handle_message_with_agent", mock_inner):
        await runner._handle_message(event)

    assert session_key not in runner._running_agents, (
        "Sentinel must be removed after handler completes"
    )


# ------------------------------------------------------------------
# Test 3: Sentinel cleaned up on exception
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sentinel_cleaned_up_on_exception():
    """If _handle_message_with_agent raises, the sentinel must still
    be cleaned up so the session is not permanently locked."""
    runner = _make_runner()
    event = _make_event()
    session_key = build_session_key(event.source)

    async def mock_inner(self_inner, ev, src, qk):
        raise RuntimeError("boom")

    with patch.object(GatewayRunner, "_handle_message_with_agent", mock_inner):
        with pytest.raises(RuntimeError, match="boom"):
            await runner._handle_message(event)

    assert session_key not in runner._running_agents, (
        "Sentinel must be removed even if handler raises"
    )


# ------------------------------------------------------------------
# Test 4: Second message during sentinel sees "already running"
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_second_message_during_sentinel_queued_not_duplicate():
    """While the sentinel is set (agent setup in progress), a second
    message for the same session must hit the 'already running' branch
    and be queued — not start a second agent."""
    runner = _make_runner()
    event1 = _make_event(text="first message")
    event2 = _make_event(text="second message")
    session_key = build_session_key(event1.source)

    barrier = asyncio.Event()

    async def slow_inner(self_inner, ev, src, qk):
        # Simulate slow setup — wait until test tells us to proceed
        await barrier.wait()
        return "ok"

    with patch.object(GatewayRunner, "_handle_message_with_agent", slow_inner):
        # Start first message (will block at barrier)
        task1 = asyncio.create_task(runner._handle_message(event1))
        # Yield so task1 enters slow_inner and sentinel is set
        await asyncio.sleep(0)

        # Verify sentinel is set
        assert runner._running_agents.get(session_key) is _AGENT_PENDING_SENTINEL

        # Second message should see "already running" and be queued
        result2 = await runner._handle_message(event2)
        assert result2 is None, "Second message should return None (queued)"

        # The second message should have been queued in adapter pending
        adapter = runner.adapters[Platform.TELEGRAM]
        assert session_key in adapter._pending_messages, (
            "Second message should be queued as pending"
        )
        assert adapter._pending_messages[session_key] is event2

        # Let first message complete
        barrier.set()
        await task1


# ------------------------------------------------------------------
# Test 5: Sentinel not placed for command messages
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_command_messages_do_not_leave_sentinel():
    """Slash commands (/help, /status, etc.) return early from
    _handle_message.  They must NOT leave a sentinel behind."""
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"
    )
    event = MessageEvent(
        text="/help", message_type=MessageType.TEXT, source=source
    )
    session_key = build_session_key(source)

    # Mock the help handler to avoid needing full runner setup
    runner._handle_help_command = AsyncMock(return_value="Help text")
    # Need hooks for command emission
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    await runner._handle_message(event)

    assert session_key not in runner._running_agents, (
        "Command handlers must not leave sentinel in _running_agents"
    )
