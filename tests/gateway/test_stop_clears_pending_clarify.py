"""Regression test: /stop cancels pending clarify prompts immediately.

Incident 2026-07-19: a turn was blocked on a ``clarify`` question when the
user ``/stop``'d it.  ``_interrupt_and_clear_session`` interrupted the agent
and invalidated the generation but left the clarify entry registered in
``tools.clarify_gateway`` (cleanup only ran in the turn's ``finally``, which
a thread parked in ``wait_for_response`` had not reached).  The user's NEXT
message was then intercepted by the gateway's clarify hook and fed to the
dead turn — whose output was suppressed by the invalidated generation — so
the message was silently swallowed and the user had to double-message.

The fix: ``_interrupt_and_clear_session`` calls ``clarify_gateway
.clear_session`` for the session, cancelling every pending entry (which also
unblocks the waiting agent thread via the empty-string sentinel).
"""

from unittest.mock import MagicMock

import pytest

from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.platforms.base import Platform
from tools import clarify_gateway


SESSION_KEY = "agent:main:discord:thread:12345:12345"


def _source():
    return SessionSource(
        platform=Platform.DISCORD,
        chat_type="group",
        chat_id="12345",
        thread_id="12345",
        user_id="u1",
    )


def _bare_runner():
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agent_tasks = {}
    runner._draining_turns = {}
    runner._pending_messages = {}
    runner.session_store = MagicMock()
    runner.session_store._entries = {}
    runner._invalidate_session_run_generation = MagicMock()
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    return runner


@pytest.fixture(autouse=True)
def _clean_clarify_state():
    clarify_gateway.clear_session(SESSION_KEY)
    yield
    clarify_gateway.clear_session(SESSION_KEY)


@pytest.mark.asyncio
async def test_stop_cancels_pending_clarify():
    """A pending clarify must not survive _interrupt_and_clear_session."""
    entry = clarify_gateway.register(
        clarify_id="stopclr0001",
        session_key=SESSION_KEY,
        question="rotate now or defer?",
        choices=[],
    )
    assert clarify_gateway.get_pending_for_session(SESSION_KEY) is not None

    runner = _bare_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="stop_command",
        invalidation_reason="stop_command",
    )

    # Entry is gone: the next inbound message cannot be intercepted as a
    # clarify answer for the dead turn.
    assert clarify_gateway.get_pending_for_session(SESSION_KEY) is None
    assert not clarify_gateway.has_pending(SESSION_KEY)
    # The blocked agent thread was unblocked via the cancellation sentinel.
    assert entry.event.is_set()


@pytest.mark.asyncio
async def test_stop_without_pending_clarify_is_noop():
    """No clarify pending → stop path must not raise or log-error."""
    runner = _bare_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="stop_command",
        invalidation_reason="stop_command",
    )
    assert clarify_gateway.get_pending_for_session(SESSION_KEY) is None


@pytest.mark.asyncio
async def test_stop_clears_clarify_even_when_release_state_false():
    """The clarify cancel runs before the release_running_state branch."""
    clarify_gateway.register(
        clarify_id="stopclr0002",
        session_key=SESSION_KEY,
        question="pick one",
        choices=["a", "b"],
    )
    runner = _bare_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="new_command",
        invalidation_reason="new_command",
        release_running_state=False,
    )
    assert not clarify_gateway.has_pending(SESSION_KEY)
