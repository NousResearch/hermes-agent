"""Lifecycle lease-wait statuses must honour the interim-assistant-messages mute.

#94658: session turn-lease contention statuses ("⏳ Another Hermes process is
using this session...", "⏳ Still waiting ... (Ns)...", "Session is free;
loading the latest transcript...") are emitted by run_agent as ``lifecycle``
status callbacks. When a platform mutes interim assistant messages
(interim_assistant_messages: off), those transient statuses must be
suppressed by _prepare_gateway_status_message rather than pushed to chat.
"""

import pytest

from agent.conversation_compression import COMPACTION_DONE_STATUS
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message

# Exact lifecycle strings emitted by run_agent._emit_status during turn-lease
# contention / reclaim (run_agent.py ~line 8646 / 8735).
LEASE_WAIT_FIRST = (
    "⏳ Another Hermes process is using this session; "
    "waiting for it to finish before starting your turn..."
)
LEASE_WAIT_STILL = (
    "⏳ Still waiting for the other Hermes process on this session (5s)..."
)
SESSION_FREE = "Session is free; loading the latest transcript..."

LIFECYCLE_MESSAGES = [LEASE_WAIT_FIRST, LEASE_WAIT_STILL, SESSION_FREE]


@pytest.mark.parametrize("message", LIFECYCLE_MESSAGES)
@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SLACK])
def test_lifecycle_lease_statuses_suppressed_when_interim_muted(platform, message):
    """interim_enabled=False must swallow the lifecycle lease-wait chatter."""
    assert (
        _prepare_gateway_status_message(platform, "lifecycle", message, interim_enabled=False)
        is None
    )


@pytest.mark.parametrize("message", LIFECYCLE_MESSAGES)
def test_lifecycle_lease_statuses_flow_when_interim_enabled(message):
    """interim_enabled=True (the default) keeps lifecycle messages flowing."""
    assert (
        _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", message)
        == message
    )


def test_lifecycle_gate_is_scoped_to_lifecycle_only():
    """A NON-lifecycle event_type must not be suppressed by the interim flag."""
    platform = Platform.TELEGRAM
    # A warn event that otherwise passes the filter must keep passing even
    # with interim assistant messages muted.
    warn = "⚠️ Tool execution failed; retrying with a different approach."
    assert _prepare_gateway_status_message(platform, "warn", warn, interim_enabled=False) == warn
    # Compacted event likewise.
    assert (
        _prepare_gateway_status_message(platform, "compacted", COMPACTION_DONE_STATUS, interim_enabled=False)
        == COMPACTION_DONE_STATUS
    )


# Durable must-see lifecycle statuses that reach the gateway today and MUST
# keep flowing even when interim assistant messages are muted. These are
# emitted as ``lifecycle`` callbacks but are NOT lease-contention chatter:
# a blanket event_type=="lifecycle" gate would silently drop them — a
# regression this suite pins against.
DURABLE_LIFECYCLE_MESSAGES = [
    # Fallback-switch notice (run_agent._emit_pending_fallback_notice, ~line 1211).
    "↻ Switched to fallback: gpt-4o (openai)",
    # Terminal provider-failure notice, buffered via _buffer_status and
    # replayed via _emit_status on terminal flush.
    "❌ Connection to provider failed after 3 attempts. The provider may be "
    "experiencing issues — try again in a moment.",
]


@pytest.mark.parametrize("message", DURABLE_LIFECYCLE_MESSAGES)
@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SLACK])
def test_durable_lifecycle_statuses_flow_when_interim_muted(platform, message):
    """interim_enabled=False must NOT swallow durable lifecycle messages."""
    assert (
        _prepare_gateway_status_message(platform, "lifecycle", message, interim_enabled=False)
        == message
    )
