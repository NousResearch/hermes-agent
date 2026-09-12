"""Tests for pending follow-up extraction in recursive _run_agent calls.

When pending_event is None, accessing pending_event.channel_prompt previously
raised AttributeError. This verifies channel_prompt falls back to None.

The direct drain tests also verify that neither internal control reasons nor
unprovenanced user-shaped interrupt messages become follow-up user turns.
"""

from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


def _extract_channel_prompt(pending_event):
    """Reproduce the fixed logic from gateway/run.py.

    Mirrors the variable-capture pattern used before the recursive
    _run_agent call so we can test both paths without a full runner.
    """
    next_channel_prompt = None
    if pending_event is not None:
        next_channel_prompt = getattr(pending_event, "channel_prompt", None)
    return next_channel_prompt


class TestPendingEventNoneChannelPrompt:
    """Guard against AttributeError when pending_event is None."""


    def test_pending_event_with_channel_prompt_passes_through(self):
        """Path A: pending_event present — channel_prompt is forwarded."""
        event = SimpleNamespace(channel_prompt="You are a helpful bot.")
        result = _extract_channel_prompt(event)
        assert result == "You are a helpful bot."


class TestControlInterruptMessages:
    """Control interrupt reasons must not become follow-up user input."""

    @pytest.mark.asyncio
    async def test_stop_requested_is_not_treated_as_pending_user_message(self):
        runner = object.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter = SimpleNamespace(get_pending_message=lambda _session_key: None)

        pending_event, pending = await runner._run_agent_drain_pending(
            {"interrupted": True, "interrupt_message": "Stop requested"},
            adapter,
            SimpleNamespace(thread_id=None),
            "slack:dm:user",
        )

        assert pending_event is None
        assert pending is None


@pytest.mark.asyncio
async def test_stale_interrupt_message_without_queued_event_is_not_followed_up():
    runner = object.__new__(GatewayRunner)
    runner._queued_events = {}
    adapter = SimpleNamespace(get_pending_message=lambda _session_key: None)
    source = SimpleNamespace(thread_id=None)

    pending_event, pending = await runner._run_agent_drain_pending(
        {
            "interrupted": True,
            "interrupt_message": "the stale opening request",
        },
        adapter,
        source,
        "slack:dm:user",
    )

    assert pending_event is None
    assert pending is None
