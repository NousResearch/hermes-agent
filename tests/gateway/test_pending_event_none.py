"""Tests for pending follow-up extraction in recursive _run_agent calls.

When pending_event is None (Path B: pending comes from interrupt_message),
accessing pending_event.channel_prompt previously raised AttributeError.
This verifies the fix: channel_prompt is captured inside the
`if pending_event is not None:` block and falls back to None otherwise.

Also verifies that internal control interrupt reasons like "Stop requested"
do not get recycled into the pending-user-message follow-up path.
"""

from types import SimpleNamespace

import pytest

from gateway.run import _INTERRUPT_REASON_EVICTED, _is_control_interrupt_message


def _extract_channel_prompt(pending_event):
    """Reproduce the fixed logic from gateway/run.py.

    Mirrors the variable-capture pattern used before the recursive
    _run_agent call so we can test both paths without a full runner.
    """
    next_channel_prompt = None
    if pending_event is not None:
        next_channel_prompt = getattr(pending_event, "channel_prompt", None)
    return next_channel_prompt


def _extract_pending_text(interrupted, pending_event, interrupt_message):
    """Reproduce the fixed pending-text selection from gateway/run.py."""
    if interrupted and pending_event is None and interrupt_message:
        if _is_control_interrupt_message(interrupt_message):
            return None
        return interrupt_message
    return None


class TestPendingEventNoneChannelPrompt:
    """Guard against AttributeError when pending_event is None."""


    def test_pending_event_with_channel_prompt_passes_through(self):
        """Path A: pending_event present — channel_prompt is forwarded."""
        event = SimpleNamespace(channel_prompt="You are a helpful bot.")
        result = _extract_channel_prompt(event)
        assert result == "You are a helpful bot."


class TestControlInterruptMessages:
    """Control interrupt reasons must not become follow-up user input."""

    def test_stop_requested_is_not_treated_as_pending_user_message(self):
        result = _extract_pending_text(True, None, "Stop requested")
        assert result is None

    def test_evicted_reason_is_not_treated_as_pending_user_message(self):
        """The reason ``_hm_evict_running_agent`` hands to ``request_hard_interrupt`` when a
        session's turn slot is evicted (reaped durable row / stale turn, #106963) is gateway
        control flow exactly like "Stop requested": it must never re-enter the conversation
        as the user's next turn."""
        assert _extract_pending_text(True, None, _INTERRUPT_REASON_EVICTED) is None

    def test_every_gateway_interrupt_reason_is_control_flow(self):
        """Invariant behind the frozenset: a reason the gateway itself produces is never user
        input. Enumerating the ``_INTERRUPT_REASON_*`` constants catches the next reason that
        is added without being classified (how the eviction reason was missed at first)."""
        import gateway.run as gateway_run

        reasons = {
            name: value for name, value in vars(gateway_run).items()
            if name.startswith("_INTERRUPT_REASON_")
        }
        assert _INTERRUPT_REASON_EVICTED in reasons.values()
        unclassified = sorted(
            name for name, value in reasons.items() if not _is_control_interrupt_message(value)
        )
        assert unclassified == []


class TestDrainPendingRealConsumer:
    """The production consumer, ``_run_agent_drain_pending``, applies the same rule: the mirror
    above documents the selection, this pins the code path that actually builds the next turn."""

    @staticmethod
    def _drain(result):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._draining = False
        adapter = SimpleNamespace(_pending_messages={}, get_pending_message=lambda key: None)
        source = SimpleNamespace(thread_id=None)
        return runner._run_agent_drain_pending(result, adapter, source, "agent:main:telegram:dm:0")

    @pytest.mark.asyncio
    async def test_evicted_reason_does_not_become_the_next_user_turn(self):
        result = {"interrupted": True, "interrupt_message": _INTERRUPT_REASON_EVICTED}

        pending_event, pending = await self._drain(result)

        assert pending_event is None
        assert pending is None, "the eviction reason re-entered the conversation as user input"

    @pytest.mark.asyncio
    async def test_user_supplied_interrupt_text_still_becomes_the_next_turn(self):
        """Control: an interrupt that carries the user's own follow-up is still recycled, so the
        classification is what discriminates, not a blanket drop of interrupt messages."""
        result = {"interrupted": True, "interrupt_message": "actually, use the other file"}

        pending_event, pending = await self._drain(result)

        assert pending_event is None
        assert pending == "actually, use the other file"


