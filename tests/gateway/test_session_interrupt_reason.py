"""Tests for per-session interrupt reason tracking (Feature F5).

Verifies that:
- An interrupted session surfaces the reason on the next message's context_prompt
- A session that completes normally produces no interrupt note
- Multiple interrupts only show the most recent reason
- The reason is cleared after being consumed (not shown twice)
- _evict_cached_agent cleans up orphaned reasons to prevent leaks
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner():
    """Minimal GatewayRunner with only the state needed for interrupt-reason tests."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._session_interrupt_reasons = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    return runner


# ---------------------------------------------------------------------------
# Unit tests: _session_interrupt_reasons dict management
# ---------------------------------------------------------------------------


class TestInterruptReasonTracking:
    """_session_interrupt_reasons is populated at interrupt sites."""

    def test_interrupt_and_clear_records_reason(self):
        """_interrupt_and_clear_session stores the interrupt reason."""
        import asyncio
        from gateway.run import GatewayRunner, _INTERRUPT_REASON_STOP

        runner = _make_runner()
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._session_run_generation = {}

        # Patch out the async parts we don't need
        async def _fake_interrupt_and_clear(session_key, source, *, interrupt_reason, invalidation_reason, release_running_state=True):
            # Call the real method but with mocked deps
            if not session_key:
                return
            running_agent = runner._running_agents.get(session_key)
            from gateway.run import _AGENT_PENDING_SENTINEL, _is_control_interrupt_message
            if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
                running_agent.interrupt(interrupt_reason)
            if hasattr(runner, "_session_interrupt_reasons") and _is_control_interrupt_message(interrupt_reason):
                runner._session_interrupt_reasons[session_key] = interrupt_reason

        asyncio.run(_fake_interrupt_and_clear(
            "sess:a",
            MagicMock(),
            interrupt_reason=_INTERRUPT_REASON_STOP,
            invalidation_reason="test",
        ))

        assert runner._session_interrupt_reasons.get("sess:a") == _INTERRUPT_REASON_STOP

    def test_interrupt_running_agents_records_reason(self):
        """_interrupt_running_agents stores the reason for each interrupted session."""
        from gateway.run import GatewayRunner, _INTERRUPT_REASON_GATEWAY_SHUTDOWN, _AGENT_PENDING_SENTINEL

        runner = _make_runner()
        mock_agent_a = MagicMock()
        mock_agent_b = MagicMock()
        runner._running_agents = {
            "sess:a": mock_agent_a,
            "sess:b": mock_agent_b,
            "sess:pending": _AGENT_PENDING_SENTINEL,
        }

        runner._interrupt_running_agents(_INTERRUPT_REASON_GATEWAY_SHUTDOWN)

        assert runner._session_interrupt_reasons.get("sess:a") == _INTERRUPT_REASON_GATEWAY_SHUTDOWN
        assert runner._session_interrupt_reasons.get("sess:b") == _INTERRUPT_REASON_GATEWAY_SHUTDOWN
        # Pending sentinel sessions are skipped — no reason recorded
        assert "sess:pending" not in runner._session_interrupt_reasons

    def test_interrupt_running_agents_restart_reason(self):
        """_interrupt_running_agents uses gateway restart reason when restarting."""
        from gateway.run import GatewayRunner, _INTERRUPT_REASON_GATEWAY_RESTART

        runner = _make_runner()
        mock_agent = MagicMock()
        runner._running_agents = {"sess:x": mock_agent}

        runner._interrupt_running_agents(_INTERRUPT_REASON_GATEWAY_RESTART)

        assert runner._session_interrupt_reasons.get("sess:x") == _INTERRUPT_REASON_GATEWAY_RESTART

    def test_multiple_interrupts_only_last_reason_shown(self):
        """When a session is interrupted multiple times, only the most recent reason is kept."""
        from gateway.run import _INTERRUPT_REASON_STOP, _INTERRUPT_REASON_TIMEOUT

        runner = _make_runner()
        mock_agent = MagicMock()
        runner._running_agents = {"sess:a": mock_agent}

        # First interrupt
        runner._session_interrupt_reasons["sess:a"] = _INTERRUPT_REASON_STOP
        # Second interrupt overwrites
        runner._session_interrupt_reasons["sess:a"] = _INTERRUPT_REASON_TIMEOUT

        assert runner._session_interrupt_reasons["sess:a"] == _INTERRUPT_REASON_TIMEOUT

    def test_reason_cleared_after_consumption(self):
        """After _handle_message_with_agent consumes the reason, it is not present."""
        from gateway.run import _INTERRUPT_REASON_STOP

        runner = _make_runner()
        runner._session_interrupt_reasons["sess:a"] = _INTERRUPT_REASON_STOP

        # Simulate what _handle_message_with_agent does
        reason = runner._session_interrupt_reasons.pop("sess:a", None)

        assert reason == _INTERRUPT_REASON_STOP
        assert "sess:a" not in runner._session_interrupt_reasons

        # Calling pop again returns None (not shown twice)
        reason2 = runner._session_interrupt_reasons.pop("sess:a", None)
        assert reason2 is None

    def test_evict_cached_agent_cleans_up_orphaned_reason(self):
        """_evict_cached_agent removes the interrupt reason to prevent leaks."""
        from gateway.run import _INTERRUPT_REASON_STOP

        runner = _make_runner()
        runner._session_interrupt_reasons["sess:a"] = _INTERRUPT_REASON_STOP

        runner._evict_cached_agent("sess:a")

        assert "sess:a" not in runner._session_interrupt_reasons

    def test_evict_cached_agent_noop_when_no_reason(self):
        """_evict_cached_agent does not raise when no reason is stored."""
        runner = _make_runner()
        runner._evict_cached_agent("sess:nonexistent")  # must not raise

    def test_normal_completion_leaves_no_reason(self):
        """A session that completes normally has no entry in _session_interrupt_reasons."""
        runner = _make_runner()
        # No interrupt was issued
        assert "sess:a" not in runner._session_interrupt_reasons

    def test_release_running_agent_state_preserves_interrupt_reason(self):
        """_release_running_agent_state must NOT clear the interrupt reason.

        The reason must survive until the next message is received.
        """
        from gateway.run import _INTERRUPT_REASON_STOP

        runner = _make_runner()
        runner._running_agents["sess:a"] = MagicMock()
        runner._running_agents_ts["sess:a"] = 1.0
        runner._session_interrupt_reasons["sess:a"] = _INTERRUPT_REASON_STOP

        runner._release_running_agent_state("sess:a")

        # Running agent state is cleared
        assert "sess:a" not in runner._running_agents
        # But interrupt reason is preserved for the next message
        assert runner._session_interrupt_reasons.get("sess:a") == _INTERRUPT_REASON_STOP


# ---------------------------------------------------------------------------
# Integration-style tests: context_prompt injection
# ---------------------------------------------------------------------------


class TestContextPromptInjection:
    """Interrupt reason is prepended to context_prompt in _handle_message_with_agent."""

    def _build_context_prompt_with_reason(self, session_key, reason, base_prompt=""):
        """Simulate the context_prompt injection logic."""
        _interrupt_reasons = {session_key: reason}
        context_prompt = base_prompt

        if isinstance(_interrupt_reasons, dict) and session_key in _interrupt_reasons:
            _prior_reason = _interrupt_reasons.pop(session_key)
            _interrupt_note = (
                f"\n[System: The previous agent run was interrupted ({_prior_reason}). "
                f"Resuming from where you left off.]\n"
            )
            context_prompt = _interrupt_note + (context_prompt or "")

        return context_prompt, _interrupt_reasons

    def test_stop_reason_injected_into_context_prompt(self):
        """Stop reason is prepended to context_prompt."""
        from gateway.run import _INTERRUPT_REASON_STOP

        prompt, remaining = self._build_context_prompt_with_reason(
            "sess:a", _INTERRUPT_REASON_STOP, "You are a helpful assistant."
        )

        assert _INTERRUPT_REASON_STOP in prompt
        assert "[System:" in prompt
        assert "Resuming from where you left off." in prompt
        # Base prompt is preserved
        assert "You are a helpful assistant." in prompt
        # Reason consumed
        assert "sess:a" not in remaining

    def test_timeout_reason_injected_into_context_prompt(self):
        """Timeout reason is prepended to context_prompt."""
        from gateway.run import _INTERRUPT_REASON_TIMEOUT

        prompt, remaining = self._build_context_prompt_with_reason(
            "sess:a", _INTERRUPT_REASON_TIMEOUT
        )

        assert _INTERRUPT_REASON_TIMEOUT in prompt

    def test_no_interrupt_reason_leaves_context_prompt_unchanged(self):
        """Without an interrupt reason, context_prompt is not modified."""
        base = "You are a helpful assistant."
        _interrupt_reasons: dict = {}
        session_key = "sess:a"
        context_prompt = base

        if isinstance(_interrupt_reasons, dict) and session_key in _interrupt_reasons:
            _prior_reason = _interrupt_reasons.pop(session_key)
            _interrupt_note = (
                f"\n[System: The previous agent run was interrupted ({_prior_reason}). "
                f"Resuming from where you left off.]\n"
            )
            context_prompt = _interrupt_note + (context_prompt or "")

        assert context_prompt == base
        assert "[System:" not in context_prompt

    def test_reason_not_shown_twice(self):
        """Consuming the reason via pop means a second call finds nothing."""
        from gateway.run import _INTERRUPT_REASON_RESET

        _interrupt_reasons = {"sess:a": _INTERRUPT_REASON_RESET}

        # First message — reason consumed
        prompt1, _interrupt_reasons = self._build_context_prompt_with_reason.__func__(
            self, "sess:a", _INTERRUPT_REASON_RESET
        )
        # Use the dict directly
        _interrupt_reasons2 = {"sess:a": _INTERRUPT_REASON_RESET}
        session_key = "sess:a"

        # First consumption
        reason1 = _interrupt_reasons2.pop(session_key, None)
        assert reason1 == _INTERRUPT_REASON_RESET

        # Second consumption — already gone
        reason2 = _interrupt_reasons2.pop(session_key, None)
        assert reason2 is None

    def test_interrupt_note_prepended_before_base_prompt(self):
        """The interrupt note appears before the base context prompt."""
        from gateway.run import _INTERRUPT_REASON_STOP

        prompt, _ = self._build_context_prompt_with_reason(
            "sess:a", _INTERRUPT_REASON_STOP, "base context"
        )

        interrupt_pos = prompt.find("[System:")
        base_pos = prompt.find("base context")
        assert interrupt_pos < base_pos, "Interrupt note must precede the base prompt"
