"""Cross-turn stream-stale circuit breaker (issue #58962).

A session wedged against an unresponsive provider can hit the stale-stream
detector on every turn and loop forever, burning the full 180s×retries each
turn with no response (observed: 494 consecutive failures over 3+ days).

These tests cover the guard added to ``interruptible_streaming_api_call``:

- a session that has already tripped the consecutive-stale threshold short
  circuits immediately (no network attempt, no 180s wait) with a clear error;
- a successful stream resets the consecutive-stale streak;
- a stale-stream kill increments the consecutive-stale streak.

The harness mirrors tests/run_agent/test_28161_anthropic_stream_pool_cleanup.py.
"""

import threading

import httpx
import pytest
from unittest.mock import MagicMock

from types import SimpleNamespace


def _make_anthropic_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-opus-4-7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    # #67142: anthropic streams now run on a request-local client; route it to
    # the test mock so .messages.stream is exercised.
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client
    return agent


def _good_stream_cm():
    """Context manager whose stream yields no events and returns a valid message."""
    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter([]))
    msg = MagicMock()
    msg.content = []
    msg.stop_reason = "end_turn"
    msg.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    stream.get_final_message = MagicMock(return_value=msg)
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestStreamStaleCircuitBreaker:
    def test_interrupted_pre_response_wait_advances_streak(self, monkeypatch):
        """Qualified pre-response interrupts advance the breaker, while early
        and mid-stream user cancellations remain neutral."""
        from agent.chat_completion_helpers import (
            _check_stale_giveup,
            _record_interrupted_provider_wait,
        )

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "2")
        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 0

        assert _record_interrupted_provider_wait(agent, 29.9, response_started=False) is False
        assert _record_interrupted_provider_wait(agent, 45.0, response_started=True) is False
        assert agent._consecutive_stale_streams == 0

        assert _record_interrupted_provider_wait(agent, 45.0, response_started=False) is True
        assert agent._consecutive_stale_streams == 1

        assert _record_interrupted_provider_wait(agent, 60.0, response_started=False) is True
        with pytest.raises(RuntimeError, match="2 consecutive stale attempts"):
            _check_stale_giveup(agent)

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_short_circuits_when_streak_at_threshold(self, monkeypatch):
        """A session already past the consecutive-stale threshold must abort
        immediately without opening a stream or waiting out the stale timeout."""
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3  # simulate prior wedged turns

        # The stream must never be opened on the short-circuit path.
        with pytest.raises(RuntimeError, match="unresponsive"):
            agent._interruptible_streaming_api_call({})

        agent._anthropic_client.messages.stream.assert_not_called()
        # The streak is NOT reset on the short-circuit so subsequent turns
        # keep failing fast instead of re-attempting forever.
        assert agent._consecutive_stale_streams == 3

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_half_open_probe_recovers_after_cooldown(self, monkeypatch):
        """A latched single-provider session eventually gets one real attempt,
        whose success closes the breaker without a model swap or new session."""
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")
        clock = [100.0]
        monkeypatch.setattr(
            "agent.chat_completion_helpers.time.monotonic", lambda: clock[0]
        )

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3
        agent._anthropic_client.messages.stream.return_value = _good_stream_cm()

        # Opening the circuit still blocks immediately and starts the cooldown.
        with pytest.raises(RuntimeError, match="unresponsive"):
            agent._interruptible_streaming_api_call({})
        agent._anthropic_client.messages.stream.assert_not_called()

        # Calls during the cooldown remain protected from another stale wait.
        clock[0] = 159.9
        with pytest.raises(RuntimeError, match="unresponsive"):
            agent._interruptible_streaming_api_call({})
        agent._anthropic_client.messages.stream.assert_not_called()

        # At the boundary, one half-open probe reaches the recovered provider.
        clock[0] = 160.0
        response = agent._interruptible_streaming_api_call({})

        assert response is not None
        agent._anthropic_client.messages.stream.assert_called_once()
        assert agent._consecutive_stale_streams == 0

    def test_guard_reads_streak_under_state_lock(self, monkeypatch):
        """A concurrent success reset cannot land after the guard snapshots
        the old streak but before it decides whether the breaker is open."""
        import agent.chat_completion_helpers as helpers

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        class AuditedLock:
            held = False

            def __enter__(self):
                self.held = True

            def __exit__(self, exc_type, exc, tb):
                self.held = False

        lock = AuditedLock()

        class AgentState:
            _stale_circuit_opened_at = 100.0
            unsafe_reads = 0

            @property
            def _consecutive_stale_streams(self):
                if not lock.held:
                    self.unsafe_reads += 1
                return 3

        agent = AgentState()
        monkeypatch.setattr(helpers, "_STALE_CIRCUIT_STATE_LOCK", lock)
        monkeypatch.setattr(helpers.time, "monotonic", lambda: 100.0)

        with pytest.raises(RuntimeError, match="unresponsive"):
            helpers._check_stale_giveup(agent)

        assert agent.unsafe_reads == 0

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_success_resets_streak(self, monkeypatch):
        """A stream that completes successfully clears the consecutive-stale
        streak so a recovered provider resumes normally."""
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 2  # below the giveup=3 threshold
        agent._anthropic_client.messages.stream.return_value = _good_stream_cm()

        resp = agent._interruptible_streaming_api_call({})
        assert resp is not None
        assert agent._consecutive_stale_streams == 0

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_stale_kill_increments_streak(self, monkeypatch):
        """Each stale-stream kill increments the consecutive-stale streak so a
        wedged session eventually trips the breaker."""
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 0
        unblock = threading.Event()

        def _blocking_gen():
            unblock.wait(timeout=5.0)
            raise httpx.ConnectError("connection dropped after close()")
            yield  # make this a generator so next() triggers the wait

        def _stream_side_effect(*args, **kwargs):
            cm = MagicMock()
            stream = MagicMock()
            stream.__iter__ = MagicMock(return_value=_blocking_gen())
            cm.__enter__ = MagicMock(return_value=stream)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        # Every attempt blocks, trips the stale detector, and fails.
        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        # #67142: the stale detector now aborts the request-local client's
        # sockets from the poll thread (not close() on the shared client), so
        # unblock on the abort to simulate the socket shutdown waking the read.
        agent._abort_request_anthropic_client = lambda *a, **k: unblock.set()

        with pytest.raises(Exception):
            agent._interruptible_streaming_api_call({})

        # At least one stale kill happened; the streak must have advanced.
        assert agent._consecutive_stale_streams >= 1
