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
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_short_circuits_when_streak_at_threshold(self, monkeypatch):
        """A session already past the consecutive-stale threshold must abort
        immediately without opening a stream or waiting out the stale timeout."""
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3  # simulate prior wedged turns

        # The stream must never be opened on the short-circuit path.
        with pytest.raises(TimeoutError, match="unresponsive"):
            agent._interruptible_streaming_api_call({})

        agent._anthropic_client.messages.stream.assert_not_called()
        # The streak is NOT reset on the short-circuit so subsequent turns
        # keep failing fast instead of re-attempting forever.
        assert agent._consecutive_stale_streams == 3

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
        agent._anthropic_client.close.side_effect = unblock.set

        with pytest.raises(Exception):
            agent._interruptible_streaming_api_call({})

        # At least one stale kill happened; the streak must have advanced.
        assert agent._consecutive_stale_streams >= 1
