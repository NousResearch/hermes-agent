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


class TestStaleGiveupHalfOpenProbe:
    """Half-open recovery for the tripped give-up breaker (issue #89587).

    Once the streak crosses HERMES_STREAM_STALE_GIVEUP, one real probe
    attempt per HERMES_STREAM_STALE_PROBE_INTERVAL_S window is allowed
    through instead of the unconditional insta-fail, so an unattended
    single-provider session self-heals after the provider recovers.
    """

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_probe_allowed_when_window_elapsed(self, monkeypatch):
        """With the probe window elapsed, the tripped breaker lets one real
        attempt through; a healthy provider completes it normally."""
        import time

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3
        agent._stale_probe_after = time.monotonic() - 1.0  # window elapsed
        agent._anthropic_client.messages.stream.return_value = _good_stream_cm()

        resp = agent._interruptible_streaming_api_call({})
        assert resp is not None
        agent._anthropic_client.messages.stream.assert_called_once()

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_successful_probe_resets_streak_via_existing_reset(self, monkeypatch):
        """A successful probe rides the existing completed-call reset: the
        streak clears and the probe window is disarmed for the next trip."""
        import time

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3
        agent._stale_probe_after = time.monotonic() - 1.0
        agent._anthropic_client.messages.stream.return_value = _good_stream_cm()

        resp = agent._interruptible_streaming_api_call({})
        assert resp is not None
        assert agent._consecutive_stale_streams == 0
        assert agent._stale_probe_after == 0.0

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_short_circuit_within_window_keeps_insta_fail(self, monkeypatch):
        """Inside the probe window the breaker still insta-fails with no
        network attempt — the interactive fail-fast UX is unchanged."""
        import time

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3
        agent._stale_probe_after = time.monotonic() + 3600.0  # window far off

        with pytest.raises(RuntimeError, match="consecutive stale attempts"):
            agent._interruptible_streaming_api_call({})

        agent._anthropic_client.messages.stream.assert_not_called()
        assert agent._consecutive_stale_streams == 3

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_first_short_circuit_arms_window(self, monkeypatch):
        """The first over-threshold short-circuit arms the probe window (the
        trip call, or an interrupt-counted overshoot that never ran the
        guard) — it does NOT probe immediately, which would re-wait the very
        stale timeout the breaker exists to avoid."""
        import time

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 4  # overshoot; window never armed
        agent._stale_probe_after = 0.0

        with pytest.raises(RuntimeError, match="consecutive stale attempts"):
            agent._interruptible_streaming_api_call({})

        agent._anthropic_client.messages.stream.assert_not_called()
        assert agent._stale_probe_after > time.monotonic()

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_failed_probe_rearms_window(self, monkeypatch):
        """A probe that itself goes stale re-bumps the streak and re-arms the
        window, so the next call insta-fails instead of probing again."""
        import time

        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3
        agent._stale_probe_after = time.monotonic() - 1.0  # probe due
        unblock = threading.Event()

        def _blocking_gen():
            unblock.wait(timeout=5.0)
            raise httpx.ConnectError("connection dropped after close()")
            yield  # pragma: no cover — generator marker

        def _stream_side_effect(*args, **kwargs):
            cm = MagicMock()
            stream = MagicMock()
            stream.__iter__ = MagicMock(return_value=_blocking_gen())
            cm.__enter__ = MagicMock(return_value=stream)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        agent._abort_request_anthropic_client = lambda *a, **k: unblock.set()

        with pytest.raises(Exception):
            agent._interruptible_streaming_api_call({})

        assert agent._consecutive_stale_streams >= 4
        assert agent._stale_probe_after > time.monotonic()
        opened = agent._anthropic_client.messages.stream.call_count

        # The follow-up call sits inside the re-armed window: insta-fail,
        # no new stream opened.
        with pytest.raises(RuntimeError, match="consecutive stale attempts"):
            agent._interruptible_streaming_api_call({})
        assert agent._anthropic_client.messages.stream.call_count == opened

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_probe_interval_zero_disables_probing(self, monkeypatch):
        """HERMES_STREAM_STALE_PROBE_INTERVAL_S=0 restores the pure latch:
        no probes, even with an elapsed window."""
        import time

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")
        monkeypatch.setenv("HERMES_STREAM_STALE_PROBE_INTERVAL_S", "0")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3
        agent._stale_probe_after = time.monotonic() - 100.0

        with pytest.raises(RuntimeError, match="consecutive stale attempts"):
            agent._interruptible_streaming_api_call({})

        agent._anthropic_client.messages.stream.assert_not_called()

    def test_probe_message_mentions_next_probe_and_keeps_classifier_substrings(
        self, monkeypatch
    ):
        """The in-window error must keep the exact substrings
        agent/error_classifier.py matches ("consecutive stale attempts",
        "aborting this call") and tell the user when the next probe fires."""
        import time

        from agent.chat_completion_helpers import _check_stale_giveup

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 5
        agent._stale_probe_after = time.monotonic() + 120.0

        with pytest.raises(RuntimeError) as excinfo:
            _check_stale_giveup(agent)

        msg = str(excinfo.value)
        assert "consecutive stale attempts" in msg
        assert "aborting this call" in msg
        assert "Next automatic probe" in msg
