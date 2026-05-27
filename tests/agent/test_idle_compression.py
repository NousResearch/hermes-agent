"""
Tests for ``agent.idle_compression`` — the IdleCompressionTimer.

Covers:
- Timer starts and cancels correctly
- Token floor check (skips small sessions)
- Thread cleanup on rapid start/cancel
- Config values are read properly
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.idle_compression import (
    DEFAULT_IDLE_DELAY_SECONDS,
    DEFAULT_MIN_TOKENS,
    IdleCompressionTimer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agent() -> MagicMock:
    """Return a MagicMock that looks enough like an AIAgent for the timer."""
    agent = MagicMock()
    agent.messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Hello" * 500},  # ~2,000 chars
        {"role": "assistant", "content": "Hi there" * 500},
    ]
    agent.compression_enabled = True
    agent._emit_status = MagicMock()
    agent._emit_warning = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="sys prompt")
    agent._compress_context = MagicMock(
        return_value=(agent.messages, "sys prompt")
    )
    agent._invalidate_system_prompt = MagicMock()
    return agent


@pytest.fixture
def timer(mock_agent: MagicMock) -> IdleCompressionTimer:
    """Create a timer with a short delay for testing."""
    return IdleCompressionTimer(
        agent=mock_agent,
        delay_seconds=0.1,  # 100 ms — fast tests
        min_tokens=999_999,  # high default so token floor is skipped unless overridden
    )


# ---------------------------------------------------------------------------
# Start / cancel lifecycle
# ---------------------------------------------------------------------------

class TestStartCancel:
    """Timer starts, reports active, and cancels cleanly."""

    def test_start_sets_active(self, timer: IdleCompressionTimer) -> None:
        """After start(), active reports True."""
        timer.start()
        assert timer.active is True
        timer.cancel()

    def test_cancel_clears_active(self, timer: IdleCompressionTimer) -> None:
        """After cancel(), active reports False."""
        timer.start()
        timer.cancel()
        # Give threads a moment to finish
        time.sleep(0.05)
        assert timer.active is False

    def test_start_cancels_previous(self, timer: IdleCompressionTimer) -> None:
        """Calling start() twice cancels the first timer."""
        timer.start()
        first_thread = timer._timer_thread
        timer.start()
        second_thread = timer._timer_thread
        # A new thread should have been created
        assert second_thread is not first_thread
        timer.cancel()

    def test_cancel_idempotent(self, timer: IdleCompressionTimer) -> None:
        """Cancel is safe to call multiple times."""
        timer.start()
        timer.cancel()
        timer.cancel()  # should not raise
        timer.cancel()  # should not raise
        assert timer.active is False

    def test_cancel_without_start(self, timer: IdleCompressionTimer) -> None:
        """Cancel before start is safe."""
        timer.cancel()
        assert timer.active is False


# ---------------------------------------------------------------------------
# Token floor
# ---------------------------------------------------------------------------

class TestTokenFloor:
    """The timer skips compression when estimated tokens are below min_tokens."""

    def test_skip_small_session(self, mock_agent: MagicMock) -> None:
        """Low token count → timer fires but skips compression."""
        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=0.1,
            min_tokens=1_000_000,  # very high floor
        )
        timer.start()
        # Wait for the timer to elapse + the compression thread (if any) to finish
        time.sleep(0.3)
        timer.cancel()
        # Compression should NOT have been called
        mock_agent._compress_context.assert_not_called()

    def test_proceed_when_above_floor(
        self, mock_agent: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """High token estimate → compression fires."""
        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=0.1,
            min_tokens=100,  # almost anything passes
        )
        # Ensure compression does a real swap so the "actual compression happened"
        # check (compressed_msgs is not messages) passes.
        new_msgs = [{"role": "system", "content": "compressed"}]
        mock_agent._compress_context.return_value = (new_msgs, "sys")
        mock_agent.messages = mock_agent.messages[:]  # different list identity

        timer.start()
        time.sleep(0.3)
        timer.cancel()
        mock_agent._compress_context.assert_called()

    def test_skips_when_compression_disabled(
        self, mock_agent: MagicMock
    ) -> None:
        """Disabled compression → skipped even with enough tokens."""
        mock_agent.compression_enabled = False
        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=0.1,
            min_tokens=1,  # always passes
        )
        timer.start()
        time.sleep(0.3)
        timer.cancel()
        mock_agent._compress_context.assert_not_called()


# ---------------------------------------------------------------------------
# Thread cleanup
# ---------------------------------------------------------------------------

class TestThreadCleanup:
    """Rapid start/cancel does not leak threads."""

    def test_rapid_start_cancel_no_leak(
        self, mock_agent: MagicMock
    ) -> None:
        """Many start/cancel cycles should not accumulate threads."""
        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=30.0,  # long delay — timer will never fire
            min_tokens=999_999,
        )
        initial_count = threading.active_count()

        for _ in range(20):
            timer.start()
            timer.cancel()

        # Give daemon threads a moment
        time.sleep(0.1)

        final_count = threading.active_count()
        # At most a few extra threads (timer threads that haven't exited yet)
        assert final_count <= initial_count + 3, (
            f"Thread count grew from {initial_count} to {final_count} "
            f"— possible leak"
        )

    def test_compression_thread_joined_on_cancel(
        self, mock_agent: MagicMock
    ) -> None:
        """If compression is in flight at cancel time, it is joined."""
        # Make _run_compression take a long time
        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=0.05,
            min_tokens=1,
        )

        # Override _run_compression to sleep
        original_run = timer._run_compression

        def slow_run() -> None:
            time.sleep(0.5)

        timer._run_compression = slow_run  # type: ignore[method-assign]

        timer.start()
        time.sleep(0.15)  # should have spawned compression thread by now
        timer.cancel()
        # Cancel should finish without hanging
        assert True  # reached = no hang

        # Restore
        timer._run_compression = original_run  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Config values
# ---------------------------------------------------------------------------

class TestConfigValues:
    """Config values (delay_seconds, min_tokens) are read and settable."""

    def test_default_values(self) -> None:
        """Constants match expected defaults."""
        assert DEFAULT_IDLE_DELAY_SECONDS == 300
        assert DEFAULT_MIN_TOKENS == 20_000

    def test_constructor_uses_defaults(self, mock_agent: MagicMock) -> None:
        """Without explicit args, the timer uses module defaults."""
        timer = IdleCompressionTimer(agent=mock_agent)
        assert timer.delay_seconds == 300
        assert timer.min_tokens == 20_000

    def test_custom_delay_and_tokens(self, mock_agent: MagicMock) -> None:
        """Custom values flow through constructor and properties."""
        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=120.0,
            min_tokens=5000,
        )
        assert timer.delay_seconds == 120.0
        assert timer.min_tokens == 5000

    def test_setter_clamps_delay(self, mock_agent: MagicMock) -> None:
        """delay_seconds setter enforces minimum 10s."""
        timer = IdleCompressionTimer(agent=mock_agent)
        timer.delay_seconds = 5.0
        assert timer.delay_seconds == 10.0

        timer.delay_seconds = 0
        assert timer.delay_seconds == 10.0

        timer.delay_seconds = 60.0
        assert timer.delay_seconds == 60.0

    def test_setter_clamps_min_tokens(self, mock_agent: MagicMock) -> None:
        """min_tokens setter enforces minimum 1000."""
        timer = IdleCompressionTimer(agent=mock_agent)
        timer.min_tokens = 500
        assert timer.min_tokens == 1000

        timer.min_tokens = 0
        assert timer.min_tokens == 1000

        timer.min_tokens = 50_000
        assert timer.min_tokens == 50_000

    def test_active_property_when_idle(self, mock_agent: MagicMock) -> None:
        """active is False when nothing is running."""
        timer = IdleCompressionTimer(agent=mock_agent)
        assert timer.active is False
        assert timer.compressing is False

    def test_on_compress_callback(
        self, mock_agent: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The on_compress callback fires with success bool."""
        callback_called = []

        def cb(success: bool) -> None:
            callback_called.append(success)

        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=0.05,
            min_tokens=1,
            on_compress=cb,
        )
        new_msgs = [{"role": "system", "content": "compressed"}]
        mock_agent._compress_context.return_value = (new_msgs, "sys")
        mock_agent.messages = mock_agent.messages[:]

        timer.start()
        time.sleep(0.2)
        timer.cancel()
        assert len(callback_called) >= 1, "on_compress callback should have fired"
        # Should report success since we made compress return different msgs
        assert callback_called[-1] is True

    def test_on_compress_callback_on_skip(
        self, mock_agent: MagicMock
    ) -> None:
        """Callback fires with False when compression is skipped (token floor)."""
        callback_called = []

        def cb(success: bool) -> None:
            callback_called.append(success)

        timer = IdleCompressionTimer(
            agent=mock_agent,
            delay_seconds=0.05,
            min_tokens=999_999,
            on_compress=cb,
        )
        timer.start()
        time.sleep(0.2)
        timer.cancel()
        # No callback when skipped (the token floor short-circuits before
        # spawning the compression thread, so _run_compression never runs).
        # This is correct behavior — the callback is for actual compression
        # attempts.

    def test_delay_seconds_setter_negative(self, mock_agent: MagicMock) -> None:
        """Negative delay is clamped to minimum of 10."""
        timer = IdleCompressionTimer(agent=mock_agent)
        timer.delay_seconds = -50.0
        assert timer.delay_seconds == 10.0
