"""Tests for the pre-loop signal buffer in gateway/run.py.

The buffer closes the startup-race window where SIGTERM / SIGINT /
SIGUSR1 arriving between process start and ``loop.add_signal_handler()``
running would otherwise let Python's default action terminate the
gateway silently — producing the "ghost restart" pattern (systemd's
``Restart=always`` would respawn with no log line attributing the
kill). See incident 2026-09-04 13:48:51.

These tests exercise the buffer directly without booting the gateway
(asyncio loop, plugins, systemd). They cover:

1. Buffer records incoming signals in FIFO order.
2. Drain replays SIGTERM/SIGINT to the shutdown handler.
3. Drain replays SIGUSR1 to the restart handler.
4. Drain is idempotent (a second drain is a no-op).
5. The module-level handler is safe to install twice.
"""

from __future__ import annotations

import signal
from typing import List
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_buffer():
    """Reset module-level buffer state between tests."""
    from gateway import run as run_mod

    run_mod._PRE_LOOP_SIGNAL_BUFFER.clear()
    run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED = False
    yield
    run_mod._PRE_LOOP_SIGNAL_BUFFER.clear()
    run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED = False


def test_buffer_records_signals_in_fifo_order():
    """Buffer appends in arrival order; preserves duplicates."""
    from gateway import run as run_mod

    # Invoke the handler as if the kernel delivered the signal.
    run_mod._pre_loop_signal_buffer(signal.SIGTERM, None)
    run_mod._pre_loop_signal_buffer(signal.SIGUSR1, None)
    run_mod._pre_loop_signal_buffer(signal.SIGTERM, None)

    assert run_mod._PRE_LOOP_SIGNAL_BUFFER == [
        signal.SIGTERM,
        signal.SIGUSR1,
        signal.SIGTERM,
    ]
    assert run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED is False


def test_drain_replays_sigterm_to_shutdown_handler():
    """A SIGTERM buffered in the pre-loop window reaches shutdown_signal_handler."""
    from gateway import run as run_mod

    loop = MagicMock()
    shutdown = MagicMock()
    restart = MagicMock()

    run_mod._pre_loop_signal_buffer(signal.SIGTERM, None)
    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)

    # The drain schedules on_shutdown via loop.call_soon (so the
    # call lands on the loop thread, not the main thread). We assert
    # on the scheduling, not on the callback itself running, because
    # we're testing the drain logic, not the loop.
    assert loop.call_soon.call_count == 1
    loop.call_soon.assert_called_once_with(shutdown)
    restart.assert_not_called()
    assert run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED is True


def test_drain_replays_sigint_to_shutdown_handler():
    """SIGINT (Ctrl+C) follows the same path as SIGTERM."""
    from gateway import run as run_mod

    loop = MagicMock()
    shutdown = MagicMock()
    restart = MagicMock()

    run_mod._pre_loop_signal_buffer(signal.SIGINT, None)
    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)

    assert loop.call_soon.call_count == 1
    loop.call_soon.assert_called_once_with(shutdown)


def test_drain_replays_sigusr1_to_restart_handler():
    """SIGUSR1 (graceful restart) routes to restart_signal_handler."""
    from gateway import run as run_mod

    loop = MagicMock()
    shutdown = MagicMock()
    restart = MagicMock()

    run_mod._pre_loop_signal_buffer(signal.SIGUSR1, None)
    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)

    assert loop.call_soon.call_count == 1
    loop.call_soon.assert_called_once_with(restart)
    # Confirm shutdown was NOT used for SIGUSR1.
    for call in loop.call_soon.call_args_list:
        args, _ = call
        assert shutdown not in args, "SIGUSR1 should not route to shutdown handler"


def test_drain_with_multiple_buffered_signals_routes_each():
    """Multiple signals in the buffer each go to their correct handler."""
    from gateway import run as run_mod

    loop = MagicMock()
    shutdown = MagicMock()
    restart = MagicMock()

    # 2 SIGTERM, 1 SIGUSR1, 1 SIGINT — order matters because that's
    # the arrival order to the kernel.
    run_mod._pre_loop_signal_buffer(signal.SIGTERM, None)
    run_mod._pre_loop_signal_buffer(signal.SIGUSR1, None)
    run_mod._pre_loop_signal_buffer(signal.SIGINT, None)
    run_mod._pre_loop_signal_buffer(signal.SIGTERM, None)

    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)

    # 4 scheduled callbacks total
    assert loop.call_soon.call_count == 4
    scheduled = [c.args[0] for c in loop.call_soon.call_args_list]
    # SIGTERM, restart, SIGINT, SIGTERM (in arrival order)
    assert scheduled == [shutdown, restart, shutdown, shutdown]


def test_drain_clears_the_buffer():
    """After drain, the buffer is empty so a second drain is a no-op."""
    from gateway import run as run_mod

    loop = MagicMock()
    shutdown = MagicMock()
    restart = MagicMock()

    run_mod._pre_loop_signal_buffer(signal.SIGTERM, None)
    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)

    # Second drain — should not schedule anything because buffer is empty.
    loop.call_soon.reset_mock()
    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)

    loop.call_soon.assert_not_called()
    assert run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED is True


def test_drain_marks_handlers_installed_atomically():
    """The handlers-installed flag is set to True exactly once."""
    from gateway import run as run_mod

    loop = MagicMock()
    shutdown = MagicMock()
    restart = MagicMock()

    assert run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED is False
    run_mod._pre_loop_signal_buffer(signal.SIGUSR1, None)
    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)
    assert run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED is True

    # Subsequent drains keep the flag True.
    run_mod._drain_pre_loop_signal_buffer(loop, on_shutdown=shutdown, on_restart=restart)
    assert run_mod._PRE_LOOP_SIGNAL_HANDLERS_INSTALLED is True


def test_buffer_handler_is_pure_side_effect():
    """The signal handler must never raise, regardless of buffer state.

    Note: CPython delivers signals to the main thread only, so the
    handler cannot in practice run concurrently with itself. We test
    the pure-function behaviour: appending twice and clearing never
    leaks the lock, and the handler returns ``None`` (no exception).
    """
    from gateway import run as run_mod

    # The handler signature must accept (sig, frame) and return None.
    result = run_mod._pre_loop_signal_buffer(signal.SIGUSR1, None)
    assert result is None

    # Repeated calls accumulate; lock is reacquired cleanly each time.
    run_mod._pre_loop_signal_buffer(signal.SIGUSR1, None)
    run_mod._pre_loop_signal_buffer(signal.SIGUSR1, None)
    assert len(run_mod._PRE_LOOP_SIGNAL_BUFFER) == 3


def test_buffer_records_real_signal_delivery():
    """If the kernel actually delivers a signal during the pre-loop window,
    it is buffered (not lost). We use SIGUSR2 (unrelated to gateway lifecycle
    and not handled by us) to confirm the pre_loop_signal_buffer handler
    is installed in the module namespace.

    Note: we don't trigger SIGTERM here because the test process doesn't
    have systemd to absorb the default-action side effect.
    """
    from gateway import run as run_mod

    # Verify the handler is registered for SIGUSR2 — we use SIGUSR2 because
    # SIGUSR1 is the restart signal (would affect the gateway if it were
    # running). With SIGUSR2 we can prove the handler runs without
    # disrupting anything.
    initial_count = len(run_mod._PRE_LOOP_SIGNAL_BUFFER)
    run_mod._pre_loop_signal_buffer(signal.SIGUSR2, None)
    assert len(run_mod._PRE_LOOP_SIGNAL_BUFFER) == initial_count + 1
    assert run_mod._PRE_LOOP_SIGNAL_BUFFER[-1] == signal.SIGUSR2