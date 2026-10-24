"""REPRO: loop-liveness watchdog vs a REAL blocked asyncio loop.

The existing tests in test_loop_liveness_watchdog.py drive the watchdog with a MagicMock
loop, so they never exercise the case that actually bit us on 2026-10-23: the gateway's
event loop was blocked by a long SYNCHRONOUS section (GIL held), and the watchdog thread
never fired.

This test runs a REAL asyncio loop, blocks it with a synchronous sleep-loop that holds the
GIL for longer than the watchdog budget, and asserts the watchdog notices.
"""

from __future__ import annotations

import asyncio
import threading
import time

from gateway.shutdown_watchdog import start_loop_liveness_watchdog


def _make_blocking_loop():
    loop = asyncio.new_event_loop()
    blocker_started = threading.Event()
    release = threading.Event()

    def _block():
        # A synchronous busy-wait on the loop thread: mimics a blocking C-extension call
        # (Pillow decode, sqlite fsync, a sync SDK call) running directly on the loop.
        blocker_started.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not release.is_set():
            time.sleep(0.05)

    def _runner():
        asyncio.set_event_loop(loop)
        loop.call_soon(_block)
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True, name="blocked-loop")
    t.start()
    assert blocker_started.wait(timeout=2.0)
    return loop, release, t


def test_watchdog_fires_when_real_loop_is_blocked_synchronously():
    loop, release, _t = _make_blocking_loop()
    exited = threading.Event()
    exit_codes: list[int] = []

    def _fake_exit(code):
        exit_codes.append(code)
        exited.set()

    import gateway.shutdown_watchdog as sw

    real_exit = sw.os._exit
    sw.os._exit = _fake_exit
    real_mark = sw._mark_exited_quietly
    sw._mark_exited_quietly = lambda *a, **k: None
    handle = None
    try:
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.2, probe_timeout=0.2, max_strikes=2
        )
        assert handle is not None
        fired = exited.wait(timeout=6.0)
        release.set()
        time.sleep(0.2)
        loop.call_soon_threadsafe(loop.stop)
    finally:
        # Stop the watchdog BEFORE restoring os._exit: a live watchdog thread would fire
        # the REAL os._exit(75) later and kill the pytest process.
        if handle is not None:
            handle.stop()
            handle.join(timeout=2.0)
        sw.os._exit = real_exit
        sw._mark_exited_quietly = real_mark

    assert fired, (
        "watchdog did NOT fire against a genuinely blocked loop - "
        "the 2026-10-23 gateway wedge would not have been caught"
    )
    assert exit_codes and exit_codes[0] != 0
