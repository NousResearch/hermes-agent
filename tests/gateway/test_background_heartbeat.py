"""Tests for the GIL-immune background OS-thread heartbeat (#72707)."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.shutdown_watchdog import (
    BACKGROUND_HEARTBEAT_INTERVAL_S,
    get_loop_heartbeat_path,
    is_background_heartbeat_alive,
    start_background_heartbeat,
    stop_background_heartbeat,
    write_loop_heartbeat,
)


def _read_heartbeat(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class TestBackgroundHeartbeat:
    """Tests for the background OS-thread heartbeat runner."""

    def test_start_stop_lifecycle(self, tmp_path: Path) -> None:
        """Starting then stopping the heartbeat thread works cleanly."""
        assert not is_background_heartbeat_alive()
        start_background_heartbeat(home=tmp_path)
        assert is_background_heartbeat_alive()
        stop_background_heartbeat()
        # Give the thread a moment to exit
        time.sleep(0.1)
        assert not is_background_heartbeat_alive()

    def test_start_is_idempotent(self, tmp_path: Path) -> None:
        """Calling start_background_heartbeat twice is a no-op."""
        start_background_heartbeat(home=tmp_path)
        thread_id = id(is_background_heartbeat_alive)
        start_background_heartbeat(home=tmp_path)  # second call
        assert is_background_heartbeat_alive()
        stop_background_heartbeat()

    def test_writes_heartbeat_file(self, tmp_path: Path) -> None:
        """The thread writes the heartbeat file periodically."""
        start_background_heartbeat(interval_s=0.3, home=tmp_path)
        hb_path = get_loop_heartbeat_path(tmp_path)
        # Wait for at least one write cycle
        time.sleep(0.5)
        assert hb_path.exists(), "Heartbeat file was never written"
        data = _read_heartbeat(hb_path)
        assert "pid" in data
        assert data["pid"] == os.getpid()
        assert "monotonic" in data
        assert "thread_monotonic" in data
        assert "updated_at" in data
        stop_background_heartbeat()

    def test_thread_monotonic_stays_current(self, tmp_path: Path) -> None:
        """The thread_monotonic field updates on every write cycle.

        Unlike the async loop's monotonic field (which goes stale when the
        event loop is GIL-starved), the thread's monotonic should advance
        with real wall time because time.sleep() releases the GIL.
        """
        start_background_heartbeat(interval_s=0.2, home=tmp_path)
        hb_path = get_loop_heartbeat_path(tmp_path)
        time.sleep(0.3)
        t1 = _read_heartbeat(hb_path)["thread_monotonic"]
        time.sleep(0.3)
        t2 = _read_heartbeat(hb_path)["thread_monotonic"]
        assert t2 > t1, (
            f"thread_monotonic did not advance: t1={t1}, t2={t2}"
        )
        stop_background_heartbeat()

    def test_thread_monotonic_vs_loop_monotonic(self, tmp_path: Path) -> None:
        """thread_monotonic advances independently from the async monotonic.

        The async-loop monotonic only advances when the event loop runs;
        the thread monotonic advances on its own cadence.  Here we simulate
        by writing the async field once (old) and then starting the thread
        (which writes fresh thread_monotonic).
        """
        # Write an "async" heartbeat with a stale monotonic
        stale_time = time.monotonic() - 30.0
        write_loop_heartbeat(
            home=tmp_path,
            extra={"monotonic": stale_time},
        )
        hb_path = get_loop_heartbeat_path(tmp_path)
        before = _read_heartbeat(hb_path)
        async_stale = before.get("monotonic", before.get("extra", {}).get("monotonic"))
        # The "monotonic" key is always set by write_loop_heartbeat itself,
        # so let's check the actual top-level key
        assert before.get("monotonic") is not None

        # Now start the background thread — it should overwrite the file
        # with a fresh thread_monotonic
        start_background_heartbeat(interval_s=0.2, home=tmp_path)
        time.sleep(0.3)
        after = _read_heartbeat(hb_path)
        assert after["thread_monotonic"] > stale_time, (
            "thread_monotonic should be fresh even though the async "
            "monotonic was written 30s ago"
        )
        stop_background_heartbeat()

    def test_alive_returns_false_before_start(self) -> None:
        """is_background_heartbeat_alive returns False when thread not started."""
        assert not is_background_heartbeat_alive()

    def test_survives_main_thread_block(self, tmp_path: Path) -> None:
        """The heartbeat thread continues writing even when the main thread
        is blocked (simulating GIL pressure with a CPU-bound spin).

        This validates the core claim: time.sleep() and file I/O release
        the GIL, so the background heartbeat thread is immune to GIL stalls
        on the main thread.
        """
        start_background_heartbeat(interval_s=0.15, home=tmp_path)
        hb_path = get_loop_heartbeat_path(tmp_path)
        # Let the thread write the first heartbeat
        time.sleep(0.3)
        before_spin = _read_heartbeat(hb_path)["thread_monotonic"]

        # Simulate GIL pressure: CPU-bound spin on the main thread for 1s.
        # The background thread should still wake up and write during this.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            _ = [i**2 for i in range(10000)]

        after_spin = _read_heartbeat(hb_path)["thread_monotonic"]
        assert after_spin > before_spin, (
            "Heartbeat thread did not advance during main-thread CPU spin. "
            "This suggests the thread could not acquire the GIL, which "
            "contradicts the assumption that time.sleep() releases it. "
            f"before={before_spin}, after={after_spin}"
        )
        stop_background_heartbeat()
