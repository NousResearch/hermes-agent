"""Regression tests: the WhatsApp stale-bridge cleanup must never kill a stranger.

The bridge records its PID in ``bridge.pid``. On the next start the gateway
SIGTERMs that PID to reap an orphaned bridge. The original code checked only
that the PID was *alive* — but once the bridge exits and is reaped the kernel
can recycle its number onto an unrelated process. Because the WhatsApp bridge
crash-loops, this cleanup ran constantly, and a recycled PID that had landed on
the user's browser main process got SIGTERMed, closing the browser at irregular
intervals (no crash, no coredump — a clean kill of a stranger).

These tests prove the identity guard: a PID is only signalled when it is still
our bridge (kernel start time matches, or — for legacy pidfiles — its command
line names node + this session). A recycled PID is left alone.
"""

import subprocess
import sys
import time

import pytest

from gateway.platforms.whatsapp import (
    _bridge_pid_is_ours,
    _kill_stale_bridge_by_pidfile,
    _write_bridge_pidfile,
)
from gateway.status import get_process_start_time


def _spawn_sleeper(*extra_argv) -> subprocess.Popen:
    """Spawn a real, short-lived process; optional extra argv shapes its cmdline."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", *extra_argv]
    )


def _wait_dead(proc: subprocess.Popen, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return False


class TestWriteAndRoundTrip:
    def test_pidfile_records_pid_and_start_time(self, tmp_path):
        proc = _spawn_sleeper()
        try:
            _write_bridge_pidfile(tmp_path, proc.pid)
            lines = (tmp_path / "bridge.pid").read_text().split("\n")
            assert int(lines[0]) == proc.pid
            # Line 2 is the kernel start time (present on Linux).
            assert int(lines[1]) == get_process_start_time(proc.pid)
        finally:
            proc.kill()
            proc.wait()


class TestIdentityGuard:
    def test_kills_when_start_time_matches(self, tmp_path):
        """A genuine bridge (recorded start time matches) IS reaped."""
        proc = _spawn_sleeper()
        try:
            _write_bridge_pidfile(tmp_path, proc.pid)
            _kill_stale_bridge_by_pidfile(tmp_path)
            assert _wait_dead(proc), "the real bridge process should be killed"
            assert not (tmp_path / "bridge.pid").exists()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_spares_recycled_pid_start_time_mismatch(self, tmp_path):
        """Alive PID whose start time changed (recycled) is NOT signalled."""
        proc = _spawn_sleeper()
        try:
            real_start = get_process_start_time(proc.pid)
            # Pidfile claims a different start time -> simulates a recycled PID.
            (tmp_path / "bridge.pid").write_text("{}\n{}".format(proc.pid, real_start + 1))
            _kill_stale_bridge_by_pidfile(tmp_path)
            assert not _wait_dead(proc, timeout=1.0), "recycled PID must survive"
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_legacy_pidfile_spares_non_bridge_cmdline(self, tmp_path):
        """Legacy pidfile (pid only): a PID that isn't node+session is spared."""
        proc = _spawn_sleeper()  # cmdline is just python -c ... — not a bridge
        try:
            (tmp_path / "bridge.pid").write_text(str(proc.pid))  # legacy: pid only
            _kill_stale_bridge_by_pidfile(tmp_path)
            assert not _wait_dead(proc, timeout=1.0), "stranger must survive"
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_legacy_pidfile_kills_matching_bridge_cmdline(self, tmp_path):
        """Legacy pidfile: a PID whose cmdline names node + session IS reaped."""
        # Shape the cmdline to look like the node bridge for this session.
        proc = _spawn_sleeper("node", str(tmp_path))
        try:
            (tmp_path / "bridge.pid").write_text(str(proc.pid))  # legacy: pid only
            _kill_stale_bridge_by_pidfile(tmp_path)
            assert _wait_dead(proc), "a cmdline-confirmed bridge should be killed"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_is_ours_false_for_dead_pid(self, tmp_path):
        assert _bridge_pid_is_ours(999999999, tmp_path, None) is False

    def test_missing_pidfile_is_noop(self, tmp_path):
        # No file -> must not raise.
        _kill_stale_bridge_by_pidfile(tmp_path)
