"""Tests for the cron tick heartbeat — issue #57191.

Heartbeat is the cross-process signal the desktop dashboard backend reads to
decide whether to skip its own cron ticker while a real ``hermes gateway run``
is still firing on the same ``$HERMES_HOME``. Heartbeat write happens at the
end of every successful ``tick()``; ``gateway_cron_alive()`` reads mtime.
"""
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_sched = importlib.import_module("cron.scheduler")


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """Point ``HERMES_HOME`` at a tempdir so the heartbeat path is sandboxed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path, cron_dir


# ---------------------------------------------------------------------------
# gateway_cron_alive — staleness semantics
# ---------------------------------------------------------------------------


class TestGatewayCronAlive:
    def test_returns_false_when_heartbeat_missing(self, tmp_hermes_home):
        _, cron_dir = tmp_hermes_home
        # No `.tick.heartbeat` exists, .tick.lock present so the dir exists.
        assert _sched.gateway_cron_alive(lock_dir=cron_dir) is False

    def test_returns_true_when_heartbeat_fresh(self, tmp_hermes_home):
        _, cron_dir = tmp_hermes_home
        # Write a heartbeat that was "just now" by directly stat-edited mtime
        # via _write_tick_heartbeat ...
        _sched._write_tick_heartbeat(cron_dir)
        assert _sched.gateway_cron_alive(lock_dir=cron_dir) is True

    def test_returns_false_when_heartbeat_stale(self, tmp_hermes_home):
        _, cron_dir = tmp_hermes_home
        _sched._write_tick_heartbeat(cron_dir)
        # 9999 seconds in the past is well past the default 180s staleness.
        import time as _t
        hb = cron_dir / ".tick.heartbeat"
        old = _t.time() - 9999.0
        # os.utime accepts atime, mtime; we want mtime
        import os
        os.utime(hb, (old, old))
        assert _sched.gateway_cron_alive(lock_dir=cron_dir) is False

    def test_custom_staleness_window_changes_alive_verdict(self, tmp_hermes_home):
        _, cron_dir = tmp_hermes_home
        _sched._write_tick_heartbeat(cron_dir)
        import os
        import time as _t
        hb = cron_dir / ".tick.heartbeat"
        # 30 seconds ago
        m30 = _t.time() - 30.0
        os.utime(hb, (m30, m30))
        # Default window is 180s — still alive
        assert _sched.gateway_cron_alive(lock_dir=cron_dir) is True
        # Tighter 10s window — stale
        assert _sched.gateway_cron_alive(
            lock_dir=cron_dir, max_age_seconds=10
        ) is False

    def test_missing_cron_dir_is_false_not_raised(self, tmp_hermes_home):
        _, cron_dir = tmp_hermes_home
        # Point at a fresh, empty subdir that has no .tick.heartbeat
        empty = cron_dir / "_no_heartbeat"
        assert _sched.gateway_cron_alive(lock_dir=empty) is False


# ---------------------------------------------------------------------------
# _write_tick_heartbeat — atomic semantics + failure tolerance
# ---------------------------------------------------------------------------


class TestWriteTickHeartbeat:
    def test_writes_heartbeat_file(self, tmp_hermes_home):
        _, cron_dir = tmp_hermes_home
        _sched._write_tick_heartbeat(cron_dir)
        hb = cron_dir / ".tick.heartbeat"
        assert hb.exists()
        assert hb.read_text(encoding="utf-8").strip()

    def test_creates_dir_if_missing(self, tmp_path):
        deep = tmp_path / "fresh" / "deeper"
        _sched._write_tick_heartbeat(deep)
        assert (deep / ".tick.heartbeat").exists()

    def test_oserror_does_not_propagate(self, tmp_hermes_home):
        """A transient FS write failure must not abort the caller's tick."""
        _, cron_dir = tmp_hermes_home
        # Force open() to raise via a non-writable parent. We must use
        # skip-preserving-permissions on parent so we can still rmtree later.
        import stat
        cron_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read+exec only, no write
        try:
            # Should swallow OSError and not raise.
            _sched._write_tick_heartbeat(cron_dir)
        finally:
            cron_dir.chmod(stat.S_IRWXU)
        # And the heartbeat should not have been written (because the dir was
        # unwritable); the test just confirms we did not crash.

    def test_overwrite_each_call(self, tmp_hermes_home):
        """Subsequent writes should still produce a valid heartbeat file."""
        _, cron_dir = tmp_hermes_home
        for _ in range(3):
            _sched._write_tick_heartbeat(cron_dir)
        assert _sched.gateway_cron_alive(lock_dir=cron_dir) is True
