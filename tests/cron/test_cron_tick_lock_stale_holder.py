"""Stale tick-lock holder diagnosis (the cron tick lock file).

#129990: after an abrupt gateway termination a tick lock can stay lost for tens of
minutes. The losing tick previously (a) truncated the lock file on every open
(write mode), so no holder stamp could ever survive, and (b) logged the skip at
DEBUG only — the heartbeat silently froze while cron jobs skipped every tick.

The winner now stamps ``pid held_since`` into the lock file on acquisition, and a
contending tick reads that stamp to log at a staleness-matched severity:

* holder PID provably not running -> ERROR naming the dead pid (the lock
  outlived its holder; on handle-bound locks deleting the path cannot release it),
* live holder holding past the stale threshold -> WARNING with the hold age,
* fresh live holder -> DEBUG, exactly like before (normal multi-gateway races
  must not spam).

Tests hold the lock with a raw ``fcntl.flock`` on a separate fd and drive
``_acquire_tick_lock`` against it, so the admission path itself is exercised.
PID liveness is delegated to ``gateway.status._pid_exists``; the dead-holder case
patches that probe (it has its own coverage), while the live case runs unpatched
against the real current process.
"""

from __future__ import annotations

import logging
import os
import time

import pytest

fcntl = pytest.importorskip("fcntl")

import cron.scheduler as scheduler_mod


@pytest.fixture
def lock_file(tmp_path):
    return tmp_path / ".tick.lock"


def _hold_lock(lock_file, stamp: str):
    """Hold the tick lock on a private fd with ``stamp`` as the file body."""
    fd = open(lock_file, "a+", encoding="utf-8")
    fd.seek(0)
    fd.truncate()
    fd.write(stamp)
    fd.flush()
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _contend(lock_file, caplog):
    with caplog.at_level(logging.DEBUG, logger="cron.scheduler"):
        return scheduler_mod._acquire_tick_lock(lock_file)


def test_winner_stamps_holder_line(lock_file):
    fd = scheduler_mod._acquire_tick_lock(lock_file)
    try:
        parts = lock_file.read_text(encoding="utf-8").split()
        assert int(parts[0]) == os.getpid()
        assert time.time() - float(parts[1]) < 60
    finally:
        scheduler_mod._release_tick_lock(fd)


def test_contention_does_not_truncate_holder_stamp(lock_file, caplog):
    holder = _hold_lock(lock_file, "%d %f\n" % (os.getpid(), time.time()))
    try:
        assert _contend(lock_file, caplog) is None
        parts = lock_file.read_text(encoding="utf-8").split()
        assert int(parts[0]) == os.getpid()
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()


def test_contention_names_dead_holder_at_error(lock_file, caplog, monkeypatch):
    from gateway import status as gateway_status

    monkeypatch.setattr(gateway_status, "_pid_exists", lambda pid: False)
    holder = _hold_lock(lock_file, "%d %f\n" % (os.getpid(), time.time() - 400))
    try:
        assert _contend(lock_file, caplog) is None
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a provably-dead lock holder must surface at ERROR, not DEBUG silence"
    assert "NOT running" in errors[0].getMessage()
    assert str(os.getpid()) in errors[0].getMessage()


def test_contention_warns_on_aged_live_holder(lock_file, caplog):
    holder = _hold_lock(lock_file, "%d %f\n" % (os.getpid(), time.time() - 400))
    try:
        assert _contend(lock_file, caplog) is None
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "an over-aged live holder must surface at WARNING"
    assert "400" in warnings[0].getMessage()


def test_fresh_live_holder_stays_debug(lock_file, caplog):
    holder = _hold_lock(lock_file, "%d %f\n" % (os.getpid(), time.time()))
    try:
        assert _contend(lock_file, caplog) is None
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_unreadable_stamp_falls_back_to_mtime_age(lock_file, caplog):
    # Windows semantics: the locked byte range blocks a competing read, so the stamp is
    # unavailable and the hold age must fall back to the file mtime (which every winning
    # tick refreshes). Empty body + old mtime -> still a WARNING.
    holder = _hold_lock(lock_file, "")
    try:
        os.utime(lock_file, (time.time() - 400, time.time() - 400))
        assert _contend(lock_file, caplog) is None
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "stamp-less aged lock must still surface via mtime fallback"
    assert "unknown" in warnings[0].getMessage()
