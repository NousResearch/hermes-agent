"""Regression test for the cron tick lock-skip log level (#53720).

Background: when two scheduler instances race on the same ``.tick.lock``
(e.g. a gateway-managed scheduler plus a desktop in-process scheduler), the
loser correctly yields to the file-lock holder and skips the tick. The
"Tick skipped — another instance holds the lock" line was logged at ``DEBUG``,
which is below the default ``INFO`` log level, so the suppression was invisible
in production logs. It must be ``INFO`` so operators can see dual-scheduler
behavior without reconfiguring logging.
"""

import fcntl
import logging

import pytest

from cron import scheduler


def test_tick_skip_is_logged_at_info(tmp_path, monkeypatch, caplog):
    """The lock-skip path must emit at INFO, not DEBUG."""
    lock_dir = tmp_path / "cron"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / ".tick.lock"

    monkeypatch.setattr(
        scheduler, "_get_lock_paths", lambda: (lock_dir, lock_file)
    )

    # Pre-acquire the cross-process lock from this process so the ``tick()``
    # call hits the non-blocking acquire failure path and skips.
    holder = open(lock_file, "w", encoding="utf-8")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            executed = scheduler.tick(verbose=False)
        assert executed == 0
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()

    skip_records = [
        r for r in caplog.records
        if "Tick skipped" in r.getMessage() and "holds the lock" in r.getMessage()
    ]
    assert skip_records, "expected a 'Tick skipped' log record"
    assert skip_records[0].levelno == logging.INFO, (
        "tick lock-skip must be logged at INFO for dual-scheduler visibility (#53720)"
    )


def test_tick_skip_not_logged_at_debug_only(tmp_path, monkeypatch, caplog):
    """The skip message must be visible at the INFO threshold (default level).

    A regression to ``DEBUG`` would drop it under the default ``INFO`` level,
    reproducing the invisibility reported in #53720.
    """
    lock_dir = tmp_path / "cron"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / ".tick.lock"

    monkeypatch.setattr(
        scheduler, "_get_lock_paths", lambda: (lock_dir, lock_file)
    )

    holder = open(lock_file, "w", encoding="utf-8")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        # No at_level override -> caplog captures at WARNING by default; set
        # INFO explicitly to model the production default log level.
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            scheduler.tick(verbose=False)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()

    messages = [r.getMessage() for r in caplog.records]
    assert any("Tick skipped" in m for m in messages), (
        "tick lock-skip not visible at INFO log level; would be silently dropped "
        "in production (#53720)"
    )