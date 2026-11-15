"""Turn-liveness watchdog vs. state.db write-retry receipts (RCA 2026-09-28).

With transcript patience at storm-class scale (240s), a turn blocked in its own
persistence flush accrues "no progress" toward the 600s liveness abort — the
skeptic's CONCERN 1 on PLAN-state-db-persistence-fix-2026-09-28. The mitigation:
every jittered lock-retry sleep stamps ``hermes_state.note_write_retry()`` and the
watchdog's sampler treats a fresh stamp as forward progress.

Polarity lives inside the tests: a fresh receipt must mask the abort (the fix);
a stale receipt must leave the abort fully armed (the pre-fix behavior, kept).
"""

import threading
import time
from types import SimpleNamespace

import pytest

import hermes_state
from agent.turn_liveness import TurnLivenessWatchdog


class _FakeAgent(SimpleNamespace):
    pass


def _make_watchdog(idle_seconds, aborts, timeout_s=600.0, stop_event=None):
    agent = _FakeAgent(
        _turn_liveness_activity_generation=0,
        _last_activity_ts=time.time() - idle_seconds,
        _emit_warning=lambda text: None,
    )
    activity_lock = threading.Lock()
    return TurnLivenessWatchdog(
        agent,
        session_id="s-storm",
        timeout_s=timeout_s,
        poll_s=15.0,
        stop_event=stop_event if stop_event is not None else threading.Event(),
        activity_lock=activity_lock,
        is_turn_active=lambda: True,
        commit_abort=lambda snapshot, message: aborts.append((snapshot, message)) or True,
        deactivate_turn=lambda: None,
    ), agent


class TestLivenessWriteRetryReceipt:
    def test_fresh_write_retry_receipt_masks_the_stall_abort(self, monkeypatch):
        """A turn blocked in a persistence flush with retries landing continuously
        must NOT be aborted for silence — that was a new death path the 240s
        patience would have opened (incident class: turn waits 240s, watchdog
        fires at 600s idle accrued before+during the wait)."""
        aborts: list = []
        monkeypatch.setattr(hermes_state, "_LAST_WRITE_RETRY_MONOTONIC", time.monotonic())
        wd, _ = _make_watchdog(idle_seconds=700.0, aborts=aborts)
        result = wd._tick()
        assert aborts == [], f"abort committed despite fresh write-retry receipt: {aborts}"
        assert result is None  # watchdog keeps watching, abort not committed

    def test_stale_receipt_keeps_the_abort_fully_armed(self, monkeypatch):
        """Polarity: with no fresh retry receipt (nothing retrying the write queue),
        a 700s-idle turn is a genuine silent wedge and the abort must fire exactly
        as before the fix."""
        aborts: list = []
        monkeypatch.setattr(hermes_state, "_LAST_WRITE_RETRY_MONOTONIC", 0.0)
        wd, _ = _make_watchdog(idle_seconds=700.0, aborts=aborts)
        result = wd._tick()
        assert aborts, "pre-fix behavior must survive: stale receipt + long idle aborts"
        assert result is False  # watchdog stopped after committed abort
        assert "no progress" in aborts[0][1]

    def test_subthreshold_idle_is_untouched_by_the_receipt(self, monkeypatch):
        """A normally-active turn (5s idle) must not even consult the receipt:
        normal accounting never changes."""
        aborts: list = []
        monkeypatch.setattr(hermes_state, "_LAST_WRITE_RETRY_MONOTONIC", 0.0)
        wd, _ = _make_watchdog(idle_seconds=5.0, aborts=aborts)
        result = wd._tick()
        assert aborts == []
        assert result is None


def test_retry_sleep_stamps_the_receipt(tmp_path):
    """The stamp actually fires from the retry path: ``_sleep_before_write_retry``
    (invoked on every locked BEGIN IMMEDIATE) must leave a fresh receipt behind —
    the liveness mask is only as real as this wiring."""
    from hermes_state import SessionDB, write_retry_in_progress

    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert d._sleep_before_write_retry(deadline=time.monotonic() + 5.0, patience_s=5.0)
        assert write_retry_in_progress(max_age_s=30.0), "retry sleep must stamp the receipt"
    finally:
        d.close()
