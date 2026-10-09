"""A SQLite write lock on state.db is a missed renewal tick for the review lease, not a lost row.

``refresh_session_turn_lease`` runs through ``SessionDB._execute_write`` and re-raises
``sqlite3.OperationalError`` once its write patience is exhausted on a WAL write lock, so any
writer holding state.db that long (a compaction publish, a sweep, another session's flush) used
to read as ``review_lease_lost``: the fork was fenced and hard-interrupted and a deferred review
dropped for good, for a row whose ``expires_at`` was still most of a TTL away. The review lease
now mirrors the foreground ``DurableTurnLease``: a lock is a missed tick while the next attempt
can still land before the row's authority runs out; a renewal locked out through that whole
authority stops the fork under its own slug (``review_lease_renewal_locked``) and WITHOUT the
yield mark, so a deferred review is requeued rather than dropped.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import types

import pytest

from agent import background_review as background_review_module
from agent import review_admission
from agent.background_review_lease import REVIEW_LEASE_TTL_SECONDS
from agent.turn_facade_lease import LEASE_TTL_SECONDS, _REFRESH_EXPIRY_MARGIN_S
from hermes_state import SessionDB


@pytest.fixture(autouse=True)
def clear_review_admission_state():
    with review_admission._lock:
        review_admission._review_runs.clear()
    yield
    with review_admission._lock:
        review_admission._review_runs.clear()


def _review_lease(tmp_path):
    """An admitted review lease on a real store, with the fork's interrupt observable."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared-session", source="test")
    interrupted = threading.Event()
    fork = types.SimpleNamespace(
        hard_interrupt=lambda *_a, **_k: interrupted.set(),
        release_clients=lambda: None,
    )
    run = background_review_module._BackgroundReviewRun()
    assert run.begin_request(fork) is True
    lease, reason = background_review_module._try_acquire_durable_review_lease(
        types.SimpleNamespace(_session_db=db), fork, "shared-session", run
    )
    assert reason is None and lease is not None
    return db, lease, run, interrupted


def _locked() -> sqlite3.OperationalError:
    return sqlite3.OperationalError(
        "database is locked (another Hermes process held the state.db write lock for over 20s)"
    )


def _lines(caplog, slug: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if slug in r.getMessage()]


def test_review_lease_authority_is_its_own_ttl(tmp_path):
    """The lock arithmetic runs against the REVIEW row's 60s TTL, never the foreground's 300s:
    the deadline starts at the admitted row's expiry and only a successful renewal moves it."""
    db, lease, run, _interrupted = _review_lease(tmp_path)
    try:
        now = time.time()
        assert now < lease._authority_deadline <= now + REVIEW_LEASE_TTL_SECONDS
        assert lease._authority_deadline < now + LEASE_TTL_SECONDS
        before = lease._authority_deadline
        assert lease.refresh_tick() is None
        assert lease._authority_deadline >= before
    finally:
        lease.stop_refresher()
        lease.release()


def test_renewal_lock_is_a_missed_tick_not_a_loss(tmp_path, caplog):
    """One locked renewal: the tick keeps the timer, the fork runs on, nothing is recorded on the
    run, and the next tick renews the row. The renewal waits no longer than the authority
    allows (never the store's full default patience past the row's expiry)."""
    db, lease, run, interrupted = _review_lease(tmp_path)
    real_refresh = db.refresh_session_turn_lease
    calls: list[dict] = []

    def refresh(session_id, holder, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _locked()
        return real_refresh(session_id, holder, **kwargs)

    db.refresh_session_turn_lease = refresh
    try:
        with caplog.at_level("INFO"):
            assert lease.refresh_tick() is None
        assert not interrupted.is_set()
        assert not run.cancel_requested.is_set()
        assert run.lease_yield_reason is None
        assert _lines(caplog, review_admission.REASON_LEASE_LOST) == []
        assert _lines(caplog, review_admission.REASON_LEASE_RENEWAL_LOCKED) == []
        assert lease.stop.is_set() is False

        expires_before = db.session_turn_lease_expires_at(
            "shared-session", lease.holder
        )
        assert lease.refresh_tick() is None
        assert len(calls) == 2
        assert (
            db.session_turn_lease_expires_at("shared-session", lease.holder)
            >= expires_before
        )
        assert not interrupted.is_set()
        for kwargs in calls:
            assert (
                0
                < kwargs["patience_s"]
                <= REVIEW_LEASE_TTL_SECONDS - _REFRESH_EXPIRY_MARGIN_S
            )
    finally:
        lease.stop_refresher()
        lease.release()
    # The retry is logged with the hashed owner, never the session id.
    retry_lines = [
        r.getMessage() for r in caplog.records if "lock" in r.getMessage().lower()
    ]
    assert retry_lines and all(lease._owner in line for line in retry_lines)
    assert all("shared-session" not in line for line in retry_lines)


def test_renewal_locked_out_through_the_authority_stops_the_fork_without_a_yield(
    tmp_path, caplog, monkeypatch
):
    """A renewal that cannot land before the row expires stops the fork (a successor may reclaim
    the row: no overlap), logged ONCE under its own slug with the hashed owner — not as a lost
    lease — and leaves the run's yield mark clear, so a deferred review is requeued."""
    db, lease, run, interrupted = _review_lease(tmp_path)

    def always_locked(session_id, holder, **kwargs):
        raise _locked()

    db.refresh_session_turn_lease = always_locked
    # Half a second short of one interval plus the expiry margin: this attempt still fits the
    # authority, but the next one could not land before the row expires.
    lease._authority_deadline = (
        time.time() + lease.refresh_interval + _REFRESH_EXPIRY_MARGIN_S - 0.5
    )
    try:
        with caplog.at_level("INFO"):
            assert lease.refresh_tick() is None
            assert interrupted.is_set()
            assert run.cancel_requested.is_set()
            assert run.lease_yield_reason is None
            assert (
                lease.refresh_tick() is None
            )  # logged once; the tick stays the escalation clock
    finally:
        lease.stop_refresher()
        lease.release()

    locked_lines = _lines(caplog, review_admission.REASON_LEASE_RENEWAL_LOCKED)
    assert len(locked_lines) == 1, caplog.text
    assert lease._owner in locked_lines[0]
    assert "shared-session" not in locked_lines[0]
    assert _lines(caplog, review_admission.REASON_LEASE_LOST) == []

    # The requeue policy sees a cancelled run with no yield mark: requeued, not dropped.
    import run_agent
    from agent import review_idle_queue

    enqueued: list[dict] = []
    monkeypatch.setattr(
        review_idle_queue.QUEUE,
        "enqueue",
        lambda _agent, _key, kwargs, **_opts: enqueued.append(kwargs),
    )
    agent = types.SimpleNamespace(
        session_id="shared-session",
        _REVIEW_REQUEUE_MAX_ATTEMPTS=run_agent.AIAgent._REVIEW_REQUEUE_MAX_ATTEMPTS,
    )
    agent._requeue_deferred_review = types.MethodType(
        run_agent.AIAgent._requeue_deferred_review, agent
    )
    with caplog.at_level("INFO"):
        run_agent.AIAgent._maybe_requeue_preempted_review(
            agent,
            run,
            {
                "task_cfg": {"defer": "auto"},
                "focus": None,
                "_requeue_attempts": 1,
                "_idle_queue_origin": True,
                "_review_session_id": "shared-session",
            },
        )
    assert len(enqueued) == 1
    assert _lines(caplog, review_admission.REASON_DROPPED_AFTER_LEASE_YIELD) == []


def test_lost_row_is_still_a_loss(tmp_path, caplog):
    """The lock tolerance narrows nothing else: a holder-fenced miss (rowcount 0) is a lost lease,
    logged once, recorded on the run so a deferred review is dropped."""
    db, lease, run, interrupted = _review_lease(tmp_path)
    db.release_session_turn_lease(
        "shared-session", lease.holder
    )  # reclaimed under the fork
    try:
        with caplog.at_level("INFO"):
            assert lease.refresh_tick() is None
            assert lease.refresh_tick() is None
    finally:
        lease.stop_refresher()
    assert interrupted.is_set()
    assert run.lease_yield_reason == review_admission.REASON_LEASE_LOST
    assert len(_lines(caplog, review_admission.REASON_LEASE_LOST)) == 1
    assert lease.refresh_tick() is False


def test_lock_after_the_fork_released_the_row_is_not_a_loss(tmp_path, caplog):
    """The fork's finally sets ``stop`` before releasing: a renewal that was blocked by the lock
    across that release ends the timer quietly."""
    db, lease, run, interrupted = _review_lease(tmp_path)

    def locked_across_release(session_id, holder, **kwargs):
        lease.stop_refresher()
        raise _locked()

    db.refresh_session_turn_lease = locked_across_release
    with caplog.at_level("INFO"):
        assert lease.refresh_tick() is False
    lease.release()
    assert not interrupted.is_set()
    assert not run.cancel_requested.is_set()
    assert _lines(caplog, review_admission.REASON_LEASE_LOST) == []
    assert _lines(caplog, review_admission.REASON_LEASE_RENEWAL_LOCKED) == []
