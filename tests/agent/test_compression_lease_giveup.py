"""The compression lease refresher gives up on a failure COUNT (#97948).

Symptom B of #97948: a large-session compression runs for 11+ minutes and then
aborts at publication with ``Compression lease lost before publication`` /
``failure_class: session_split_failed``, rolls back, and the next turn
re-triggers the same work.

The lease IS refreshed while the summary streams, so these controls pin how a
still-running compression ends up without one: ``_CompressionLockLeaseRefresher
._run`` (``agent/conversation_compression.py:2271``) breaks permanently after
``max(1, int(ttl / interval))`` consecutive falsy refreshes -- 5 on production
values. The durable hazard is that COUNT, not any particular wall clock: the
loop never consults ``expires_at`` or elapsed time.

The wall-clock give-up window is therefore not a constant, because ``_run()``
starts waiting its interval only AFTER each refresh call returns:

* **Zero-latency floor.** With refreshes that fail instantly, attempts land at
  ``t = 0, 60, 120, 180, 240``; the break-triggering 5th failure is at
  ``(threshold - 1) * interval = 240s``, one interval short of the 300s TTL.
  This is the minimum, and it is what ``TestGiveUpTiming``'s first case
  measures on the real loop.
* **Contended ceiling.** ``SessionDB.refresh_compression_lock``
  (``hermes_state.py:7821``) runs ``_execute_write(_do)`` on the routine
  ``_WRITE_PATIENCE_S = 20.0`` budget (``hermes_state.py:4426``) and converts
  an exhausted retry into ``False``, so a fully contended attempt burns that
  whole budget before returning. Attempts then start at ``0, 80, 160, 240,
  320`` and the 5th ``False`` returns around **340s** -- past the TTL, not
  short of it.

So neither "the window is 240s" nor "write contention shorter than the TTL is
fatal" is established here: the window is
``(threshold - 1) * (interval + call_latency) + call_latency``, and
``TestGiveUpTiming`` measures both ends of that range on the real thread
rather than restating a formula. (Both framings are corrections from review
on #98867; the first version of this file asserted
``threshold * interval == TTL``.)

What survives independent of latency is the contradiction with
``SessionDB.refresh_compression_lock``, which decides ownership by the
``holder`` column alone precisely so a starved owner can revive its
still-unclaimed row "on the next tick" -- the loop guarantees there is no
next tick.

Provenance caveat: ``FlakyDB`` returns falsy outcomes directly. It does not
drive ``_execute_write``, create real SQLite contention, or record production
refresh telemetry, so nothing here proves the reported Windows attempt
actually experienced five consecutive refresh failures. This is a reachable
candidate mechanism for that report, not its established root cause.

#99216 has since landed the production repair: ``publish_compression_child``
now takes one final in-transaction lease refresh (``WHERE session_id AND
holder``, same conn as the expiry check) immediately before publication, so a
refresher that gave up but whose row nobody else claimed still lets the
rotation through. This file is diagnostic/provenance evidence for that fix,
not an independent settlement: ``TestGiveUpWindow`` / ``TestGiveUpTiming`` /
``TestRefresherStopsPermanently`` / ``TestOwnershipStaysRecoverable`` still
pin the refresher-thread defect itself (#99216 does not touch
``_CompressionLockLeaseRefresher._run``), while
``TestPublicationAfterRefresherGivesUp`` is the regression guard for the
merged fix -- it drives the real refresher to its real give-up point and
then asserts publication now SUCCEEDS when nobody stole the lock,
complementing the wrong-holder adversarial cases already covered by
#99216's own ``tests/state/test_compression_lease_refresh_before_publish.py``.
"""

import time

import pytest

from agent.conversation_compression import _CompressionLockLeaseRefresher
from hermes_state import CompressionSessionBusyError, SessionDB

# agent/conversation_compression.py:3018 —
# `_lock_ttl = float(getattr(agent, "_compression_lock_ttl_seconds", 300.0) or 300.0)`
PROD_TTL_S = 300.0
# _CompressionLockLeaseRefresher.__init__: max(1.0, min(60.0, ttl / 2.0))
PROD_INTERVAL_S = 60.0

SESSION = "sess-parent"
HOLDER = "holder-A"

# Compressed timescale with the same threshold arithmetic as production
# (ttl / interval == 5 either way); keeps the suite around ten seconds.
# INTERVAL_S must stay >= the constructor's 0.1 floor, or the clamp
# silently changes the threshold this file is about. 0.2s (rather than the
# original 0.1s) leaves the off-by-one timing assertions in TestGiveUpTiming
# a full interval's margin against scheduling jitter.
TTL_S = 1.0
INTERVAL_S = 0.2


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    try:
        yield d
    finally:
        d.close()


class FlakyDB:
    """Fails the first *n* refreshes, then delegates to the real SessionDB.

    A falsy return is the only thing the refresher can observe, and on the
    real path it covers both a genuinely lost row and a write that escaped
    ``_execute_write``'s retry budget under contention.  That ambiguity is the
    defect these tests describe.

    ``fail_latency_s`` models how long a failing call takes to return.  It is
    the difference between the two ends of the give-up range: production's
    ``refresh_compression_lock`` spends up to ``_WRITE_PATIENCE_S`` inside
    ``_execute_write`` before converting an exhausted retry into ``False``,
    and ``_run()`` starts its interval wait only after that return.  Both
    ``attempt_times`` (entry) and ``return_times`` (exit) are recorded so a
    test can measure the schedule instead of assuming instant calls.
    """

    def __init__(self, real, fail_first, fail_latency_s=0.0):
        self._real = real
        self._remaining = fail_first
        self._fail_latency_s = fail_latency_s
        self.attempts = 0
        self.granted = 0
        self.attempt_times: list = []
        self.return_times: list = []

    def refresh_compression_lock(self, session_id, holder, ttl_seconds=PROD_TTL_S):
        self.attempts += 1
        self.attempt_times.append(time.monotonic())
        if self._remaining > 0:
            self._remaining -= 1
            if self._fail_latency_s:
                time.sleep(self._fail_latency_s)
            self.return_times.append(time.monotonic())
            return False
        self.granted += 1
        result = self._real.refresh_compression_lock(
            session_id, holder, ttl_seconds=ttl_seconds
        )
        self.return_times.append(time.monotonic())
        return result


def _threshold(ttl, interval):
    """The refresher's own give-up rule, restated — including the clamps.

    ``__init__`` floors the interval at 0.1s before deriving
    ``_max_consecutive_failures``, so a test that passes a smaller one
    would be measuring a different threshold than it declares.
    """
    return max(1, int(ttl / max(0.1, interval)))


def _drive(real_db, *, fail_first, ticks, release=True, fail_latency_s=0.0):
    """Run a real refresher over a real, genuinely held lock.

    The lock MUST be acquired first: ``refresh_compression_lock`` updates
    ``compression_locks WHERE session_id = ? AND holder = ?``, so without a
    row every delegated refresh returns False and the loop would break for
    the wrong reason — making the counterfactual below pass vacuously.

    Acquires with the SAME ``TTL_S`` the refresher itself uses, not
    ``PROD_TTL_S``: a caller that keeps the row alive past this call (via
    ``release=False``) needs it to have actually expired by the time the
    refresher gave up, not still be sitting on a fresh 300-second lease for
    another ~299s — the mismatch that made the old publication tests never
    reproduce "refresher stops -> lease expires -> publication rejects"
    (review on #98867).

    ``fail_latency_s`` makes each failing refresh take that long to return,
    the way a contended ``_execute_write`` does; ``ticks`` then counts whole
    call-plus-interval cycles rather than bare intervals.
    """
    assert real_db.try_acquire_compression_lock(
        SESSION, HOLDER, ttl_seconds=TTL_S
    )
    flaky = FlakyDB(real_db, fail_first, fail_latency_s=fail_latency_s)
    refresher = _CompressionLockLeaseRefresher(
        flaky, SESSION, HOLDER, TTL_S, INTERVAL_S
    ).start()
    try:
        time.sleep((INTERVAL_S + fail_latency_s) * (ticks + 0.5))
    finally:
        refresher.stop()
        if release:
            real_db.release_compression_lock(SESSION, HOLDER)
    return flaky


class TestGiveUpWindow:
    """The give-up rule, on the values production actually uses.

    The rule itself is a failure COUNT. These cases bound the wall-clock
    window that count produces; ``TestGiveUpTiming`` then measures both
    bounds on the real loop rather than leaving them as arithmetic.
    """

    def test_production_threshold_is_five_failures(self):
        assert _threshold(PROD_TTL_S, PROD_INTERVAL_S) == 5

    def test_the_zero_latency_window_is_one_interval_short_of_the_ttl(self):
        """Lower bound: five instant failures, four intervals apart = 240s.

        The first refresh fires at t=0; the loop waits one interval before
        each subsequent attempt, so with refreshes that fail immediately the
        5th (break-triggering) failure lands at ``(5 - 1) * 60s = 240s``.
        That is the FLOOR of the give-up window, reached only when refresh
        calls cost nothing — not the window a contended production refresh
        actually experiences (see the next case).
        """
        threshold = _threshold(PROD_TTL_S, PROD_INTERVAL_S)
        window = (threshold - 1) * PROD_INTERVAL_S
        assert window == 240.0
        assert window < PROD_TTL_S

    def test_exhausted_write_patience_pushes_the_window_past_the_ttl(self):
        """Upper bound: failing calls are not free.

        ``refresh_compression_lock`` runs ``_execute_write(_do)`` on the
        routine ``_WRITE_PATIENCE_S`` budget and converts an exhausted retry
        into ``False``, and ``_run()`` starts its interval wait only after
        that return. Fully contended attempts therefore START at
        ``0, 80, 160, 240, 320`` and the 5th ``False`` comes back around
        340s — PAST the 300s TTL. A stall that clears before then can still
        be caught by the 5th attempt and recover, so "write contention
        shorter than the TTL is fatal" does not follow from this code.
        """
        threshold = _threshold(PROD_TTL_S, PROD_INTERVAL_S)
        patience = SessionDB._WRITE_PATIENCE_S
        window = (threshold - 1) * (PROD_INTERVAL_S + patience) + patience
        assert window == 340.0
        assert window > PROD_TTL_S

    def test_a_long_compression_outlives_even_the_contended_window(self):
        """The aborted attempt in the report ran 710s (total_duration_ms).

        That is long enough to contain the give-up window at either end of
        the range, which is what makes this mechanism reachable for that
        report. It does NOT show the reported attempt's refreshes actually
        failed — no refresh-outcome telemetry exists for that run.
        """
        threshold = _threshold(PROD_TTL_S, PROD_INTERVAL_S)
        contended = (threshold - 1) * (
            PROD_INTERVAL_S + SessionDB._WRITE_PATIENCE_S
        ) + SessionDB._WRITE_PATIENCE_S
        assert 710.0 > contended > (threshold - 1) * PROD_INTERVAL_S


class TestGiveUpTiming:
    """Drives the real loop and measures when it actually gives up, instead
    of trusting the restated formula in ``TestGiveUpWindow`` (review on
    #98867: the first version of this file asserted
    ``threshold * interval``, but ``_run()`` waits an interval AFTER each
    refresh, so with instant refreshes the break-triggering attempt lands
    one interval earlier — and with slow ones, later).
    """

    def test_instant_failures_give_up_one_interval_early(self, db):
        """The floor: refresh calls that cost nothing."""
        threshold = _threshold(TTL_S, INTERVAL_S)
        flaky = _drive(db, fail_first=threshold, ticks=threshold + 3)
        assert len(flaky.attempt_times) == threshold
        elapsed = flaky.attempt_times[-1] - flaky.attempt_times[0]
        # threading.Event.wait() timing has real jitter; the assertion only
        # needs to distinguish (threshold - 1) intervals from threshold — a
        # full INTERVAL_S apart.
        assert elapsed == pytest.approx(
            (threshold - 1) * INTERVAL_S, abs=INTERVAL_S * 0.5
        )
        assert elapsed < threshold * INTERVAL_S - INTERVAL_S * 0.25

    def test_the_window_grows_with_refresh_call_latency(self, db):
        """The controlled-latency witness: 240s is a floor, not the window.

        Each failing refresh is made to take one interval to return, the way
        a contended ``_execute_write`` burns its patience budget before
        answering ``False``. The measured give-up point moves to
        ``(threshold - 1) * (interval + latency) + latency`` — strictly past
        the zero-latency floor the previous case measures, which is exactly
        why the production window cannot be quoted as 240s.
        """
        threshold = _threshold(TTL_S, INTERVAL_S)
        latency = INTERVAL_S
        flaky = _drive(
            db, fail_first=threshold, ticks=threshold + 3, fail_latency_s=latency
        )
        assert len(flaky.attempt_times) == threshold
        # Entry of the first attempt to RETURN of the break-triggering one:
        # the loop dies when that last call answers, not when it starts.
        span = flaky.return_times[-1] - flaky.attempt_times[0]
        expected = (threshold - 1) * (INTERVAL_S + latency) + latency
        assert span == pytest.approx(expected, abs=(INTERVAL_S + latency) * 0.5)
        assert span > (threshold - 1) * INTERVAL_S


class TestRefresherStopsPermanently:
    def test_loop_exits_at_the_threshold(self, db):
        threshold = _threshold(TTL_S, INTERVAL_S)
        flaky = _drive(db, fail_first=threshold, ticks=threshold * 2)
        assert flaky.attempts == threshold

    def test_no_refresh_is_granted_after_the_threshold(self, db):
        threshold = _threshold(TTL_S, INTERVAL_S)
        flaky = _drive(db, fail_first=threshold, ticks=threshold * 2)
        assert flaky.granted == 0

    def test_one_failure_below_the_threshold_recovers(self, db):
        """Isolates the give-up rule: the same stall, one failure shorter."""
        threshold = _threshold(TTL_S, INTERVAL_S)
        flaky = _drive(db, fail_first=threshold - 1, ticks=threshold + 5)
        assert flaky.granted > 0
        assert flaky.attempts > threshold


class TestOwnershipStaysRecoverable:
    """The DB contract the loop stops honouring."""

    def test_an_expired_lease_is_still_revivable_by_its_owner(self, db):
        assert db.try_acquire_compression_lock(SESSION, HOLDER, ttl_seconds=0.05)
        time.sleep(0.2)
        assert db.refresh_compression_lock(
            SESSION, HOLDER, ttl_seconds=PROD_TTL_S
        ) is True
        db.release_compression_lock(SESSION, HOLDER)

    def test_a_reclaimed_lease_is_not_revivable(self, db):
        """The give-up rule's legitimate case: ownership genuinely changed."""
        assert db.try_acquire_compression_lock(SESSION, HOLDER, ttl_seconds=0.05)
        time.sleep(0.2)
        assert db.try_acquire_compression_lock(SESSION, "holder-B", ttl_seconds=60.0)
        assert db.refresh_compression_lock(SESSION, HOLDER) is False
        db.release_compression_lock(SESSION, "holder-B")


class TestPublicationAfterRefresherGivesUp:
    """Regression guard for #99216, not the original bug pin.

    #99216's own ``tests/state/test_compression_lease_refresh_before_publish
    .py`` seeds the lock row directly and covers the wrong-holder/adversarial
    side of the pre-publication refresh. This drives the REAL
    ``_CompressionLockLeaseRefresher`` thread through its real give-up timing
    first, over the same TTL/holder/row that publication then checks, so the
    row under test is the one an actual give-up event would leave behind.
    """

    def test_publish_succeeds_when_nobody_stole_the_lock(self, db):
        db.create_session(session_id=SESSION, source="desktop", model="m")
        threshold = _threshold(TTL_S, INTERVAL_S)
        flaky = _drive(db, fail_first=threshold, ticks=threshold + 5, release=False)
        assert flaky.granted == 0  # the refresher really did give up
        # _drive already slept past TTL_S with no successful refresh, so the
        # un-refreshed lease has genuinely expired by now.

        db.publish_compression_child(
            parent_session_id=SESSION,
            child_session_id="sess-child",
            source="desktop",
            messages=[{"role": "user", "content": "handoff"}],
            compression_lock_holder=HOLDER,
            require_compression_lease=True,
            require_lease_refresh=True,
            lease_ttl_seconds=TTL_S,
        )
        assert db.get_session("sess-child") is not None
        assert db.get_session(SESSION)["ended_at"] is not None
        db.release_compression_lock(SESSION, HOLDER)

    def test_publish_still_refuses_when_another_holder_won_the_row(self, db):
        """The give-up rule's legitimate case, still refused post-#99216:
        the row was genuinely reclaimed, not merely left un-refreshed."""
        db.create_session(session_id=SESSION, source="desktop", model="m")
        threshold = _threshold(TTL_S, INTERVAL_S)
        _drive(db, fail_first=threshold, ticks=threshold + 5, release=False)
        assert db.try_acquire_compression_lock(SESSION, "holder-B", ttl_seconds=60.0)

        with pytest.raises(CompressionSessionBusyError, match="lease lost"):
            db.publish_compression_child(
                parent_session_id=SESSION,
                child_session_id="sess-child",
                source="desktop",
                messages=[{"role": "user", "content": "handoff"}],
                compression_lock_holder=HOLDER,
                require_compression_lease=True,
                require_lease_refresh=True,
                lease_ttl_seconds=TTL_S,
            )
        assert db.get_session("sess-child") is None
        assert db.get_session(SESSION)["ended_at"] is None
        db.release_compression_lock(SESSION, "holder-B")

    def test_a_live_lease_publishes_normally(self, db):
        """Negative control: the check is about the lease, nothing else."""
        db.create_session(session_id=SESSION, source="desktop", model="m")
        assert db.try_acquire_compression_lock(SESSION, HOLDER, ttl_seconds=PROD_TTL_S)
        db.publish_compression_child(
            parent_session_id=SESSION,
            child_session_id="sess-child",
            source="desktop",
            messages=[{"role": "user", "content": "handoff"}],
            compression_lock_holder=HOLDER,
            require_compression_lease=True,
        )
        assert db.get_session("sess-child") is not None
        db.release_compression_lock(SESSION, HOLDER)
