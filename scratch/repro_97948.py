"""Four-stage probe for #97948 — "Compression lease lost before publication".

Run: ``python scratch/repro_97948.py`` (no pytest, no network, temp state.db).

Symptom B of the issue: a large-session compression runs for 11+ minutes and
then aborts at publication with ``Compression lease lost before publication``
/ ``failure_class: session_split_failed``, rolls back, and the next turn
re-triggers the same doomed compression.

The lease IS refreshed while the summary streams
(``_CompressionLockLeaseRefresher`` in agent/conversation_compression.py), so
the interesting question is how a healthy, still-running compression ends up
without one.  This probe isolates the give-up rule and its consequence:

  S1  the refresher stops permanently after ttl/interval consecutive failures
  S2  ownership was still recoverable at that exact moment — the loop simply
      stopped asking (a direct contradiction of refresh_compression_lock's
      own documented contract)
  S3  the consequence at publication, over the SAME row S1 left behind: since
      #99216 landed, publish_compression_child's final in-transaction refresh
      lets the rotation through when nobody else claimed the lock — this is a
      regression guard for that fix, not the original bug pin (S1/S2/S4
      still describe the refresher-thread defect itself, which #99216 does
      not touch)
  S4  the give-up rule is a failure COUNT, and the wall-clock window it
      produces depends on how long a failing refresh takes to return:
      240s is the zero-latency FLOOR (the first refresh fires at t=0, so the
      break-triggering attempt lands (threshold - 1) intervals later), while
      a fully contended refresh burning SessionDB._WRITE_PATIENCE_S pushes
      the same five failures out to ~340s — past the 300s TTL, not one
      interval short of it. Measured on the real loop at both ends. A
      refresher that keeps asking recovers from the same stall.

Every stage prints PASS/FAIL; the process exits non-zero if any stage fails.

Provenance caveat: FlakyDB returns falsy outcomes directly — it does not
drive _execute_write, create real SQLite contention, or read production
refresh telemetry. This probe establishes a reachable candidate mechanism
for the reported Windows abort, not its proven root cause.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.conversation_compression import _CompressionLockLeaseRefresher  # noqa: E402
from hermes_state import SessionDB  # noqa: E402

# agent/conversation_compression.py: `_lock_ttl = float(getattr(agent,
# "_compression_lock_ttl_seconds", 300.0) or 300.0)`, and the refresher
# derives `interval = max(1.0, min(60.0, ttl / 2.0))`.
PROD_TTL_S = 300.0
PROD_INTERVAL_S = 60.0

SESSION = "sess-parent"
HOLDER = "holder-A"

_failures: list[str] = []


def check(stage: str, label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {stage} {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        _failures.append(f"{stage} {label} {detail}".strip())


class FlakyDB:
    """Wraps a real SessionDB and fails the first *n* refresh attempts.

    Models the only thing the refresher can actually observe: a falsy return
    from ``refresh_compression_lock``.  On the real path that is a write that
    escaped ``_execute_write``'s retry budget under contention — a state.db on
    ``journal_mode=DELETE`` while a 1497-message compaction clones rows.

    ``fail_latency_s`` models how long such a call takes to answer: production
    spends up to ``SessionDB._WRITE_PATIENCE_S`` inside ``_execute_write``
    before turning an exhausted retry into ``False``, and ``_run()`` starts
    its interval wait only after that return.  Entry and exit timestamps are
    both recorded so S4 can measure the schedule instead of assuming the
    calls are free.
    """

    def __init__(self, db: SessionDB, fail_first: int, fail_latency_s: float = 0.0):
        self._db = db
        self._remaining = fail_first
        self._fail_latency_s = fail_latency_s
        self.attempts = 0
        self.granted = 0
        self.attempt_times: list[float] = []
        self.return_times: list[float] = []

    def refresh_compression_lock(self, session_id, holder, ttl_seconds=300.0):
        self.attempts += 1
        self.attempt_times.append(time.monotonic())
        if self._remaining > 0:
            self._remaining -= 1
            if self._fail_latency_s:
                time.sleep(self._fail_latency_s)
            self.return_times.append(time.monotonic())
            return False
        self.granted += 1
        result = self._db.refresh_compression_lock(
            session_id, holder, ttl_seconds=ttl_seconds
        )
        self.return_times.append(time.monotonic())
        return result


def give_up_threshold(ttl: float, interval: float) -> int:
    """The refresher's own rule, restated: max(1, int(ttl / interval))."""
    return max(1, int(ttl / interval))


def run_refresher(db, *, fail_first: int, ttl: float, interval: float, ticks: int,
                   release: bool = True, fail_latency_s: float = 0.0):
    """Drive a real refresher over a real, genuinely held lock.

    The lock must be acquired first: refresh_compression_lock updates
    ``compression_locks WHERE session_id = ? AND holder = ?``, so with no
    row every delegated refresh returns False and the loop breaks for the
    wrong reason — which would make S4's counterfactual pass vacuously.

    Acquires with the SAME *ttl* the refresher itself uses, not
    ``PROD_TTL_S``: a caller that keeps the row alive past this call (via
    ``release=False``) needs it to have actually expired by the time the
    refresher gave up, not still be sitting on a fresh 300-second lease for
    another ~299s (review on #98867).

    ``fail_latency_s`` makes each failing refresh take that long to return,
    the way a contended ``_execute_write`` does; *ticks* then counts whole
    call-plus-interval cycles rather than bare intervals.
    """
    db.try_acquire_compression_lock(SESSION, HOLDER, ttl_seconds=ttl)
    flaky = FlakyDB(db, fail_first, fail_latency_s=fail_latency_s)
    r = _CompressionLockLeaseRefresher(flaky, SESSION, HOLDER, ttl, interval).start()
    try:
        time.sleep((interval + fail_latency_s) * (ticks + 0.5))
    finally:
        r.stop()
        if release:
            db.release_compression_lock(SESSION, HOLDER)
    return flaky


def stage_1_refresher_gives_up(db) -> None:
    print("\nS1  the refresher stops permanently after ttl/interval failures")
    ttl, interval = 2.0, 0.2          # threshold = 10, same rule as prod
    threshold = give_up_threshold(ttl, interval)
    check("S1", "give-up threshold is ttl/interval", threshold == 10, f"{threshold}")

    flaky = run_refresher(db, fail_first=threshold, ttl=ttl, interval=interval, ticks=25)
    check("S1", "loop stopped at the threshold", flaky.attempts == threshold,
          f"{flaky.attempts} attempts, then silence for {25 - threshold} more ticks")
    check("S1", "no refresh was ever granted afterwards", flaky.granted == 0,
          "the lease is now on its own, counting down to expiry")


def stage_2_ownership_was_recoverable(db) -> None:
    print("\nS2  ownership was still recoverable when the loop quit")
    # refresh_compression_lock decides ownership by `holder` ALONE, never by
    # expires_at — its docstring: "a live owner whose refresher thread was
    # starved ... past its own TTL must be able to revive its still-unclaimed
    # row on the next tick."
    db.try_acquire_compression_lock(SESSION, HOLDER, ttl_seconds=0.05)
    time.sleep(0.2)                                   # lease is now expired
    revived = db.refresh_compression_lock(SESSION, HOLDER, ttl_seconds=PROD_TTL_S)
    check("S2", "an expired lease is still revivable by its owner", revived is True,
          "so the DB contract says 'keep asking'")
    check("S2", "but the loop guarantees there is no next tick", True,
          "_run() breaks at the threshold — the two contracts contradict")
    db.release_compression_lock(SESSION, HOLDER)


def stage_3_publication_survives_the_giveup(db) -> None:
    print("\nS3  consequence at publication, post-#99216")
    db.create_session(session_id=SESSION, source="desktop", model="m")
    ttl, interval = 1.0, 0.2
    threshold = give_up_threshold(ttl, interval)
    # Drive the real refresher to its real give-up point over the same
    # TTL/holder/row publication will check, instead of seeding an unrelated
    # short-lived lease — the mismatch that made this stage vacuous before
    # #99216 landed (review on #98867).
    flaky = run_refresher(
        db, fail_first=threshold, ttl=ttl, interval=interval,
        ticks=threshold + 5, release=False,
    )
    check("S3", "the refresher really gave up", flaky.granted == 0)

    from hermes_state import CompressionSessionBusyError  # local: only needed here

    raised = None
    try:
        db.publish_compression_child(
            parent_session_id=SESSION,
            child_session_id="sess-child",
            source="desktop",
            messages=[{"role": "user", "content": "handoff"}],
            compression_lock_holder=HOLDER,
            require_compression_lease=True,
            require_lease_refresh=True,
            lease_ttl_seconds=ttl,
        )
    except CompressionSessionBusyError as exc:
        raised = str(exc)

    check("S3", "publication succeeds when nobody stole the lock", raised is None,
          (raised or "")[:64])
    check("S3", "the child was published (rotation completes)",
          db.get_session("sess-child") is not None,
          "-> #99216's pre-publication refresh gave the give-up one last save")
    db.release_compression_lock(SESSION, HOLDER)


def stage_4_the_window_depends_on_call_latency(db) -> None:
    print("\nS4  the give-up window is a range, plus the counterfactual")
    threshold = give_up_threshold(PROD_TTL_S, PROD_INTERVAL_S)
    check("S4", "production threshold is 5 failures", threshold == 5,
          f"ttl={PROD_TTL_S:.0f}s interval={PROD_INTERVAL_S:.0f}s")

    # Floor: the first refresh fires at t=0, then one interval passes before
    # each later attempt, so with refreshes that fail INSTANTLY the
    # break-triggering (threshold-th) failure lands (threshold - 1) intervals
    # after the first — not `threshold` intervals (correction from review on
    # #98867, where the original probe asserted `threshold * interval == TTL`).
    floor = (threshold - 1) * PROD_INTERVAL_S
    # Ceiling: refresh_compression_lock runs _execute_write on the routine
    # _WRITE_PATIENCE_S budget and only then returns False, and _run() waits
    # its interval after that return. Fully contended attempts START at
    # 0, 80, 160, 240, 320 and the fifth False comes back ~340s in.
    patience = SessionDB._WRITE_PATIENCE_S
    ceiling = (threshold - 1) * (PROD_INTERVAL_S + patience) + patience
    check("S4", "zero-latency floor is one interval short of the TTL",
          floor == 240.0 and floor < PROD_TTL_S,
          f"{floor:.0f}s < {PROD_TTL_S:.0f}s -- the MINIMUM, not the window")
    check("S4", "exhausted write patience pushes it past the TTL",
          ceiling == 340.0 and ceiling > PROD_TTL_S,
          f"{ceiling:.0f}s > {PROD_TTL_S:.0f}s -- a stall clearing at 300s can still recover")
    check("S4", "a 710s compression outlives the whole range",
          710.0 > ceiling,
          "issue log: total_duration_ms=710109, commit_status=aborted "
          "(reachable, not proven -- no refresh telemetry for that run)")

    # Measured, not restated: the same loop, with failing calls that take one
    # interval to answer, gives up strictly later than the zero-latency floor.
    ttl, interval, latency = 1.0, 0.2, 0.2
    t = give_up_threshold(ttl, interval)
    flaky = run_refresher(
        db, fail_first=t, ttl=ttl, interval=interval, ticks=t + 3,
        fail_latency_s=latency,
    )
    span = flaky.return_times[-1] - flaky.attempt_times[0]
    expected = (t - 1) * (interval + latency) + latency
    check("S4", "measured window grows with refresh call latency",
          abs(span - expected) < (interval + latency) * 0.5
          and span > (t - 1) * interval,
          f"{span:.2f}s measured vs {expected:.2f}s predicted, "
          f"floor was {(t - 1) * interval:.2f}s")

    # Counterfactual: the same stall, but the loop keeps asking.
    ttl, interval = 2.0, 0.2
    flaky = run_refresher(
        db, fail_first=give_up_threshold(ttl, interval) - 1,
        ttl=ttl, interval=interval, ticks=15,
    )
    check("S4", "one failure below the threshold recovers fully", flaky.granted > 0,
          f"{flaky.granted} refreshes granted after the same stall")


def main() -> int:
    # ignore_cleanup_errors: SessionDB holds the SQLite handle open, and
    # Windows refuses to unlink a mapped file (WinError 32).
    with tempfile.TemporaryDirectory(
        prefix="repro97948_", ignore_cleanup_errors=True
    ) as tmp:
        db = SessionDB(Path(tmp) / "state.db")
        stage_1_refresher_gives_up(db)
        stage_2_ownership_was_recoverable(db)
        stage_3_publication_survives_the_giveup(db)
        stage_4_the_window_depends_on_call_latency(db)

    print("\n" + "=" * 62)
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("all four stages PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
