"""agent/periodic_scheduler: shared timer thread + bounded workers for periodic timers."""

import threading
import time

from agent import periodic_scheduler
from agent.periodic_scheduler import PeriodicScheduler, schedule


def _wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_two_intervals_fire_proportionally_and_cancel_stops_one():
    sched = PeriodicScheduler()
    fast, slow = [], []
    h_fast = sched.schedule(lambda: fast.append(time.monotonic()), 0.01)
    h_slow = sched.schedule(lambda: slow.append(time.monotonic()), 0.05)

    assert _wait_until(lambda: len(slow) >= 3)
    assert len(fast) > len(slow)  # 5x interval ratio -> clearly more fast ticks
    # Both ran within this scheduler's bounded pool, not unconstrained new threads.
    before = threading.active_count()
    sched.schedule(lambda: None, 0.01).cancel()
    assert threading.active_count() == before
    assert sched._thread is not None and sched._thread.is_alive()

    h_fast.cancel()
    n_fast = len(fast)
    time.sleep(0.1)
    assert len(fast) == n_fast, "cancelled callback kept firing"
    assert len(slow) > 3, "sibling callback stopped when another was cancelled"
    h_slow.cancel()


def test_raising_callback_is_rescheduled_and_does_not_kill_sibling():
    sched = PeriodicScheduler()
    boom, ok = [], []

    def raises():
        boom.append(1)
        raise RuntimeError("bad callback")

    h1 = sched.schedule(raises, 0.01)
    h2 = sched.schedule(lambda: ok.append(1), 0.01)
    assert _wait_until(lambda: len(boom) >= 3 and len(ok) >= 3)
    h1.cancel()
    h2.cancel()


def test_returning_false_stops_callback_and_cancel_wait_joins_inflight():
    sched = PeriodicScheduler()
    calls = []
    sched.schedule(lambda: (calls.append(1), False)[1], 0.01)
    assert _wait_until(lambda: len(calls) == 1)
    time.sleep(0.05)
    assert calls == [1]

    entered = threading.Event()
    release = threading.Event()

    def blocking():
        entered.set()
        release.wait(2.0)

    h = sched.schedule(blocking, 0.01)
    assert entered.wait(2.0)
    threading.Timer(0.05, release.set).start()
    t0 = time.monotonic()
    h.cancel(wait=2.0)  # returns once the in-flight run finished
    assert release.is_set()
    assert time.monotonic() - t0 < 1.5


def test_module_level_schedule_uses_shared_default():
    hits = []
    h = schedule(lambda: hits.append(1), 0.01)
    assert _wait_until(lambda: hits)
    h.cancel()
    thread = periodic_scheduler._DEFAULT._thread
    assert thread is not None and thread.name == "hermes-periodic-scheduler"
    # Scheduling more timers on the shared default adds no OS threads beyond bounded pool.
    before = threading.active_count()
    handles = [schedule(lambda: None, 0.01) for _ in range(20)]
    assert threading.active_count() == before
    for handle in handles:
        handle.cancel()


def test_blocked_callback_does_not_stall_due_sibling(monkeypatch):
    """Regression test for Issue #102574: one blocked callback must not stall sibling due timers."""
    scheduler = PeriodicScheduler()
    monkeypatch.setattr(periodic_scheduler, "_DEFAULT", scheduler)
    blocker_entered = threading.Event()
    release_blocker = threading.Event()
    sibling_ran = threading.Event()

    def blocker():
        blocker_entered.set()
        release_blocker.wait(2.0)
        return False

    def sibling():
        sibling_ran.set()
        return False

    blocker_handle = schedule(blocker, 0.01)
    assert blocker_entered.wait(1.0)
    sibling_handle = schedule(sibling, 0.01)
    try:
        assert sibling_ran.wait(0.30), (
            "a blocked periodic callback stalled an unrelated due callback"
        )
    finally:
        release_blocker.set()
        blocker_handle.cancel(wait=1.0)
        sibling_handle.cancel(wait=1.0)


def test_worker_pool_growth_bounded_with_many_handles():
    """Verify thread growth remains strictly bounded under heavy handle load."""
    max_workers = 4
    sched = PeriodicScheduler(max_workers=max_workers)
    events = [threading.Event() for _ in range(20)]
    release_all = threading.Event()

    def make_task(ev):
        def task():
            ev.set()
            release_all.wait(0.5)
            return False
        return task

    handles = [sched.schedule(make_task(ev), 0.005) for ev in events]
    time.sleep(0.05)
    # Total worker threads created for this scheduler must not exceed max_workers
    assert len(sched._workers) <= max_workers
    release_all.set()
    for h in handles:
        h.cancel(wait=1.0)


def test_self_cancel_from_inside_callback_does_not_deadlock():
    """Verify a callback calling cancel(wait=...) on its own handle does not deadlock."""
    sched = PeriodicScheduler()
    handle_box = []
    completed = threading.Event()

    def self_cancelling():
        handle_box[0].cancel(wait=1.0)
        completed.set()
        return False

    h = sched.schedule(self_cancelling, 0.01)
    handle_box.append(h)
    assert completed.wait(2.0), "self-cancellation deadlocked the worker"
