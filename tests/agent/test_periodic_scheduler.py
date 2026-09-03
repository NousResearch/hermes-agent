"""agent/periodic_scheduler: one timer thread dispatches isolated callbacks."""

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
    # Scheduling another timer creates no persistent per-handle thread.
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


def test_blocking_callback_does_not_starve_sibling():
    sched = PeriodicScheduler()
    entered = threading.Event()
    release = threading.Event()
    sibling_fired = threading.Event()

    def blocking():
        entered.set()
        release.wait(2.0)

    blocked = sched.schedule(blocking, 0.01)
    sibling = sched.schedule(sibling_fired.set, 0.02)
    try:
        assert entered.wait(2.0)
        assert sibling_fired.wait(0.5), (
            "one blocked periodic callback starved an unrelated sibling"
        )
    finally:
        release.set()
        blocked.cancel(wait=2.0)
        sibling.cancel(wait=2.0)


def test_slow_callback_never_overlaps_itself():
    sched = PeriodicScheduler()
    lock = threading.Lock()
    second_run_finished = threading.Event()
    running = 0
    peak_running = 0
    completed = 0

    def slow():
        nonlocal running, peak_running, completed
        with lock:
            running += 1
            peak_running = max(peak_running, running)
        time.sleep(0.05)
        with lock:
            running -= 1
            completed += 1
            if completed >= 2:
                second_run_finished.set()

    handle = sched.schedule(slow, 0.01)
    try:
        assert second_run_finished.wait(2.0)
        assert peak_running == 1
    finally:
        handle.cancel(wait=2.0)


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
    # Scheduling more timers on the shared default adds no OS threads.
    before = threading.active_count()
    handles = [schedule(lambda: None, 0.01) for _ in range(20)]
    assert threading.active_count() == before
    for handle in handles:
        handle.cancel()
