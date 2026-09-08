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


def test_blocked_callback_does_not_stall_due_sibling(monkeypatch):
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


def test_slow_callback_never_overlaps_itself():
    sched = PeriodicScheduler()
    lock = threading.Lock()
    two_runs = threading.Event()
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
                two_runs.set()

    handle = sched.schedule(slow, 0.01)
    try:
        assert two_runs.wait(2.0)
        assert peak_running == 1, "a handle ran concurrently with itself"
    finally:
        handle.cancel(wait=2.0)


def test_worker_start_failure_keeps_timer(monkeypatch):
    sched = PeriodicScheduler()
    fired: list = []
    real_thread = threading.Thread
    attempts = {"n": 0}

    def flaky(*args, **kwargs):
        name = kwargs.get("name", "")
        if name.startswith(periodic_scheduler._CALLBACK_THREAD_PREFIX) and attempts["n"] == 0:
            attempts["n"] += 1

            class Boom:
                def start(self):
                    raise RuntimeError("no threads")

            return Boom()
        attempts["n"] += 1
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(periodic_scheduler.threading, "Thread", flaky)
    handle = sched.schedule(lambda: fired.append(1), 0.01)
    try:
        assert _wait_until(lambda: bool(fired), timeout=3.0), (
            "worker-start failure silently retired the timer"
        )
        assert not handle.cancelled
    finally:
        handle.cancel()
