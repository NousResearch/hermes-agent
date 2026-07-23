"""Tests for plugins/plugin_utils.py — thread-safe lazy singleton helpers.

These exercise the actual concurrency guarantee with real threads (not mocks):
a barrier releases N threads simultaneously into the accessor, and we assert
the factory ran exactly once.
"""

import threading

import pytest

from plugins.plugin_utils import SingletonSlot, lazy_singleton


# --- lazy_singleton -------------------------------------------------------


def test_lazy_singleton_builds_once_and_returns_same_instance():
    calls = []

    @lazy_singleton
    def get():
        calls.append(1)
        return object()

    a = get()
    b = get()
    assert a is b
    assert len(calls) == 1


def test_lazy_singleton_reset_rebuilds():
    counter = {"n": 0}

    @lazy_singleton
    def get():
        counter["n"] += 1
        return counter["n"]

    assert get() == 1
    assert get() == 1
    get.reset()
    assert get() == 2


def test_lazy_singleton_factory_exception_not_cached():
    state = {"fail": True}

    @lazy_singleton
    def get():
        if state["fail"]:
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError):
        get()
    # First call raised → nothing cached → retry succeeds once we stop failing.
    state["fail"] = False
    assert get() == "ok"


def test_lazy_singleton_concurrent_first_call_builds_once():
    build_count = {"n": 0}
    build_lock = threading.Lock()
    barrier = threading.Barrier(16)
    results = []
    results_lock = threading.Lock()

    @lazy_singleton
    def get():
        # Count builds under a lock so the assertion is exact even if the
        # double-checked lock had a bug and let two through.
        with build_lock:
            build_count["n"] += 1
        # Simulate an expensive build so threads genuinely overlap.
        import time
        time.sleep(0.01)
        return object()

    def worker():
        barrier.wait()  # release all threads at once
        obj = get()
        with results_lock:
            results.append(obj)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert build_count["n"] == 1, "factory must run exactly once under race"
    assert len(results) == 16
    assert all(r is results[0] for r in results), "all callers share one instance"


# --- SingletonSlot --------------------------------------------------------


def test_slot_caches_first_value():
    slot: SingletonSlot = SingletonSlot()
    assert slot.peek() is None
    v1 = slot.get(lambda: "first")
    assert slot.peek() == "first"
    # Subsequent factory is ignored — first value wins.
    v2 = slot.get(lambda: "second")
    assert v1 == v2 == "first"


def test_slot_reset():
    slot: SingletonSlot = SingletonSlot()
    slot.get(lambda: "a")
    slot.reset()
    assert slot.peek() is None
    assert slot.get(lambda: "b") == "b"


def test_slot_factory_exception_not_cached():
    slot: SingletonSlot = SingletonSlot()

    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        slot.get(boom)
    assert slot.peek() is None
    assert slot.get(lambda: "recovered") == "recovered"


def test_slot_concurrent_first_call_builds_once():
    build_count = {"n": 0}
    build_lock = threading.Lock()
    barrier = threading.Barrier(16)
    slot: SingletonSlot = SingletonSlot()
    results = []
    results_lock = threading.Lock()

    def factory():
        with build_lock:
            build_count["n"] += 1
        import time
        time.sleep(0.01)
        return object()

    def worker():
        barrier.wait()
        obj = slot.get(factory)
        with results_lock:
            results.append(obj)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert build_count["n"] == 1
    assert len(results) == 16
    assert all(r is results[0] for r in results)
