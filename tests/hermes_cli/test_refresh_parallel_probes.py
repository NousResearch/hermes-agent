"""Regression tests for bounded-parallel endpoint probing on the model-picker refresh path.

Issue #123793: with many user-configured ``providers:`` entries, ``model.options``
``refresh=true`` blocked the picker for minutes because section 3/4 endpoint probes
ran serially (31 providers x up to 5 s per live /models probe ~= 155 s worst case).
The refresh path now probes in a bounded thread pool with a wall-clock deadline;
groups that miss the deadline degrade to their cached/curated model list instead of
stalling the whole wave.
"""
from __future__ import annotations

import threading
import time

import pytest

from hermes_cli.model_switch_providers import _run_endpoint_probes_parallel


def test_serial_fallback_single_probe():
    """n <= 1 never spawns a pool: the single probe fn runs inline."""
    results = _run_endpoint_probes_parallel([lambda: ("m", False, True)], deadline=1.0)
    assert results == [("m", False, True)]


def test_serial_fallback_empty():
    assert _run_endpoint_probes_parallel([], deadline=1.0) == []


def test_parallel_results_ordered():
    """Result i corresponds to probe fn i regardless of completion order."""
    import concurrent.futures

    def probe(i):
        def _fn():
            time.sleep(0.01 * (len([i]) and (5 - i % 5)))  # later-index fns finish first
            return ("model", False, True)[0:1] * 0 + ("m%d" % i, False, True)
        return _fn

    fns = [probe(i) for i in range(10)]
    results = _run_endpoint_probes_parallel(fns, deadline=10.0)
    assert results == [("m%d" % i, False, True) for i in range(10)]


def test_parallel_runs_on_multiple_threads():
    """The pool genuinely overlaps probes (the whole point of the fix)."""
    observed = []
    lock = threading.Lock()

    def probe(i):
        def _fn():
            with lock:
                observed.append(threading.get_ident())
            time.sleep(0.05)
            return (f"m{i}", False, True)
        return _fn

    fns = [probe(i) for i in range(8)]
    results = _run_endpoint_probes_parallel(fns, deadline=10.0)
    assert results == [(f"m{i}", False, True) for i in range(8)]
    assert len(set(observed)) > 1, "probes must run concurrently, not serially"


def test_raising_probe_degrades():
    """A probe that raises degrades that slot to (None, False, False); others still run."""

    def good():
        return ("m", False, True)

    def bad():
        raise RuntimeError("endpoint exploded")

    results = _run_endpoint_probes_parallel([good, bad, good], deadline=10.0)
    assert results[0] == ("m", False, True)
    assert results[1] == (None, False, False)
    assert results[2] == ("m", False, True)


def test_deadline_degrades_slow_probes():
    """Probes still running when the deadline expires degrade; fast ones keep results.

    Also asserts the deadline is a real latency bound: the call itself must return
    promptly even when a probe is still in flight (shutdown(wait=False) path)."""

    def fast():
        time.sleep(0.01)
        return ("m-fast", False, True)

    def slow():
        time.sleep(3.0)
        return ("m-slow", False, True)

    t0 = time.monotonic()
    results = _run_endpoint_probes_parallel([fast, slow], deadline=0.2)
    elapsed = time.monotonic() - t0
    assert results[0] == ("m-fast", False, True)
    assert results[1] == (None, False, False), "a probe past the deadline must degrade"
    # The caller's latency must track the deadline, not the slow probe's full
    # duration. 5x the deadline is a generous margin for CI jitter.
    assert elapsed < 0.2 * 5, f"deadline was not honored: returned in {elapsed:.2f}s"


def test_deadline_hard_bound_many_slow_probes():
    """Many probes that all overshoot the deadline: the call still returns on time.

    Regression guard for the pre-fix behavior where ``with ThreadPoolExecutor``'s
    ``__exit__`` joined every worker, so a 3 s probe held the caller for 3 s even
    with ``deadline=0.2``."""

    def slow():
        time.sleep(3.0)
        return ("m-slow", False, True)

    fns = [slow for _ in range(8)]
    t0 = time.monotonic()
    results = _run_endpoint_probes_parallel(fns, deadline=0.2)
    elapsed = time.monotonic() - t0
    assert results == [(None, False, False)] * 8
    assert elapsed < 1.5, f"deadline must bound the caller's latency, saw {elapsed:.2f}s"


def test_single_probe_skips_pool_by_design():
    """n == 1 runs inline: the deadline does not cap it (documented, intentional).

    A single probe cannot be overlapped, so the serial fallback executes it
    directly and its full duration is the caller's latency. Recorded here so the
    bypass is explicit rather than accidental."""

    def slow():
        time.sleep(0.2)
        return ("m-slow", False, True)

    t0 = time.monotonic()
    results = _run_endpoint_probes_parallel([slow], deadline=0.05)
    elapsed = time.monotonic() - t0
    assert results == [("m-slow", False, True)]
    assert elapsed >= 0.2, "n==1 path runs the probe inline, deadline does not apply"


def test_worker_cap_respected():
    """No more than _PROBE_POOL_MAX_WORKERS probes may be executing at the same time."""
    from hermes_cli import model_switch_providers as msp

    in_flight = 0
    peak = 0
    lock = threading.Lock()
    entered = threading.Event()
    workers = msp._PROBE_POOL_MAX_WORKERS
    released = threading.Event()

    def probe(i):
        def _fn():
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            if in_flight == workers:
                # All pool workers are now inside a probe: hold them until we record the peak,
                # so a second batch cannot inflate the count.
                entered.set()
                released.wait(timeout=10.0)
            time.sleep(0.02)
            with lock:
                in_flight -= 1
            return ("m", False, True)
        return _fn

    fns = [probe(i) for i in range(workers * 2)]
    try:
        _run_endpoint_probes_parallel(fns, deadline=30.0)
    finally:
        released.set()
    assert peak == workers, f"expected exactly {workers} overlapping probes, saw {peak}"
