"""The /api/status gateway health fallback must return at its declared route deadline.

`_bounded_health_probe` advertises `_GATEWAY_HEALTH_ROUTE_TIMEOUT`, but it used to open a
`ThreadPoolExecutor` per request and return from *inside* the `with` block, so leaving the
context called `Executor.__exit__ -> shutdown(wait=True)` and joined a callee that had already
outlived the budget. The worker is now a process-wide single-flight pool: the route returns at
the deadline, a running call keeps its slot until it really exits, and later polls queue behind
it instead of accumulating one worker per timed-out poll (issue #138388).
"""

import threading
import time

import pytest

from hermes_cli.web_routers import status as status_module

# Small route budget so the suite stays fast; the budget is read per call.
BUDGET = 0.1
SLACK = 0.4  # generous room for a loaded CI box, far below the callee's own delay
CALLEE_DELAY = 2.0  # how long a "stuck" callee blocks before it releases itself


@pytest.fixture(autouse=True)
def _drop_probe_worker():
    yield
    reset = getattr(status_module, "_reset_health_probe_worker", None)
    if reset is not None:
        reset()


def _blocking_probe(released: threading.Event, result=("live", {"ok": True})):
    def probe():
        released.wait(CALLEE_DELAY)
        return result

    return probe


def test_healthy_probe_returns_its_result(monkeypatch):
    monkeypatch.setattr(status_module, "_probe_gateway_health", lambda: ("ok", {"up": True}))
    monkeypatch.setattr(status_module, "_GATEWAY_HEALTH_ROUTE_TIMEOUT", BUDGET)
    assert status_module._bounded_health_probe() == ("ok", {"up": True})


def test_returns_at_the_deadline_while_the_callee_still_runs(monkeypatch):
    released = threading.Event()
    monkeypatch.setattr(status_module, "_probe_gateway_health", _blocking_probe(released))
    monkeypatch.setattr(status_module, "_GATEWAY_HEALTH_ROUTE_TIMEOUT", BUDGET)
    try:
        started = time.monotonic()
        assert status_module._bounded_health_probe() == (False, None)
        elapsed = time.monotonic() - started
        assert elapsed < BUDGET + SLACK, f"route waited {elapsed:.3f}s past its {BUDGET}s budget"
        assert not released.is_set(), "callee finished early: the deadline path was not exercised"

        # A second poll while the first call is still live returns at its own deadline too.
        started = time.monotonic()
        assert status_module._bounded_health_probe() == (False, None)
        assert time.monotonic() - started < BUDGET + SLACK
    finally:
        released.set()


def test_one_worker_is_reused_and_readmitted_after_the_callee_exits(monkeypatch):
    released = threading.Event()
    monkeypatch.setattr(status_module, "_probe_gateway_health", _blocking_probe(released))
    monkeypatch.setattr(status_module, "_GATEWAY_HEALTH_ROUTE_TIMEOUT", BUDGET)
    try:
        assert status_module._bounded_health_probe() == (False, None)
        worker = status_module._health_probe_worker()

        released.set()
        deadline = time.monotonic() + 5.0
        result = None
        while time.monotonic() < deadline:
            result = status_module._bounded_health_probe()
            if result == ("live", {"ok": True}):
                break
            time.sleep(0.02)

        assert result == ("live", {"ok": True}), "the slot was not readmitted after the callee exited"
        assert status_module._health_probe_worker() is worker, "a new worker was created per poll"
    finally:
        released.set()


def test_reset_drops_the_worker_without_waiting_on_a_live_callee(monkeypatch):
    released = threading.Event()
    monkeypatch.setattr(status_module, "_probe_gateway_health", _blocking_probe(released))
    monkeypatch.setattr(status_module, "_GATEWAY_HEALTH_ROUTE_TIMEOUT", BUDGET)
    try:
        assert status_module._bounded_health_probe() == (False, None)
        worker = status_module._health_probe_worker()

        started = time.monotonic()
        status_module._reset_health_probe_worker()
        assert time.monotonic() - started < SLACK, "shutdown joined a callee that was still running"
        assert status_module._health_probe_worker() is not worker
    finally:
        released.set()


def test_probe_failure_still_reports_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("gateway probe exploded")

    monkeypatch.setattr(status_module, "_probe_gateway_health", boom)
    monkeypatch.setattr(status_module, "_GATEWAY_HEALTH_ROUTE_TIMEOUT", BUDGET)
    assert status_module._bounded_health_probe() == (False, None)
