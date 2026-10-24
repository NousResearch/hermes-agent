"""``/api/status`` gateway health fallback honours its route deadline (#138388).

The fallback used to run inside ``with ThreadPoolExecutor(...)``; ``__exit__`` joins the
worker, so a slow gateway held the route past ``_GATEWAY_HEALTH_ROUTE_TIMEOUT`` anyway.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_cli.web_routers import status


@pytest.fixture
def blocked_probe(monkeypatch):
    release = threading.Event()
    calls: list[int] = []

    def _slow_probe():
        calls.append(1)
        release.wait(10.0)
        return True, {"ok": True}

    monkeypatch.setattr(status, "_probe_gateway_health", _slow_probe)
    monkeypatch.setattr(status, "_GATEWAY_HEALTH_ROUTE_TIMEOUT", 0.2)
    monkeypatch.setattr(status, "_health_probe_inflight", None, raising=False)
    try:
        yield calls
    finally:
        release.set()
        inflight = status._health_probe_inflight
        if inflight is not None:
            inflight.result(timeout=5.0)


def test_fallback_returns_at_the_deadline_without_joining_the_worker(blocked_probe):
    start = time.monotonic()
    result = status._bounded_health_probe()
    elapsed = time.monotonic() - start

    assert result == (False, None)
    assert elapsed < 2.0, f"route waited {elapsed:.2f}s for a probe it had already given up on"


def test_polls_during_a_hung_probe_share_one_worker(blocked_probe):
    assert status._bounded_health_probe() == (False, None)
    assert status._bounded_health_probe() == (False, None)

    assert len(blocked_probe) == 1, "each poll must not strand another stuck probe thread"
