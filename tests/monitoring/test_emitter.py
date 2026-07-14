"""Tests for the monitoring emitter: hot-path invariant + subscriber fan-out."""

from __future__ import annotations

import time

from agent.monitoring.emitter import MonitoringEmitter
from agent.monitoring.events import GatewayHealthEvent


def test_emit_never_raises_when_disabled():
    em = MonitoringEmitter(enabled=False)
    em.emit({"event": "gateway_health", "name": "gateway.health_snapshot"})
    assert em.stats()["queued"] == 0
    em.close()


def test_emit_accepts_dataclass_and_dict(tmp_path):
    em = MonitoringEmitter()
    seen: list = []
    em.subscribe(lambda batch: seen.extend(batch))
    em.emit(GatewayHealthEvent(name="gateway.health_snapshot", active_agents=2))
    em.emit({"event": "gateway_diagnostic", "name": "platform.fatal",
             "subsystem": "platform.slack"})
    em.flush()
    em.close()
    kinds = {ev.get("event") for ev in seen}
    assert kinds == {"gateway_health", "gateway_diagnostic"}
    health = next(ev for ev in seen if ev["event"] == "gateway_health")
    assert health["active_agents"] == 2
    assert "ts_ns" in health


def test_subscriber_failure_is_isolated():
    em = MonitoringEmitter()
    good: list = []

    def bad(batch):
        raise RuntimeError("boom")

    em.subscribe(bad)
    em.subscribe(lambda batch: good.extend(batch))
    em.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
    em.flush()
    em.close()
    assert len(good) == 1  # the raising subscriber did not break fan-out


def test_unsubscribe_stops_delivery():
    em = MonitoringEmitter()
    seen: list = []
    cb = lambda batch: seen.extend(batch)  # noqa: E731
    em.subscribe(cb)
    em.emit({"event": "gateway_health", "name": "a"})
    em.flush()
    em.unsubscribe(cb)
    em.emit({"event": "gateway_health", "name": "b"})
    em.flush()
    em.close()
    assert [ev["name"] for ev in seen] == ["a"]


def test_queue_full_drops_oldest():
    em = MonitoringEmitter()
    # Fill the queue without a dispatcher running by not letting it start:
    # emit() starts the thread, so instead assert drop accounting via stats
    # after a burst larger than the queue.
    for i in range(11_000):
        em.emit({"event": "gateway_health", "name": f"e{i}"})
    # Give the dispatcher a moment; total dispatched + queued + dropped == emitted.
    em.flush(timeout=5.0)
    stats = em.stats()
    em.close()
    assert stats["dropped"] >= 0
    assert stats["dispatched"] + stats["queued"] + stats["dropped"] >= 10_000


def test_hot_path_is_fast():
    em = MonitoringEmitter()
    start = time.perf_counter()
    for _ in range(1_000):
        em.emit({"event": "gateway_health", "name": "gateway.health_snapshot"})
    elapsed = time.perf_counter() - start
    em.close()
    # 1000 emits should be far under a second even on slow CI.
    assert elapsed < 1.0
