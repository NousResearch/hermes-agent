"""Tests for the monitoring emitter: hot-path invariant + subscriber fan-out."""

from __future__ import annotations


from agent.monitoring.emitter import MonitoringEmitter


def test_emit_never_raises_when_disabled():
    em = MonitoringEmitter(enabled=False)
    em.emit({"event": "gateway_health", "name": "gateway.health_snapshot"})
    assert em.stats()["queued"] == 0
    em.close()


def test_process_singleton_stays_dormant_until_subscribed():
    from agent.monitoring import emitter

    emitter.reset_emitter_for_tests()
    try:
        emitter.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
        singleton = emitter.get_emitter()
        assert singleton.stats()["queued"] == 0
        assert singleton._started is False

        subscriber = lambda _batch: None  # noqa: E731
        singleton.subscribe(subscriber)
        emitter.emit({"event": "gateway_health", "name": "gateway.lifecycle"})
        assert singleton._started is True
        singleton.unsubscribe(subscriber)
    finally:
        emitter.reset_emitter_for_tests()








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




