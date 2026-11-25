"""Contract cases for ``persistence.core.health.StorageHealth``: the latch and the readiness probe answer without creating a store."""

from __future__ import annotations

from persistence import StorageHealth, StorageProbe, StorageState


def test_health_is_the_storage_health_protocol(backend, scope_a):
    assert isinstance(backend.open(scope_a).health, StorageHealth)


def test_probe_reports_a_missing_store_as_not_initialized_and_never_creates_it(backend, scope_a):
    store_a = backend.open(scope_a)
    assert store_a.health.probe() == StorageProbe(status="ok", detail="not initialized")
    assert store_a.health.probe() == StorageProbe(status="ok", detail="not initialized")
    assert store_a.health.storage_state() is StorageState.OK
    assert store_a.health.probe() == StorageProbe(status="ok", detail="not initialized")


def test_probe_reports_a_written_store_as_healthy(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.state_meta.set("probe-key", "value")
    assert store_a.health.probe() == StorageProbe(status="ok", detail=None)
    assert store_a.health.storage_state() is StorageState.OK
    assert store_b.health.probe() == StorageProbe(status="ok", detail="not initialized")
