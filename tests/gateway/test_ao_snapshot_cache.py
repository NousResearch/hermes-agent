from gateway.ao_snapshot_cache import AOSnapshotCache


def test_ao_snapshot_cache_reuses_memory_snapshot(monkeypatch):
    monkeypatch.setenv("ORYN_AO_SNAPSHOT_CACHE", "memory")
    cache = AOSnapshotCache(ttl_seconds=60)
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        return {
            "sessions": [{"id": "ao-1", "project_id": "OrynWorkspace", "status": "running"}],
            "health_by_id": {"ao-1": {"runtime_health": "alive"}},
        }

    first = cache.get_or_load(project_id="OrynWorkspace", load=load)
    second = cache.get_or_load(project_id="OrynWorkspace", load=load)

    assert calls == 1
    assert first.cache_status == "miss"
    assert second.cache_status == "memory_hit"
    assert second.sessions == first.sessions
    assert second.health_by_id["ao-1"]["runtime_health"] == "alive"


def test_ao_snapshot_cache_invalidate_forces_reload(monkeypatch):
    monkeypatch.setenv("ORYN_AO_SNAPSHOT_CACHE", "memory")
    cache = AOSnapshotCache(ttl_seconds=60)
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        return {
            "sessions": [{"id": f"ao-{calls}", "status": "running"}],
            "health_by_id": {},
        }

    assert cache.get_or_load(project_id=None, load=load).sessions[0]["id"] == "ao-1"
    cache.invalidate()
    assert cache.get_or_load(project_id=None, load=load).sessions[0]["id"] == "ao-2"
