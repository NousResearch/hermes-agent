from benchmarks.backends.holographic_adapter import HolographicBenchmarkAdapter


def test_holographic_adapter_store_and_recall():
    backend = HolographicBenchmarkAdapter()
    backend.reset()
    backend.store("Production region is us-east-1")
    results = backend.recall("What production region are we using?", top_k=3)
    assert any("us-east-1" in r for r in results)


def test_holographic_adapter_stats_and_reset():
    backend = HolographicBenchmarkAdapter()
    backend.reset()
    backend.store("Primary database is PostgreSQL 16")
    stats = backend.get_stats()
    assert stats["fact_count"] >= 1
    backend.reset()
    stats2 = backend.get_stats()
    assert stats2["fact_count"] == 0
