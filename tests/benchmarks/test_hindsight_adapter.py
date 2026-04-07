import pytest
from benchmarks.backends.hindsight_adapter import HindsightBenchmarkAdapter


def test_hindsight_adapter_requires_base_url(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="HINDSIGHT_BASE_URL"):
        HindsightBenchmarkAdapter()


def test_hindsight_adapter_accepts_base_url(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BASE_URL", "http://localhost:8888")
    adapter = HindsightBenchmarkAdapter()
    stats = adapter.get_stats()
    assert stats["backend"] == "hindsight"
    assert "base_url" in stats


def test_hindsight_adapter_get_stats(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BASE_URL", "http://localhost:8888")
    adapter = HindsightBenchmarkAdapter()
    stats = adapter.get_stats()
    assert stats["base_url"] == "http://localhost:8888"
