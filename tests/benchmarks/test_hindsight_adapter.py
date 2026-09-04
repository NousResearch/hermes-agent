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


def test_reset_rotates_bank_id(monkeypatch):
    """reset() must rotate to a fresh bank so scenarios are isolated."""
    monkeypatch.setenv("HINDSIGHT_BASE_URL", "http://localhost:8888")
    adapter = HindsightBenchmarkAdapter()
    original_bank = adapter._bank_id

    adapter.reset()

    assert adapter._bank_id != original_bank
    assert adapter._bank_id.startswith("benchmark-bank-")


def test_reset_clears_cached_worker(monkeypatch):
    """reset() must drop the cached worker so the next call binds to the new bank."""
    monkeypatch.setenv("HINDSIGHT_BASE_URL", "http://localhost:8888")
    adapter = HindsightBenchmarkAdapter()
    # Simulate a previously-created worker
    adapter._worker = object()
    assert adapter._worker is not None

    adapter.reset()

    assert adapter._worker is None


def test_repeated_reset_produces_unique_banks(monkeypatch):
    """Each reset() must produce a different bank id."""
    monkeypatch.setenv("HINDSIGHT_BASE_URL", "http://localhost:8888")
    adapter = HindsightBenchmarkAdapter()
    banks = {adapter._bank_id}
    for _ in range(5):
        adapter.reset()
        banks.add(adapter._bank_id)
    assert len(banks) == 6
