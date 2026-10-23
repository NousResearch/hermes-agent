import pytest
from benchmarks.backends.retaindb_adapter import RetainDBBenchmarkAdapter


def test_retaindb_adapter_requires_api_key(monkeypatch):
    monkeypatch.delenv("RETAINDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RETAINDB_API_KEY"):
        RetainDBBenchmarkAdapter()


def test_retaindb_adapter_reset_rotates_user_id(monkeypatch):
    monkeypatch.setenv("RETAINDB_API_KEY", "test-key")
    adapter = RetainDBBenchmarkAdapter()
    uid_before = adapter._user_id
    adapter.reset()
    uid_after = adapter._user_id
    assert uid_before != uid_after
    assert uid_after.startswith("bench-")


def test_retaindb_adapter_get_stats(monkeypatch):
    monkeypatch.setenv("RETAINDB_API_KEY", "test-key")
    adapter = RetainDBBenchmarkAdapter()
    stats = adapter.get_stats()
    assert stats["backend"] == "retaindb"
    assert "user_id" in stats
    assert "project" in stats
