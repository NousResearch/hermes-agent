import sys
import pytest
from unittest.mock import MagicMock, patch
from benchmarks.backends.mem0_adapter import Mem0BenchmarkAdapter


def test_mem0_adapter_requires_api_key(monkeypatch):
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MEM0_API_KEY"):
        Mem0BenchmarkAdapter()


def test_mem0_adapter_reset_isolates_user_id(monkeypatch):
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    with patch.dict(sys.modules, {"mem0": MagicMock()}):
        with patch("mem0.MemoryClient", return_value=MagicMock()):
            adapter = Mem0BenchmarkAdapter()
            uid_before = adapter._user_id
            adapter.reset()
            uid_after = adapter._user_id
            assert uid_before != uid_after


def test_mem0_adapter_get_stats(monkeypatch):
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    with patch.dict(sys.modules, {"mem0": MagicMock()}):
        with patch("mem0.MemoryClient", return_value=MagicMock()):
            adapter = Mem0BenchmarkAdapter()
            stats = adapter.get_stats()
            assert stats["backend"] == "mem0"
            assert "user_id" in stats
