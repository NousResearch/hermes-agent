"""mem0 rerank flag must use shared truthy aliases (and reject 'off')."""

from __future__ import annotations

import json

import pytest

from plugins.memory.mem0 import Mem0MemoryProvider


class _CaptureBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query, *, filters, top_k=10, rerank=False):
        self.calls.append(
            {"query": query, "filters": filters, "top_k": top_k, "rerank": rerank}
        )
        return [{"id": "m1", "memory": "ok", "score": 1.0}]


def _provider_with_backend(backend, *, rerank_default=False) -> Mem0MemoryProvider:
    provider = Mem0MemoryProvider()
    provider._user_id = "u1"
    provider._agent_id = "hermes"
    provider._backend = backend
    provider._rerank_default = rerank_default
    return provider


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE", " On "])
def test_mem0_search_truthy_rerank_aliases(raw):
    backend = _CaptureBackend()
    provider = _provider_with_backend(backend)
    out = json.loads(
        provider.handle_tool_call("mem0_search", {"query": "q", "rerank": raw})
    )
    assert out["count"] == 1
    assert backend.calls[-1]["rerank"] is True


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", "OFF", ""])
def test_mem0_search_falsy_rerank_aliases(raw):
    backend = _CaptureBackend()
    provider = _provider_with_backend(backend, rerank_default=True)
    provider.handle_tool_call("mem0_search", {"query": "q", "rerank": raw})
    assert backend.calls[-1]["rerank"] is False


def test_mem0_search_garbage_string_does_not_enable_rerank():
    """Regression: previously anything not in {false,0,no} enabled rerank."""
    backend = _CaptureBackend()
    provider = _provider_with_backend(backend)
    provider.handle_tool_call("mem0_search", {"query": "q", "rerank": "potato"})
    assert backend.calls[-1]["rerank"] is False


@pytest.mark.parametrize("raw", ["on", "1", "yes", "true"])
def test_mem0_config_rerank_truthy_default(monkeypatch, tmp_path, raw):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    (tmp_path / "mem0.json").write_text(json.dumps({"rerank": raw}))
    provider = Mem0MemoryProvider()
    provider._create_backend = lambda: None  # type: ignore[method-assign]
    provider.initialize("test-session")
    assert provider._rerank_default is True


def test_mem0_config_rerank_off_stays_false(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    (tmp_path / "mem0.json").write_text(json.dumps({"rerank": "off"}))
    provider = Mem0MemoryProvider()
    provider._create_backend = lambda: None  # type: ignore[method-assign]
    provider.initialize("test-session")
    assert provider._rerank_default is False
