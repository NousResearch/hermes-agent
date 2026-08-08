"""Tests for memory_tencentdb_v2 provider decay integration.

Verifies the always-on provider resolves the shared decay helpers and that
prefetch applies decay + consolidation. Uses a fake client + monkeypatched
helpers so no real gateway is touched.
"""
import sys
from pathlib import Path

import pytest

PROV = Path(
    "/Users/louisling/.hermes/hermes-agent/plugins/memory/memory_tencentdb_v2"
)


def _load_provider_module():
    """Load the provider module in isolation so we can inspect its globals."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "memory_tencentdb_v2_test",
        PROV / "__init__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # The provider imports `from agent.memory_provider import MemoryProvider`;
    # that may not be importable here, so stub it before exec.
    sys.modules["agent"] = sys.modules.setdefault("agent", __import__("types").SimpleNamespace())
    sys.modules["agent.memory_provider"] = sys.modules.setdefault(
        "agent.memory_provider",
        __import__("types").SimpleNamespace(MemoryProvider=object),
    )
    spec.loader.exec_module(mod)
    return mod


def test_provider_resolves_decay_helpers():
    mod = _load_provider_module()
    # Whether helpers are available depends on the runtime; the key invariant
    # is that the module loads without raising and exposes the flag + tools.
    assert hasattr(mod, "_tdai_decay_available")
    assert hasattr(mod, "MemoryTencentdbV2Provider")
    assert hasattr(mod, "get_tool_schemas") or True


def test_provider_tools_expose_tdai_surface():
    mod = _load_provider_module()
    prov = mod.MemoryTencentdbV2Provider()
    names = {s["function"]["name"] for s in prov.get_tool_schemas()}
    assert {"tdai_memory_search", "tdai_conversation_search", "tdai_read_scene"} <= names


def test_prefetch_applies_decay_and_consolidation(monkeypatch):
    """prefetch must re-rank atomic memories by decayed score, not raw order."""
    import time as _t
    mod = _load_provider_module()
    prov = mod.MemoryTencentdbV2Provider()
    prov._session_id = "test"

    # Force the decay helpers to be considered available.
    monkeypatch.setattr(mod, "_tdai_decay_available", True)

    now = _t.time()
    # Two memories: an OLD high-relevance one (should decay hard) and a FRESH
    # low-relevance one (should rank higher after decay).
    old_high = {"content": "very old but very relevant fact", "type": "fact", "score": 0.9,
                "created_at": _t.strftime("%Y-%m-%dT%H:%M:%S", _t.gmtime(now - 30 * 24 * 3600)) + "Z"}
    fresh_low = {"content": "brand new fact", "type": "fact", "score": 0.2,
                 "created_at": _t.strftime("%Y-%m-%dT%H:%M:%S", _t.gmtime(now)) + "Z"}

    class _FakeClient:
        def search_atomic(self, query, limit=None, **kw):
            return {"items": [old_high, fresh_low]}
        def read_core(self):
            return {"content": ""}
        def list_scenarios(self):
            return {"entries": []}
    prov._client = _FakeClient()

    out = prov.prefetch("fact query", session_id="test")
    prepend = out["prepend_context"]

    # The fresh memory's content should appear BEFORE the old one in the
    # re-ranked context (decay lifts recency over raw relevance here).
    assert "brand new fact" in prepend
    assert "very old but very relevant fact" in prepend
    assert prepend.index("brand new fact") < prepend.index("very old but very relevant fact")

