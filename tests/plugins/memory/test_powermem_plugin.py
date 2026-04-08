"""Unit tests for plugins/memory/powermem — MemoryProvider integration."""

from __future__ import annotations

import json
import sys
import types

import pytest

from plugins.memory.powermem import PowermemMemoryProvider, register


class FakePowermemMemory:
    """Minimal stand-in for powermem.Memory."""

    def __init__(self) -> None:
        self.search_calls: list = []
        self.add_calls: list = []
        self.get_all_calls: list = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"results": [{"memory": "fact one", "score": 0.91}]}

    def get_all(self, **kwargs):
        self.get_all_calls.append(kwargs)
        return {"results": [{"content": "from content field"}, {"memory": "from memory field"}]}

    def add(self, messages, **kwargs):
        self.add_calls.append({"messages": messages, **kwargs})
        return {"results": [{"event": "ADD", "memory": "ok"}]}


@pytest.fixture
def provider(monkeypatch) -> PowermemMemoryProvider:
    fake = FakePowermemMemory()

    monkeypatch.setattr(
        "plugins.memory.powermem._powermem_installed",
        lambda: True,
    )

    fake_pkg = types.ModuleType("powermem")

    def _fake_create_memory(**_kw):
        return fake

    fake_pkg.create_memory = _fake_create_memory
    monkeypatch.setitem(sys.modules, "powermem", fake_pkg)

    p = PowermemMemoryProvider()
    p.initialize(
        "sess-1",
        hermes_home="/tmp/hermes-test-home",
        user_id="user-99",
        agent_identity="coder",
    )
    assert p._memory is fake
    return p


def test_register_calls_ctx(monkeypatch):
    calls = []

    class Ctx:
        def register_memory_provider(self, p):
            calls.append(p)

    register(Ctx())
    assert len(calls) == 1
    assert calls[0].name == "powermem"


def test_is_available_without_package(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.powermem._powermem_installed",
        lambda: False,
    )
    p = PowermemMemoryProvider()
    assert p.is_available() is False


def test_search_scopes_user_and_agent(provider: PowermemMemoryProvider):
    out = json.loads(
        provider.handle_tool_call(
            "powermem_search", {"query": "prefs", "limit": 7}
        )
    )
    assert out["count"] == 1
    assert out["results"][0]["memory"] == "fact one"
    mem = provider._memory
    assert mem.search_calls[0]["user_id"] == "user-99"
    assert mem.search_calls[0]["agent_id"] == "coder"
    assert mem.search_calls[0]["limit"] == 7


def test_profile_merges_content_and_memory_fields(provider: PowermemMemoryProvider):
    out = json.loads(provider.handle_tool_call("powermem_profile", {"limit": 20}))
    assert out["count"] == 2
    assert "from content field" in out["result"]
    assert "from memory field" in out["result"]


def test_add_stores_verbatim_infer_false(provider: PowermemMemoryProvider):
    out = json.loads(
        provider.handle_tool_call("powermem_add", {"content": "likes tea"})
    )
    assert out.get("ok") is True
    mem = provider._memory
    assert len(mem.add_calls) == 1
    assert mem.add_calls[0]["infer"] is False
    assert mem.add_calls[0]["messages"] == "likes tea"


def test_sync_turn_intelligent_add(provider: PowermemMemoryProvider):
    provider.sync_turn("hi", "hello back", session_id="s1")
    provider._sync_thread.join(timeout=2)
    mem = provider._memory
    assert len(mem.add_calls) == 1
    call = mem.add_calls[0]
    assert call["infer"] is True
    assert call["user_id"] == "user-99"
    assert call["agent_id"] == "coder"
    msgs = call["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["role"] == "assistant"


def test_tool_error_when_memory_uninitialized(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.powermem._powermem_installed",
        lambda: True,
    )
    p = PowermemMemoryProvider()
    p._memory = None
    raw = json.loads(p.handle_tool_call("powermem_search", {"query": "x"}))
    assert "error" in raw
    assert "not initialized" in raw["error"].lower()
