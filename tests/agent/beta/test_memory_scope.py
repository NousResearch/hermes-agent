from types import SimpleNamespace

import pytest

from agent.beta.memory_scope import MemoryScope, ScopedMemory, classify_memory


class FakeMemoryManager:
    def __init__(self, hindsight=True):
        self.hindsight = hindsight
        self.calls = []

    def has_tool(self, name):
        return self.hindsight and name in {"hindsight_retain", "hindsight_recall"}

    def handle_tool_call(self, name, args):
        self.calls.append((name, args))
        return {"result": "ok"}

    def on_memory_write(self, action, target, content, metadata=None):
        self.calls.append((action, target, content, metadata))

    def prefetch_all(self, query):
        self.calls.append(("prefetch", query))
        return "memory"


def test_chief_preference_routes_to_strategic_memory():
    route = classify_memory("O Chefe prefere relatórios curtos")
    assert route.scope == MemoryScope.STRATEGIC
    assert route.tag == "beta:strategic"


def test_postgresql_configuration_routes_to_dba_memory():
    route = classify_memory("PostgreSQL uses shared_buffers=4GB")
    assert route.scope == MemoryScope.TECHNICAL
    assert route.specialist_id == "dba"


def test_specialist_cannot_write_strategic_or_other_scope():
    memory = ScopedMemory(FakeMemoryManager())
    with pytest.raises(PermissionError, match="strategic"):
        memory.retain("O Chefe prefere respostas curtas", actor="dba")
    with pytest.raises(PermissionError, match="own"):
        memory.retain("Firewall policy", actor="dba", specialist_id="security")


def test_hindsight_retain_and_recall_use_strict_scope_tags():
    manager = FakeMemoryManager()
    memory = ScopedMemory(manager)
    memory.retain("PostgreSQL uses shared_buffers=4GB", actor="dba", specialist_id="dba")
    memory.recall("shared_buffers", actor="dba")
    assert manager.calls[0][0] == "hindsight_retain"
    assert "beta:specialist:dba" in manager.calls[0][1]["tags"]
    assert manager.calls[1] == (
        "hindsight_recall",
        {"query": "shared_buffers", "tags": ["beta:specialist:dba"], "tags_match": "all_strict"},
    )


def test_hindsight_provider_accepts_per_call_recall_tags():
    from plugins.memory.hindsight import HindsightMemoryProvider, RECALL_SCHEMA

    provider = HindsightMemoryProvider()
    captured = {}

    class Client:
        def arecall(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(results=[])

    provider._run_hindsight_operation = lambda operation: operation(Client())
    provider._recall_types = []
    provider.handle_tool_call(
        "hindsight_recall",
        {"query": "locks", "tags": ["beta:specialist:dba"], "tags_match": "all_strict"},
    )
    properties = RECALL_SCHEMA["parameters"]["properties"]
    assert "tags" in properties
    assert captured["tags"] == ["beta:specialist:dba"]
    assert captured["tags_match"] == "all_strict"

