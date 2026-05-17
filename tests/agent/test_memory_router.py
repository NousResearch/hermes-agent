import json

from agent.memory_provider import MemoryProvider
from agent.memory_router import (
    build_recall_gate_context,
    decide_recall,
    heuristic_recall_decision,
    stable_recall_key,
)
from agent.memory_manager import MemoryManager


class DummyProvider(MemoryProvider):
    name = "dummy"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def initialize(self, session_id: str, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return f"prefetch:{query}:{session_id}"

    def recall_now(self, query: str, **kwargs) -> str:
        self.calls.append((query, kwargs))
        return json.dumps({"query": query, "kwargs": kwargs}, ensure_ascii=False)


def test_gate_context_strips_memory_blocks_and_tool_messages():
    ctx = build_recall_gate_context(
        "继续图记忆 recall",
        [
            {"role": "user", "content": "hello <memory-context>SECRET</memory-context> visible"},
            {"role": "tool", "content": "tool output should not be included"},
            {"role": "assistant", "content": "answer"},
        ],
        platform_context={"platform": "discord", "user_id": "raw-user-123"},
        session_metadata={"session_id": "s1"},
    )

    dumped = json.dumps(ctx, ensure_ascii=False)
    assert "SECRET" not in dumped
    assert "tool output" not in dumped
    assert "raw-user-123" not in dumped
    assert "\"s1\"" not in dumped
    assert ctx["platform_context"]["user_id"].startswith("sha256:")
    assert ctx["session_metadata"]["session_id"].startswith("sha256:")
    assert "visible" in dumped
    assert ctx["platform_context"]["platform"] == "discord"


def test_gate_context_caps_long_current_message():
    ctx = build_recall_gate_context(
        "x" * 5000,
        [{"role": "assistant", "content": "y" * 5000}],
        platform_context={"platform": "discord"},
        max_chars=500,
    )

    assert len(json.dumps(ctx, ensure_ascii=False)) <= 500
    assert ctx["recent_turns"] == []
    assert len(ctx["current_message"]) < 5000


def test_heuristic_skips_short_ack_and_routes_project_memory():
    assert heuristic_recall_decision("好的").should_recall is False

    decision = heuristic_recall_decision("Graphiti 的 recall gate 设计")
    assert decision.should_recall is True
    assert decision.depth == "light"
    assert decision.sources == ["graph"]


def test_default_recall_router_is_privacy_preserving_heuristic():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["memory"]["recall_router"]
    assert cfg["enabled"] is True
    assert cfg["strategy"] == "heuristic"
    assert cfg["timeout"] == 8.0


def test_heuristic_caps_evidence_to_standard_for_auto():
    decision = decide_recall(
        {"current_message": "找一下之前认证层的原文证据"},
        strategy="heuristic",
        max_depth="standard",
        max_budget="small",
    )
    assert decision.should_recall is True
    assert decision.depth == "standard"
    assert decision.sources == ["graph", "session_fts"]
    assert decision.budget == "small"


def test_stable_recall_key_normalizes_whitespace_and_sources():
    a = stable_recall_key("Graphiti   recall", depth="light", sources=["session_fts", "graph"])
    b = stable_recall_key("graphiti recall", depth="light", sources=["graph", "session_fts"])
    assert a == b


def test_memory_manager_recall_now_passes_strategy_metadata():
    provider = DummyProvider()
    manager = MemoryManager()
    manager.add_provider(provider)

    result = manager.recall_now_all(
        "Graphiti recall",
        mode="auto",
        depth="standard",
        sources=["graph", "session_fts"],
        budget="small",
        provenance="ids",
        session_id="session-1",
    )

    assert "Graphiti recall" in result
    assert provider.calls[0][0] == "Graphiti recall"
    kwargs = provider.calls[0][1]
    assert kwargs["mode"] == "auto"
    assert kwargs["depth"] == "standard"
    assert kwargs["sources"] == ["graph", "session_fts"]
    assert kwargs["budget"] == "small"
    assert kwargs["provenance"] == "ids"
    assert kwargs["session_id"] == "session-1"
