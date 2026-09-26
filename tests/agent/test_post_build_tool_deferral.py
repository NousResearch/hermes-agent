"""Behavior contracts for provider-owned tools named in ``tools.tool_search.defer``.

Regression test for #110341: memory-provider and context-engine tools join ``agent.tools``
after Tool Search assembly already ran, so naming them in ``defer`` had no effect.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.context_engine import ContextEngine
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider

DEFERRED = ("stub_recall", "stub_memory_search")


class StubEngine(ContextEngine):
    name = "stub"

    def __init__(self):
        self.calls = []

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, **kwargs):
        return messages

    def get_tool_schemas(self):
        return [
            {"name": "stub_recall", "description": "Recall", "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "stub_status", "description": "Status", "parameters": {"type": "object", "properties": {}}},
        ]

    def handle_tool_call(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        return json.dumps({"name": name})


class StubProvider(MemoryProvider):
    name = "stub"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [{"name": "stub_memory_search", "description": "Search memory", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]

    def handle_tool_call(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        return json.dumps({"name": name})


def _call(name, arguments, call_id=None):
    return SimpleNamespace(id=call_id or f"id-{name}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _bridge_call(name, arguments):
    return _call("tool_call", {"calls": [{"name": name, "arguments": arguments}]})


def _names(tool_defs):
    return {t["function"]["name"] for t in tool_defs}


def _listing(agent):
    return next(t["function"]["description"] for t in agent.tools if t["function"]["name"] == "tool_search")


def _run(agent, *calls, messages=None, concurrent=False):
    """Run one assistant tool batch through the real executor; returns the tool result contents."""
    from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
    messages = messages if messages is not None else [{"role": "user", "content": "marker"}]
    start = len(messages)
    execute = execute_tool_calls_concurrent if concurrent else execute_tool_calls_sequential
    execute(agent, SimpleNamespace(content="", tool_calls=list(calls)), messages, "task")
    return [m["content"] for m in messages[start:] if m.get("role") == "tool"]


def _use_config(monkeypatch, defer):
    config = {"context": {"engine": "stub"}, "tools": {"tool_search": {"enabled": "on", "defer": list(defer)}},
              "agent": {}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)


@pytest.fixture
def build_agent(monkeypatch):
    """Build real AIAgents with a stub engine and memory provider; only external boundaries are faked."""
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_k: 204_800)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda **_k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", Mock())

    def build():
        from run_agent import AIAgent
        engine, provider, manager = StubEngine(), StubProvider(), MemoryManager()
        manager.add_provider(provider)
        monkeypatch.setattr("plugins.context_engine.load_context_engine", lambda _name: engine)
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://example.test", quiet_mode=True,
                        skip_context_files=True, skip_memory=False, memory_manager=manager)
        return agent, engine, provider

    return build


@pytest.fixture
def deferred(build_agent, monkeypatch):
    _use_config(monkeypatch, DEFERRED)
    return build_agent()


def test_named_tools_leave_the_request_and_join_the_catalog(deferred):
    agent, _, _ = deferred
    assert not set(DEFERRED) & _names(agent.tools)
    assert not set(DEFERRED) & agent.valid_tool_names
    assert "stub_status" in _names(agent.tools)
    assert all(name in _listing(agent) for name in DEFERRED)
    assert _names(agent._deferred_post_build_tools) == set(DEFERRED)


def test_unnamed_tools_stay_loaded(build_agent, monkeypatch):
    _use_config(monkeypatch, [])
    agent, _, _ = build_agent()
    expected = {"stub_recall", "stub_status", "stub_memory_search"}
    assert expected <= _names(agent.tools)
    assert expected <= agent.valid_tool_names
    assert agent._deferred_post_build_tools == []


def test_deferred_calls_reach_their_own_handlers(deferred):
    agent, engine, provider = deferred
    messages = [{"role": "user", "content": "current-turn-marker"}]
    _run(agent, _bridge_call("stub_recall", {"query": "x"}), messages=messages)
    assert [c[0] for c in engine.calls] == ["stub_recall"]
    assert any(m.get("content") == "current-turn-marker" for m in engine.calls[0][2]["messages"])

    _run(agent, _bridge_call("stub_memory_search", {"query": "y"}))
    assert [(c[0], c[1]) for c in provider.calls] == [("stub_memory_search", {"query": "y"})]


def test_bridge_searches_describes_and_validates_deferred_tools(deferred):
    agent, engine, _ = deferred
    search, describe = _run(agent, _call("tool_search", {"queries": ["stub"]}, "s1"),
                            _call("tool_describe", {"names": ["stub_recall"]}, "d1"))
    assert all(name in search for name in DEFERRED)
    assert "stub_recall" in describe and '"query"' in describe and '"required"' in describe

    # Bridge lookups are parallel-safe, so this batch runs through the concurrent executor.
    results = _run(agent, _call("tool_search", {"queries": ["stub"]}, "c1"),
                   _call("tool_search", {"queries": ["stub"]}, "c2"), concurrent=True)
    assert len(results) == 2 and all(name in r for r in results for name in DEFERRED)

    (blocked,) = _run(agent, _bridge_call("stub_recall", {}))
    assert "query" in blocked
    assert engine.calls == []


def test_refresh_keeps_tool_bytes_and_deferred_calls(deferred):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    agent, engine, _ = deferred
    before = json.dumps(agent.tools)

    refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True)
    assert json.dumps(agent.tools) == before
    refresh_agent_mcp_tools(agent, content_aware=True)
    assert json.dumps(agent.tools) == before

    assert _names(agent._deferred_post_build_tools) == set(DEFERRED)
    _run(agent, _bridge_call("stub_recall", {"query": "x"}))
    assert [c[0] for c in engine.calls] == ["stub_recall"]


def test_refresh_drops_a_deferred_tool_whose_toolset_is_disabled(deferred):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    agent, engine, provider = deferred

    refresh_agent_mcp_tools(agent, disabled_override=["memory"])
    assert _names(agent._deferred_post_build_tools) == {"stub_recall"}
    (search,) = _run(agent, _call("tool_search", {"queries": ["stub"]}))
    assert "stub_memory_search" not in search and "stub_recall" in search
    _run(agent, _bridge_call("stub_memory_search", {"query": "x"}))
    assert provider.calls == []
    _run(agent, _bridge_call("stub_recall", {"query": "x"}))
    assert [c[0] for c in engine.calls] == ["stub_recall"]

    # The sent bytes stay frozen until the next content-aware rebuild (compaction boundary).
    refresh_agent_mcp_tools(agent, content_aware=True)
    assert "stub_memory_search" not in _listing(agent) and "stub_recall" in _listing(agent)


def test_refresh_routes_a_deferred_engine_tool_it_admits(build_agent, monkeypatch):
    """The bridge names stay the same across these refreshes, so the request bytes do not change,
    but engine routing must follow the deferred catalog or an admitted call fails as unknown."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    _use_config(monkeypatch, [*DEFERRED, "stub_status", "stub_recall_new"])
    agent, engine, _ = build_agent()

    refresh_agent_mcp_tools(agent, enabled_override=["memory"], content_aware=True)
    assert "stub_recall" not in _names(agent._deferred_post_build_tools)
    refresh_agent_mcp_tools(agent, enabled_override=["memory", "context_engine"])
    _run(agent, _bridge_call("stub_recall", {"query": "re-enabled"}))
    assert [c[0] for c in engine.calls] == ["stub_recall"]

    schemas = [{**s, "name": "stub_recall_new"} if s["name"] == "stub_recall" else s
               for s in engine.get_tool_schemas()]
    monkeypatch.setattr(engine, "get_tool_schemas", lambda: schemas)
    refresh_agent_mcp_tools(agent, preserve_prefix=True)
    _run(agent, _bridge_call("stub_recall_new", {"query": "renamed"}))
    assert [c[0] for c in engine.calls] == ["stub_recall", "stub_recall_new"]


def test_deferral_follows_each_profile(build_agent, monkeypatch, tmp_path):
    homes = {}
    for label, defer in (("a", "[stub_recall, stub_memory_search]"), ("b", "[]")):
        home = tmp_path / label
        home.mkdir()
        (home / "config.yaml").write_text(
            f"context:\n  engine: stub\ntools:\n  tool_search:\n    enabled: 'on'\n    defer: {defer}\n",
            encoding="utf-8")
        homes[label] = home

    built = []
    for label in ("a", "b", "a"):
        monkeypatch.setenv("HERMES_HOME", str(homes[label]))
        built.append((label, *build_agent()))

    for label, agent, _, _ in built:
        if label == "b":
            assert agent._deferred_post_build_tools == []
            assert set(DEFERRED) <= _names(agent.tools)
        else:
            assert _names(agent._deferred_post_build_tools) == set(DEFERRED)
            assert not set(DEFERRED) & _names(agent.tools)

    a_agents = [(agent, provider) for label, agent, _, provider in built if label == "a"]
    monkeypatch.setenv("HERMES_HOME", str(homes["a"]))
    for agent, provider in a_agents:
        _run(agent, _bridge_call("stub_memory_search", {"query": "x"}))
    assert [len(provider.calls) for _, provider in a_agents] == [1, 1]
