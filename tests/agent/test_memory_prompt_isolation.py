"""Prompt isolation for #51797 must not break the #65429 memory-tool store.

Real agent initialization, memory handler and prompt cache boundary run offline;
the model client is a stub (no inference).
"""
import json

import pytest

from agent.system_prompt import invalidate_system_prompt
from run_agent import AIAgent
from tests.agent.test_skip_memory_store import _FakeOpenAI
from tools.memory_tool import memory_tool


def _prepare_prompt(agent, text):
    from agent import conversation_loop as loop
    from agent.turn_context import build_turn_context
    return build_turn_context(
        agent, text, None, None, None, None, None,
        restore_or_build_system_prompt=loop._restore_or_build_system_prompt,
        install_safe_stdio=loop._install_safe_stdio,
        sanitize_surrogates=loop._sanitize_surrogates,
        summarize_user_message_for_log=loop._summarize_user_message_for_log,
        set_session_context=loop.set_session_context,
        set_current_write_origin=loop.set_current_write_origin, ra=loop._ra,
    ).active_system_prompt


@pytest.mark.parametrize("isolated", [False, True])
def test_memory_prompt_gate_preserves_store_and_cache(monkeypatch, tmp_path, isolated):
    home = tmp_path / "home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("memory-gate-sentinel\n", encoding="utf-8")
    (memories / "USER.md").write_text("user-gate-sentinel\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
    agent = AIAgent(api_key="test-key", base_url="http://test", provider="openrouter",
                    api_mode="chat_completions", max_iterations=1, quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
                    skip_memory_prompt=isolated)
    try:
        assert agent._memory_store is not None
        assert agent._memory_manager is None
        first = _prepare_prompt(agent, "first probe")
        assert ("memory-gate-sentinel" in first) is not isolated
        assert ("user-gate-sentinel" in first) is not isolated
        result = json.loads(memory_tool(action="add", target="memory",
                                       content="persisted-tool-sentinel", store=agent._memory_store))
        assert result["success"] is True
        assert "persisted-tool-sentinel" in (memories / "MEMORY.md").read_text(encoding="utf-8")
        assert _prepare_prompt(agent, "second probe") == first
        invalidate_system_prompt(agent)  # sanctioned compression boundary
        rebuilt = _prepare_prompt(agent, "boundary probe")
        for sentinel in ("memory-gate-sentinel", "user-gate-sentinel", "persisted-tool-sentinel"):
            assert (sentinel in rebuilt) is not isolated
        from agent.context_breakdown import _memory_blocks
        blocks = " ".join(_memory_blocks(agent))
        assert ("memory-gate-sentinel" in blocks) is not isolated
    finally:
        agent.close()


@pytest.mark.parametrize("isolated", [False, True])
def test_review_fork_preserves_memory_prompt_gate(isolated):
    from unittest.mock import patch
    from agent.background_review import build_cache_parity_fork
    from tests.agent.test_background_review_cache_parity import _make_agent_stub, _make_recorder_class

    parent = _make_agent_stub(AIAgent)
    parent.skip_memory_prompt = isolated
    with patch("run_agent.AIAgent", _make_recorder_class()):
        fork, _, _ = build_cache_parity_fork(parent, max_iterations=1)
        assert fork.skip_memory_prompt is isolated
        assert fork._cached_system_prompt == parent._cached_system_prompt


@pytest.mark.parametrize("isolated", [False, True])
def test_memory_prompt_gate_leaves_provider_tools_available(monkeypatch, tmp_path, isolated):
    from types import SimpleNamespace
    from agent.memory_manager_tools import inject_memory_provider_tools
    from agent.system_prompt import _memory_parts

    manager = SimpleNamespace(
        build_system_prompt=lambda: "EXTERNAL_MEMORY_SENTINEL",
        get_all_tool_schemas=lambda: [{"name": "provider_recall", "parameters": {"type": "object"}}],
    )
    agent = SimpleNamespace(skip_memory_prompt=isolated, _memory_store=None,
                            _memory_manager=manager, tools=[], valid_tool_names=set(),
                            enabled_toolsets=["memory"], disabled_toolsets=None)
    assert inject_memory_provider_tools(agent) == 1
    assert "provider_recall" in agent.valid_tool_names
    assert ("EXTERNAL_MEMORY_SENTINEL" in " ".join(_memory_parts(agent))) is not isolated
