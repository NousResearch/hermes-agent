"""Skill requirements include tools offered through the scoped Tool Search bridge."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("has_bridge", [False, True])
def test_skills_follow_the_session_deferred_tool_scope(tmp_path, monkeypatch, has_bridge):
    from agent import system_prompt

    tool = "mcp__fixture__lookup"
    directory = tmp_path / "skills"
    for name, conditions in {
        "requires-lookup": f"requires_tools: [{tool}]\n    requires_toolsets: [fixture]",
        "fallback-lookup": f"fallback_for_tools: [{tool}]\n    fallback_for_toolsets: [fixture]",
        "requires-excluded": "requires_tools: [mcp__fixture__excluded]",
    }.items():
        skill_dir = directory / "general" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test tool requirement\nmetadata:\n"
            f"  hermes:\n    {conditions}\n---\nbody\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(system_prompt, "_agent_skills_dir", lambda _agent: directory)
    monkeypatch.setattr("model_tools.get_toolset_for_tool", lambda name: "fixture" if name == tool else None)
    scoped = Mock(return_value=frozenset({tool}))
    monkeypatch.setattr("agent.tool_executor._tool_search_scoped_names", scoped)
    names = {"skills_list", "tool_search", "tool_call"} if has_bridge else {"skills_list"}

    prompt = system_prompt._skills_prompt(SimpleNamespace(valid_tool_names=names, platform="cli"))

    assert ("requires-lookup" in prompt) is has_bridge
    assert ("fallback-lookup" in prompt) is not has_bridge
    assert "requires-excluded" not in prompt
    assert scoped.call_count == int(has_bridge)
