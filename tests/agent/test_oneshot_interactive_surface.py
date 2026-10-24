"""A one-shot run (``hermes chat -q`` / ``-z``) has nobody to ask, steer, or live-control children for.

It drops ``clarify``, the mid-turn steering briefing and ``delegate_task``'s live-control parameters on both
paths that build its surface (the agent's tool build and the gateway's session inventory). An interactive
session, the control in every test, keeps the full surface with byte-identical schemas (prompt cache).
The [SKILL_PRUNED] reload rule is only meaningful once compaction created markers, so it rides the
compaction handoff instead of every fresh system prompt.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

LIVE_CONTROL = {"action", "subagent_id", "message"}


def _built_agent(monkeypatch, oneshot: bool):
    from agent.agent_init import _load_tools

    if oneshot:
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    else:
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    agent = SimpleNamespace(platform="cli", quiet_mode=True)
    _load_tools(agent, ["hermes-cli"], None)
    return agent


def _by_name(agent):
    return {tool["function"]["name"]: tool["function"] for tool in agent.tools}


def test_oneshot_tool_build_drops_clarify_and_delegate_live_control(monkeypatch):
    import model_tools

    interactive = _built_agent(monkeypatch, oneshot=False)
    tools = _by_name(interactive)
    assert "clarify" in tools and LIVE_CONTROL <= set(tools["delegate_task"]["parameters"]["properties"])
    registry_delegate = next(t["function"] for t in model_tools.get_tool_definitions(
        enabled_toolsets=["hermes-cli"], disabled_toolsets=interactive.disabled_toolsets, quiet_mode=True)
        if t["function"]["name"] == "delegate_task")
    assert tools["delegate_task"] == registry_delegate  # interactive schema untouched

    oneshot = _built_agent(monkeypatch, oneshot=True)
    tools = _by_name(oneshot)
    assert "clarify" not in tools and "clarify" in oneshot.disabled_toolsets  # bridge + children lose it too
    assert "skill_manage" in tools  # maintainer ruling: one-shots may still author skills
    delegate = tools["delegate_task"]
    assert not LIVE_CONTROL & set(delegate["parameters"]["properties"])
    assert "steer" not in delegate["description"] and "BETWEEN your turns" not in delegate["description"]


def test_gateway_oneshot_inventory_drops_clarify(tmp_path, monkeypatch):
    from gateway.session_local_inventory import _policy_tools
    from gateway.session_policy import build_policy, policy_scope

    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)

    def names(source):
        policy = build_policy({"source": source, "cwd": str(tmp_path), "model": "m"}, {})
        with policy_scope(policy):
            return {t["function"]["name"] for t in _policy_tools(policy, {})[1]}

    assert "clarify" not in names("oneshot")
    assert "clarify" in names("cli")


def test_oneshot_prompt_drops_steering_note(monkeypatch):
    from agent import system_prompt

    def guidance(oneshot):
        agent = SimpleNamespace(valid_tool_names={"terminal"}, model="m", _tool_use_enforcement=False,
                                _execution_guidance=False, _kanban_worker_guidance="", _oneshot_session=oneshot)
        return "\n".join(p for p in system_prompt._guidance_parts(agent) if p)

    assert "OUT-OF-BAND USER MESSAGE" in guidance(False)
    assert "OUT-OF-BAND USER MESSAGE" not in guidance(True)


def _compress(messages):
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="test/model", quiet_mode=True, protect_first_n=1, protect_last_n=2)
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nBuild the report.\n\n## Completed Actions\n1. Worked on it."
    with patch.object(c, "_find_tail_cut_by_tokens", return_value=9), \
            patch("agent.context_compressor.call_llm", return_value=response):
        result = c.compress(messages, force=True)
    return next(m["content"] for m in result if ContextCompressor.classify_summary_content(m.get("content")))


def _transcript(tool, tool_content):
    return [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Build the PDF report"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {"name": tool, "arguments": '{"name":"pdf"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        {"role": "assistant", "content": "Loaded the skill, working."},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "more work " + "z" * 500},
        {"role": "user", "content": "latest ask"},
        {"role": "assistant", "content": "ack"},
    ]


@pytest.mark.parametrize("pruned", [True, False])
def test_skill_safety_rule_rides_the_compaction_handoff_not_the_system_prompt(pruned):
    from agent import system_prompt
    from agent.context_compressor import _skill_pruned_marker

    fresh = system_prompt._tool_guidance_block(SimpleNamespace(valid_tool_names={"skill_manage"},
                                                               _kanban_worker_guidance=""))
    assert fresh and "SKILL_PRUNED" not in fresh and "Skill Safety Rule" not in fresh
    # Control: no skill was ever loaded, so no marker exists and the handoff stays rule-free.
    summary = _compress(_transcript("skill_view", "[skill_view] name=pdf (48,201 chars) " + _skill_pruned_marker("pdf"))
                        if pruned else _transcript("terminal", "ok"))
    assert ("## Skill Safety Rule\nA skill placeholder containing `[SKILL_PRUNED]`" in summary) is pruned
    assert summary.count("## Skill Safety Rule") == int(pruned)
