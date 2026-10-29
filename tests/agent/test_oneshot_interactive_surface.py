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


def _built_agent(monkeypatch, oneshot: bool, disabled=None):
    from agent.agent_init import _load_tools

    if oneshot:
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    else:
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    agent = SimpleNamespace(platform="cli", quiet_mode=True)
    _load_tools(agent, ["hermes-cli"], disabled)
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


@pytest.mark.parametrize("config_disables_clarify", [False, True])
def test_oneshot_resume_keeps_the_pinned_interactive_tool_bytes(monkeypatch, config_disables_clarify):
    # `hermes -z --resume <interactive session>`: the same-code pin's bytes head every request (prompt cache), so
    # neither the one-shot delegate_task variant nor the one-shot clarify gate may rewrite or drop a pinned tool;
    # a clarify the user's config disables since the pin still drops, as it would for an interactive resume.
    from tools.mcp_tool_agent import restore_agent_tool_prefix, tool_pin_version

    interactive = _built_agent(monkeypatch, oneshot=False)
    oneshot = _built_agent(monkeypatch, oneshot=True, disabled=["clarify"] if config_disables_clarify else None)
    restore_agent_tool_prefix(oneshot, {"version": tool_pin_version(), "tools": interactive.tools})

    expected = [t for t in interactive.tools if not (config_disables_clarify and t["function"]["name"] == "clarify")]
    assert oneshot.tools == expected


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


def test_skill_safety_rule_survives_prune_only_and_salvage_rewrites():
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY, SKILL_PRUNED_MARKER_PREFIX, ContextCompressor, _SUMMARY_END_MARKER,
        _skill_pruned_marker, salvage_grown_transcript,
    )
    from agent.context_compressor_tail import SKILL_SAFETY_RULE

    def texts(messages):
        return "\n".join(str(m.get("content") or "") for m in messages)

    # Prune-only: no handoff exists, so the row the prune turns into a marker carries the rule itself.
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        c = ContextCompressor(model="test", quiet_mode=True, threshold_percent=0.5, protect_first_n=2,
                              protect_last_n=4, proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    calls = [("skill_view", '{"name":"pdf"}', "P" * 9000)] + [("terminal", '{"cmd":"ls"}', ch * 9000) for ch in "AB"]
    calls += [("terminal", '{"cmd":"ls"}', "ok")] * 5
    messages = [{"role": "system", "content": "sys"}]
    for i, (tool, args, content) in enumerate(calls):
        messages += [{"role": "assistant", "content": "", "tool_calls": [{
            "id": f"c{i}", "type": "function", "function": {"name": tool, "arguments": args}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": content}]
    pruned, count = c.prune_tool_results_only(messages, current_tokens=120_000)
    assert count and SKILL_PRUNED_MARKER_PREFIX in texts(pruned) and texts(pruned).count(SKILL_SAFETY_RULE) == 1
    assert c._prune_old_tool_results(pruned, protect_tail_count=4)[1] == 0  # the ruled stub is still a stub

    # Salvage elides an oversized handoff; the rule must outlive the cut while marked rows remain.
    summary = "## Goal\n" + "x" * 20_000 + "\n\n" + SKILL_SAFETY_RULE + "\n\n" + _SUMMARY_END_MARKER
    grown = [{"role": "user", "content": summary, COMPRESSED_SUMMARY_METADATA_KEY: True},
             {"role": "assistant", "content": "", "tool_calls": [{
                 "id": "s", "type": "function", "function": {"name": "skill_view", "arguments": '{"name":"pdf"}'}}]},
             {"role": "tool", "tool_call_id": "s", "content": "[skill_view] name=pdf (9,000 chars) "
                                                                + _skill_pruned_marker("pdf")},
             {"role": "user", "content": "latest ask"}]
    salvaged = salvage_grown_transcript([{"role": "user", "content": "y" * 40_000}], grown)
    assert salvaged is not None and len(salvaged[0]["content"]) < len(summary)
    assert SKILL_SAFETY_RULE in salvaged[0]["content"]
