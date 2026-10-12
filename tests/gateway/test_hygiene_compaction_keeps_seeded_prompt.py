"""Gateway hygiene / gateway /compress must not persist a reduced-toolset prompt over a live session.

Both paths compact with a detached ``AIAgent(enabled_toolsets=["memory"])`` seeded with the session's
stored prompt (``_seed_hygiene_system_prompt``). Compaction rebuilds the prompt at the commit boundary
(#98426) and the commit persists the result, so without the seed being honoured the session row lost the
skills index / provider blocks and every later fresh agent restored those degraded bytes verbatim.
"""
from types import SimpleNamespace
from unittest.mock import patch

from agent import conversation_compression as cc
from gateway.run import _seed_hygiene_system_prompt

STORED = "You are Hermes.\n\n## Skills (mandatory)\n<available_skills>\n- a: b\n</available_skills>"
DEGRADED = "You are Hermes."


def _agent(**over):
    built = []

    def _build(system_message=None):
        built.append(system_message)
        return DEGRADED

    def _invalidate():
        agent._cached_system_prompt = None

    fields = dict(_cached_system_prompt=None, session_id="sess-1",
                  _build_system_prompt=_build, _invalidate_system_prompt=_invalidate)
    fields.update(over)
    agent = SimpleNamespace(**fields)
    return agent, built


def _rebuild(agent):
    with patch.object(cc, "_refresh_agent_tool_definitions", return_value=False), \
         patch("agent.system_prompt.reconstruct_static_prefix"):
        return cc._rebuild_system_prompt_at_boundary(agent, "")


def test_seeded_hygiene_agent_keeps_the_stored_prompt_at_the_commit_boundary():
    agent, built = _agent()
    assert _seed_hygiene_system_prompt(agent, {"system_prompt": STORED}) is True

    result = _rebuild(agent)

    assert result is STORED
    assert agent._cached_system_prompt is STORED
    assert built == []

    empty, built_empty = _agent()
    assert _seed_hygiene_system_prompt(empty, None) is False
    assert _rebuild(empty) == ""
    assert built_empty == []


def test_live_agent_compaction_still_rebuilds():
    agent, built = _agent(_cached_system_prompt=STORED)

    assert _rebuild(agent) == DEGRADED
    assert built == [""]
