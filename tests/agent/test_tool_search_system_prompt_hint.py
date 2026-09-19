"""tool_search system-prompt hint when the tool-search bridge is active.

#107515: when tools.tool_search.enabled is auto/on and the bridge is
activated, MCP/plugin tools disappear from the model-visible tools array.
The only discovery hint lived in the tool_search *tool description*. Models
that skip that schema conclude the hidden tool doesn't exist.
"""

from __future__ import annotations

from types import SimpleNamespace

import agent.system_prompt as system_prompt


def test_bridge_active_injects_tool_search_hint():
    # Fake agent with post-assembly names that include the bridge.
    agent = SimpleNamespace(valid_tool_names={"tool_search", "tool_describe", "tool_call", "terminal"})
    block = system_prompt._tool_guidance_block(agent) or ""
    assert "tool_search" in block
    # Must tell the model to search before concluding a tool is missing.
    assert "missing" in block.lower() or "not see" in block.lower() or "search" in block.lower()


def test_bridge_inactive_no_hint():  # fail-open CONTROL — already passes on main
    agent = SimpleNamespace(valid_tool_names={"terminal", "read_file"})
    block = system_prompt._tool_guidance_block(agent) or ""
    assert "tool_search" not in block


def test_empty_tools_no_hint():  # fail-open
    agent = SimpleNamespace(valid_tool_names=set())
    block = system_prompt._tool_guidance_block(agent) or ""
    assert "tool_search" not in (block or "")


def test_tool_search_guidance_names_the_bridge():
    from agent.prompt_builder import TOOL_SEARCH_GUIDANCE

    assert "tool_search" in TOOL_SEARCH_GUIDANCE
