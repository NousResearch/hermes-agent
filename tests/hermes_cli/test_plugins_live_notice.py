"""``live_notice``: what an open chat is told when a plugin goes live in it.

The notice stays in the chat's history and rides every later request, so it names how to reach the
plugin (its skills, each MCP server's tool prefix) and never lists the tools one by one: a live run
delivered a 4,994-char notice that listed all of Blender's tools.
"""

from __future__ import annotations

import pytest

from agent.prompt_builder import CONTROL_FRAME_OPENERS
from hermes_cli.plugins_activation_live import live_notice
from tools.registry import registry

SKILL = "blender:blender"
TOOLS = ["mcp__blender__execute_blender_code", "mcp__blender__get_objects_summary",
         "mcp__blender__render_viewport_to_path"]


@pytest.fixture
def blender_tools():
    """The server's tools as the registry holds them once it connected."""
    for name in TOOLS:
        registry.register(name=name, toolset="mcp-blender", handler=lambda args, **kw: "{}",
                          schema={"name": name, "description": f"Run {name}. More detail.",
                                  "parameters": {"type": "object", "properties": {}}})
    yield TOOLS
    for name in TOOLS:
        registry.deregister(name)


def _activation(servers, skills=(SKILL,), deferred=None):
    return {"name": "blender", "key": "blender", "deferred": deferred or {},
            "live_now": {"mcp_servers": servers, "skills": [{"name": s, "description": ""} for s in skills]}}


def test_the_notice_names_the_skill_and_the_tool_prefix_and_lists_no_tool(blender_tools):
    note = live_notice(_activation([
        {"name": "blender", "connected": True, "tools": blender_tools},
        {"name": "blender-render", "connected": False, "tools": [], "error": "spawn failed"},
    ]))

    assert f"{SKILL}: load it with skill_view" in note
    assert "MCP server blender: 3 tools, mcp__blender__*." in note
    assert "MCP server blender-render: not connected (spawn failed)." in note
    assert all(tool not in note for tool in blender_tools)
    for bridge in ("tool_search", "tool_describe", "tool_call"):
        assert bridge in note
    # A Hermes control frame, so a hosted room relabels a reply that copies it.
    assert note.startswith("[") and note[1:].startswith(CONTROL_FRAME_OPENERS) and note.endswith("]")
    assert len(note) < 600


def test_a_plugin_with_only_skills_is_still_announced_and_one_with_nothing_live_is_not():
    note = live_notice(_activation([]))

    assert SKILL in note and "tool_search" not in note
    assert live_notice(_activation([], skills=())) == ""
