from __future__ import annotations

from toolsets import get_toolset
from model_tools import get_tool_definitions


def test_skills_toolset_includes_skill_candidates():
    toolset = get_toolset("skills")
    assert "skill_candidates" in toolset["tools"]


def test_skill_candidates_is_exposed_when_skills_toolset_enabled():
    definitions = get_tool_definitions(enabled_toolsets=["skills"], quiet_mode=True)
    names = [t["function"]["name"] for t in definitions]
    assert "skill_candidates" in names
