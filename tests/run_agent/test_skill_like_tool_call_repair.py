"""Tests for repairing skill-like invalid tool calls."""

import json
from types import SimpleNamespace

from run_agent import AIAgent


def _make_tool_call(name: str, arguments: str = "{}"):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_repair_skill_like_tool_call_rewrites_to_skill_view(monkeypatch):
    agent = AIAgent.__new__(AIAgent)
    agent.valid_tool_names = {"skill_view", "skills_list"}

    monkeypatch.setattr(
        "tools.skills_tool.skills_list",
        lambda: json.dumps({"skills": [{"name": "github-auth"}]}),
    )

    tool_call = _make_tool_call("github-auth")
    repaired = agent._repair_skill_name_tool_call(tool_call)

    assert repaired is True
    assert tool_call.function.name == "skill_view"
    assert json.loads(tool_call.function.arguments) == {"name": "github-auth"}


def test_repair_skill_like_tool_call_preserves_existing_arguments(monkeypatch):
    agent = AIAgent.__new__(AIAgent)
    agent.valid_tool_names = {"skill_view", "skills_list"}

    monkeypatch.setattr(
        "tools.skills_tool.skills_list",
        lambda: json.dumps({"skills": [{"name": "text2sql"}]}),
    )

    tool_call = _make_tool_call("text2sql", arguments=json.dumps({"file_path": "references/schema.md"}))
    repaired = agent._repair_skill_name_tool_call(tool_call)

    assert repaired is True
    assert tool_call.function.name == "skill_view"
    assert json.loads(tool_call.function.arguments) == {
        "file_path": "references/schema.md",
        "name": "text2sql",
    }


def test_repair_skill_like_tool_call_skips_when_skill_view_unavailable(monkeypatch):
    agent = AIAgent.__new__(AIAgent)
    agent.valid_tool_names = {"terminal"}

    monkeypatch.setattr(
        "tools.skills_tool.skills_list",
        lambda: json.dumps({"skills": [{"name": "github-auth"}]}),
    )

    tool_call = _make_tool_call("github-auth")
    repaired = agent._repair_skill_name_tool_call(tool_call)

    assert repaired is False
    assert tool_call.function.name == "github-auth"
