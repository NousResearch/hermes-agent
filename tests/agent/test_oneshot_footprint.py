"""One-shot sessions (``hermes chat -q``) drop the self-improvement footprint.

Across 21 one-shot benchmark trajectories the agent created 7 skills and patched a bundled one
mid-task, spent 37 of ~215 tool calls on skill_view/skill_manage, and spawned review subagents of
its own work. None of that has a consumer in a finite run. The marker is the same
``HERMES_SINGLE_QUERY_SESSION`` the approval gate and delegation dispatcher read, so an interactive
session — the control in every test here — keeps the full surface.
"""

import pytest

from agent import oneshot_footprint
from agent.prompt_builder import build_skills_system_prompt


@pytest.fixture
def oneshot(monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")


def _tools(*names):
    return [{"type": "function", "function": {"name": n}} for n in names]


def test_oneshot_hides_skill_manage_and_skill_authoring_coaching(oneshot, interactive_prompt, tmp_path):
    """-q: no skill_manage tool and a skills prompt that neither asks to save/patch skills nor pushes process
    skills; skill reading stays. The interactive prompt for the same skills dir is the control."""
    kept = {t["function"]["name"] for t in oneshot_footprint.prune_oneshot_tools(
        _tools("skill_manage", "skill_view", "skills_list", "terminal"))}
    assert "skill_manage" not in kept and {"skill_view", "skills_list", "terminal"} <= kept

    prompt = build_skills_system_prompt(available_tools={"skill_view", "skills_list"}, skills_dir_override=_skills_dir(tmp_path))
    assert "demo-skill" in prompt and "skill_view" in prompt
    assert "skill_manage" not in prompt and "offer to save as a skill" not in prompt
    assert "skill_manage" in interactive_prompt and "offer to save as a skill" in interactive_prompt


def _skills_dir(tmp_path):
    d = tmp_path / "skills" / "misc" / "demo-skill"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo skill for tests.\n---\n# Demo\n", encoding="utf-8")
    return tmp_path / "skills"


@pytest.fixture
def interactive_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    prompt = build_skills_system_prompt(available_tools={"skill_view", "skills_list", "skill_manage"},
                                        skills_dir_override=_skills_dir(tmp_path))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    return prompt


def test_oneshot_delegation_budget_charges_total_children_then_refuses(oneshot, monkeypatch):
    from tools import delegate_tool

    monkeypatch.setattr(delegate_tool, "_get_oneshot_max_children", lambda: 2)
    parent = type("P", (), {})()
    assert delegate_tool._oneshot_spawn_budget(parent, 1) is None
    assert delegate_tool._oneshot_spawn_budget(parent, 1) is None
    err = delegate_tool._oneshot_spawn_budget(parent, 1)
    assert err and "oneshot_max_children" in err
    # Interactive sessions are never charged, whatever the count.
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION")
    assert delegate_tool._oneshot_spawn_budget(parent, 50) is None
