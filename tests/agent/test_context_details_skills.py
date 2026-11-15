"""Expanded costs follow the rendered skill index, including older prompt layouts."""
from types import SimpleNamespace

import pytest

from agent.context_breakdown import compute_context_details, render_context_details_lines


@pytest.mark.parametrize("layout", ["volatile", "stable", "both"])
def test_context_details_find_the_rendered_skill_index(tmp_path, monkeypatch, layout):
    from agent import system_prompt

    home = tmp_path / "home"
    skill = home / "skills" / "context-probe" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: context-probe\ndescription: A context cost probe.\n---\nUse this probe.\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    index = "<available_skills>\n  - context-probe: A context cost probe.\n</available_skills>"
    parts = {"stable": "identity", "context": "", "volatile": "memory"}
    if layout == "stable":
        parts["stable"] = index
    else:
        parts["volatile"] = index
        if layout == "both":
            parts["stable"] = "<available_skills>\n  - stale-skill: Old index.\n</available_skills>"
    monkeypatch.setattr(system_prompt, "build_system_prompt_parts", lambda agent: parts)

    details = compute_context_details(SimpleNamespace(tools=[]))
    assert [entry["name"] for entry in details["skills"]] == ["context-probe"]
    cost = details["skills"][0]
    assert cost["index_tokens"] > 0 and cost["skill_md_tokens"] > 0
    assert "context-probe" in "\n".join(render_context_details_lines(details))
