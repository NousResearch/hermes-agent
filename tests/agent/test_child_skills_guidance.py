"""A delegated child's skills block defers to its brief; the parent's block is unchanged."""

from agent.delegation_context import delegated_child_context
from agent.prompt_builder import CHILD_SKILLS_LOAD_GUIDANCE, build_skills_system_prompt


def _skills_dir(tmp_path):
    d = tmp_path / "skills" / "misc" / "demo-skill"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo skill for tests.\n---\n# Demo\n",
                                encoding="utf-8")
    return tmp_path / "skills"


def _prompt(skills):
    return build_skills_system_prompt(available_tools={"skill_view", "skills_list", "skill_manage"},
                                      skills_dir_override=skills)


def test_child_skills_block_drops_must_load_and_keeps_index_while_parent_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    skills = _skills_dir(tmp_path)
    parent_before = _prompt(skills)
    with delegated_child_context():
        child = _prompt(skills)
    parent_after = _prompt(skills)

    assert "MUST load" in parent_before and parent_after == parent_before
    assert child.startswith(CHILD_SKILLS_LOAD_GUIDANCE)
    assert "MUST load" not in child and "Err on the side of" not in child
    assert "demo-skill: Demo skill for tests." in child
