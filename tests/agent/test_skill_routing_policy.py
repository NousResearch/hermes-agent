from agent.prompt_builder import (
    build_skills_system_prompt,
    clear_skills_system_prompt_cache,
)


def test_skill_routing_policy_is_selective_and_search_aware(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "tools" / "python-debug"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: python-debug\ndescription: Debug Python scripts\n---\n",
        encoding="utf-8",
    )
    clear_skills_system_prompt_cache(clear_snapshot=True)

    result = build_skills_system_prompt()

    assert "mandatory routing check" in result
    assert 'skills_list(query="task summary", limit=5)' in result
    assert "Do not reload identical skill content" in result
    assert "If no skill clearly matches, proceed without loading one" in result
    assert "Err on the side of loading" not in result
