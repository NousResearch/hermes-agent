"""End-to-end coverage for the skills index v2 prompt and skill_describe."""

from pathlib import Path


def _seed_skills(skills_dir: Path, n: int = 100, n_categories: int = 10) -> None:
    skills_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        category = f"cat-{index % n_categories:02d}"
        skill_dir = skills_dir / category / f"skill-{index:03d}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        priority = "critical" if index < 5 else "normal"
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: skill-{index:03d}\n"
            f"description: synthetic skill {index}\n"
            f"priority: {priority}\n---\nbody\n",
            encoding="utf-8",
        )


def test_v2_budget_and_full_reachability(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    _seed_skills(skills_dir, n=100, n_categories=10)
    monkeypatch.setattr("agent.skill_inventory.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.skill_inventory.get_all_skills_dirs", lambda: [skills_dir])
    from agent.skill_inventory import clear_inventory_cache

    clear_inventory_cache(clear_snapshot=True)

    from agent.prompt_builder import build_skills_system_prompt

    prompt = build_skills_system_prompt(index_v2=True, index_token_budget=2000)
    assert len(prompt) // 4 <= 2200

    for index in range(5):
        assert f"skill-{index:03d}: synthetic skill {index}" in prompt

    from tools.skills_tool import skill_describe

    for index in range(100):
        result = skill_describe(names=[f"skill-{index:03d}"])
        assert result["success"], f"skill-{index:03d} not reachable"

