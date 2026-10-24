"""A prompt budget removes descriptions, never the skill catalogue or loadable names."""

from agent import prompt_builder as pb
from agent.model_metadata import estimate_tokens_rough
from tests.agent.test_skills_index_delta import _home, _write_skill


def test_budget_change_rebuilds_index_without_losing_names(tmp_path, monkeypatch):
    skills = _home(tmp_path, monkeypatch)
    for i in range(40):
        _write_skill(skills, f"skill-{i:02}", "日本語の長い説明です" * 10)
    config = tmp_path / "config.yaml"
    config.write_text("skills:\n  listing_max_tokens: 0\n", encoding="utf-8")
    full = pb.build_skills_system_prompt()
    assert "skill-00:" in full
    config.write_text("skills:\n  listing_max_tokens: 500\n", encoding="utf-8")
    small = pb.build_skills_system_prompt()
    assert estimate_tokens_rough(small) < estimate_tokens_rough(full) / 2
    assert "[names only]" in small
    assert all(f"skill-{i:02}" in small for i in range(40))
    assert "skill_view" in small
    config.write_text("skills:\n  listing_max_tokens: 0\n", encoding="utf-8")
    assert pb.build_skills_system_prompt() == full


def test_tiny_budget_preserves_catalogue_and_warns(caplog, tmp_path, monkeypatch):
    skills = _home(tmp_path, monkeypatch)
    for i in range(20):
        _write_skill(skills, f"skill-{i:02}", "Keep this workflow.")
    (tmp_path / "config.yaml").write_text("skills:\n  listing_max_tokens: 1\n", encoding="utf-8")
    prompt = pb.build_skills_system_prompt()
    assert all(f"skill-{i:02}" in prompt for i in range(20))
    assert "names-only" in caplog.text and "budget" in caplog.text
