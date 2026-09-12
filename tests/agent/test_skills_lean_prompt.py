"""Opt-in lean skill index: retain discovery, isolate profiles and caches."""
import pytest

from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache


def seed(home, mode=None):
    home.mkdir(parents=True, exist_ok=True)
    if mode is not None:
        (home / "config.yaml").write_text(f"skills:\n  prompt_mode: {mode}\n")
    for name in ("alpha", "beta"):
        skill = home / "skills" / "demo" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Detailed instructions for {name}\n---\nBody.\n"
        )


def test_lean_config_keeps_names_without_full_descriptions(tmp_path, monkeypatch):
    seed(tmp_path, "lean")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    clear_skills_system_prompt_cache(clear_snapshot=True)
    prompt = build_skills_system_prompt()
    assert "alpha" in prompt and "beta" in prompt
    assert "Detailed instructions" not in prompt
    assert "even partially relevant" not in prompt
    assert "skill_view" in prompt
    assert "skills_list" in prompt


@pytest.mark.parametrize("mode", [None, "full", "unknown", "true", "[]", "{}"])
def test_default_and_invalid_modes_preserve_full_policy(tmp_path, monkeypatch, mode):
    seed(tmp_path, mode)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    clear_skills_system_prompt_cache(clear_snapshot=True)
    prompt = build_skills_system_prompt()
    assert "Detailed instructions for alpha" in prompt
    assert "even partially relevant" in prompt


def test_profile_override_does_not_inherit_lean_policy(tmp_path, monkeypatch):
    root, other = tmp_path / "default", tmp_path / "other"
    seed(root, "lean")
    seed(other)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.chdir(tmp_path)
    clear_skills_system_prompt_cache(clear_snapshot=True)
    lean = build_skills_system_prompt()
    full = build_skills_system_prompt(skills_dir_override=other / "skills")
    assert "Detailed instructions" not in lean
    assert "Detailed instructions" in full
    assert build_skills_system_prompt() == lean


def test_modes_have_separate_index_cache_entries(tmp_path, monkeypatch):
    seed(tmp_path, "lean")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    clear_skills_system_prompt_cache(clear_snapshot=True)
    lean = build_skills_system_prompt()
    (tmp_path / "config.yaml").write_text("skills:\n  prompt_mode: full\n")
    full = build_skills_system_prompt()
    assert "Detailed instructions" not in lean
    assert "Detailed instructions" in full


def test_lean_preserves_collisions_and_no_dangling_discovery_tool():
    from agent.prompt_builder import _render_skills_index

    categories = {
        "demo": [("alpha", "Ordinary description")],
        "org:test": [("beta", "[name collision — also exists personally; load via category path]")],
    }
    prompt = _render_skills_index(categories, {}, None, {"skill_view"}, lean=True)
    assert "alpha" in prompt and "beta" in prompt
    assert "name collision" in prompt
    assert "load via category path" in prompt
    assert "Ordinary description" not in prompt
    assert "skills_list" not in prompt


def test_disabled_skills_stay_disabled_in_lean_mode(tmp_path, monkeypatch):
    seed(tmp_path, "lean")
    (tmp_path / "config.yaml").write_text("skills:\n  prompt_mode: lean\n  disabled: [beta]\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    clear_skills_system_prompt_cache(clear_snapshot=True)
    prompt = build_skills_system_prompt()
    assert "alpha" in prompt
    assert "beta" not in prompt
