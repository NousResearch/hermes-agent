from pathlib import Path

from hermes_android.bootstrap import bootstrap_android_runtime


def test_bootstrap_android_runtime_syncs_bundled_skills_and_sets_optional_env(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    bundled_skill = repo_root / "skills" / "android" / "sample-skill"
    bundled_skill.mkdir(parents=True)
    (bundled_skill / "SKILL.md").write_text("# sample\n", encoding="utf-8")

    optional_skill = repo_root / "optional-skills" / "experimental" / "optional-skill"
    optional_skill.mkdir(parents=True)
    (optional_skill / "SKILL.md").write_text("# optional\n", encoding="utf-8")

    monkeypatch.setattr("hermes_android.bundled_assets.repo_root", lambda: repo_root)

    result = bootstrap_android_runtime(str(tmp_path / "files"), api_server_port=8877, api_server_key="bootstrap-key")

    copied_skill = Path(result["runtime"]["hermes_home"]) / "skills" / "android" / "sample-skill" / "SKILL.md"
    assert copied_skill.exists()
    assert copied_skill.read_text(encoding="utf-8") == "# sample\n"
    assert result["skill_env"]["HERMES_OPTIONAL_SKILLS"] == str(repo_root / "optional-skills")
    assert Path(result["skill_env"]["HERMES_OPTIONAL_SKILLS"]).exists()
