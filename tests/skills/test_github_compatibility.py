"""Persisted GitHub skill names resolve through the real runtime loader."""

import json
import shutil
from pathlib import Path

import pytest


LEGACY_NAMES = (
    "github-auth", "github-code-review", "github-issue-to-pr",
    "github-issues", "github-pr-workflow", "github-repo-management",
)


@pytest.fixture
def github_home(tmp_path, monkeypatch):
    from agent import skill_utils
    from tools import skills_tool

    home = tmp_path / "hermes"
    skills = home / "skills"
    shutil.copytree(
        Path(__file__).resolve().parents[2] / "skills/software-development/github",
        skills / "software-development/github",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills)
    monkeypatch.setattr(skill_utils, "get_project_skills_dirs", lambda: [])
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    return home


@pytest.mark.parametrize("name", LEGACY_NAMES)
@pytest.mark.parametrize("separator", ["", "/", ":"])
def test_persisted_names_load_the_consolidated_workflow(github_home, name, separator):
    from agent.skill_commands import build_preloaded_skills_prompt
    from cron.scheduler_prompt import _load_cron_skill_parts
    from tools.skills_tool import skill_view

    identifier = f"github{separator}{name}" if separator else name
    # Round-trip a stored task payload before dispatch, rather than just checking a map.
    task_file = github_home / "saved-task.json"
    task_file.write_text(json.dumps({"id": "fixture", "skills": [identifier]}))
    task = json.loads(task_file.read_text())
    loaded = json.loads(skill_view(task["skills"][0], preprocess=False))
    assert loaded["success"], loaded
    assert loaded["name"] == "github"
    assert Path(loaded["skill_dir"]) == github_home / "skills/software-development/github"
    for reference in loaded["linked_files"]["references"]:
        resource = json.loads(skill_view(identifier, file_path=reference, preprocess=False))
        assert resource["success"], resource
        assert "skills/github/github-auth/" not in resource["content"]

    prompt, names, missing = build_preloaded_skills_prompt(task["skills"])
    assert names == ["github"]
    assert missing == []
    assert "references/pr-workflow.md" in prompt
    assert "references/pr-workflow.md" in "\n".join(_load_cron_skill_parts(task, task["skills"]))


@pytest.mark.parametrize("disabled", ["github", "github-auth"])
def test_migration_respects_disabled_and_existing_custom_skills(github_home, disabled):
    from tools.skills_tool import skill_view

    config = github_home / "config.yaml"
    config.write_text(f"skills:\n  disabled: [{disabled}]\n")
    result = json.loads(skill_view("github-auth", preprocess=False))
    assert not result["success"]
    assert "disabled" in result["error"].lower()

    config.write_text("skills:\n  disabled: []\n")
    custom = github_home / "skills/custom/github-auth"
    custom.mkdir(parents=True)
    (custom / "SKILL.md").write_text(
        "---\nname: github-auth\ndescription: Custom auth.\n---\nLocal workflow.\n"
    )
    result = json.loads(skill_view("github-auth", preprocess=False))
    assert result["success"], result
    assert result["name"] == "github-auth"
    assert Path(result["skill_dir"]) == custom
