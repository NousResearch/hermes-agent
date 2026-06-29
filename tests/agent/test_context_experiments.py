"""Tests for session-scoped context experiments."""

from pathlib import Path

from agent.prompt_builder import build_context_files_prompt


def _experiment_config(old_file: Path, new_file: Path, *, new_skills=None):
    return {
        "context_experiments": {
            "agentsmd-split": {
                "enabled": True,
                "assignment": "round_robin",
                "arms": {
                    "old": {"context_file": str(old_file)},
                    "new": {
                        "context_file": str(new_file),
                        "skills": list(new_skills or []),
                    },
                },
            }
        }
    }


def test_context_experiment_round_robin_replaces_default_project_context(
    monkeypatch, tmp_path
):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / "AGENTS.md").write_text("DEFAULT AGENTS", encoding="utf-8")
    old_file = tmp_path / "AGENTS.old.md"
    new_file = tmp_path / "AGENTS.slim.md"
    old_file.write_text("OLD RULES", encoding="utf-8")
    new_file.write_text("NEW RULES", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _experiment_config(old_file, new_file),
    )

    first = build_context_files_prompt(cwd=str(tmp_path), session_id="session-a")
    second = build_context_files_prompt(cwd=str(tmp_path), session_id="session-b")
    first_again = build_context_files_prompt(cwd=str(tmp_path), session_id="session-a")

    assert "OLD RULES" in first
    assert "NEW RULES" not in first
    assert "NEW RULES" in second
    assert "OLD RULES" not in second
    assert first_again == first
    assert "DEFAULT AGENTS" not in first + second


def test_context_experiment_arm_can_preload_skills(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    old_file = tmp_path / "AGENTS.old.md"
    new_file = tmp_path / "AGENTS.slim.md"
    old_file.write_text("OLD RULES", encoding="utf-8")
    new_file.write_text("NEW RULES", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _experiment_config(old_file, new_file, new_skills=["eval-skill"]),
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_preloaded_skills_prompt",
        lambda skills, task_id=None: (
            "[loaded skill body]",
            ["eval-skill"],
            [],
        ),
    )

    # First new session receives old; second receives new + skill preload.
    build_context_files_prompt(cwd=str(tmp_path), session_id="session-a")
    prompt = build_context_files_prompt(cwd=str(tmp_path), session_id="session-b")

    assert "NEW RULES" in prompt
    assert "[loaded skill body]" in prompt
