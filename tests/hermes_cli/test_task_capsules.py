from pathlib import Path

import pytest

from hermes_cli.task_capsules import (
    build_capsule,
    default_output_path,
    discover_relevant_files,
    discover_test_commands,
    slugify,
    write_capsule,
)


def test_slugify_is_path_safe():
    assert slugify("Add Atlas task capsules!") == "add-atlas-task-capsules"


def test_build_capsule_includes_required_sections_and_stays_under_budget(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    (hermes_home / "MEMORY.md").write_text(
        "Atlas coding handoffs should avoid dumping huge context.\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("Run tests before handoff.\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (repo / "task_capsules.py").write_text(
        "def build_task_capsule():\n    pass\n", encoding="utf-8"
    )

    markdown = build_capsule(
        title="Generate Atlas task capsules",
        repo_path=repo,
        goal="Create a markdown handoff capsule command.",
        acceptance=["Capsule includes goal, constraints, files, tests, acceptance."],
        word_budget=300,
    )

    assert "## Goal" in markdown
    assert "## Constraints" in markdown
    assert "## Relevant files" in markdown
    assert "## Commands to run" in markdown
    assert "## Acceptance criteria" in markdown
    assert "task_capsules.py" in markdown
    assert "python -m pytest -q" in markdown
    assert len(markdown.split()) <= 300


def test_relevant_files_skips_dependency_directories(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "capsule.js").write_text("capsule", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "capsule.py").write_text("capsule", encoding="utf-8")

    files = discover_relevant_files(repo, "capsule", max_files=5)
    rels = {str(item.path.relative_to(repo)) for item in files}

    assert "src/capsule.py" in rels
    assert "node_modules/capsule.js" not in rels


def test_discover_test_commands_detects_common_projects(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (repo / "package.json").write_text('{"scripts":{"test":"vitest"}}', encoding="utf-8")

    commands = discover_test_commands(repo)

    assert "python -m pytest -q" in commands
    assert "npm test" in commands


def test_default_output_uses_hermes_home_and_write_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    output = default_output_path("My Task")
    assert output == tmp_path / "task-capsules" / "my-task.md"

    written = write_capsule("# hi\n", output)
    assert written == output.resolve()
    assert output.read_text(encoding="utf-8") == "# hi\n"
    with pytest.raises(FileExistsError):
        write_capsule("# bye\n", output)
