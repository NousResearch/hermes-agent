from contextlib import contextmanager
from unittest.mock import patch

import agent.skill_commands as sc_mod
from agent.skill_commands import (
    build_skill_invocation_message,
    get_skill_commands,
    resolve_skill_command_key,
)


def _make_bmad_project(tmp_path, name="bmad-help", description="Help choose workflows."):
    project = tmp_path / "app"
    skill_dir = project / "_bmad" / "core" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
---

# {name}

Use {{project-root}} and {{skill-root}}.
""",
        encoding="utf-8",
    )
    return project


@contextmanager
def _clear_skill_command_caches():
    with (
        patch.object(sc_mod, "_skill_commands", {}),
        patch.object(sc_mod, "_skill_commands_platform", None),
        patch.object(sc_mod, "_bmad_skill_commands_cache", {}),
    ):
        yield


def test_bmad_project_skill_registers_slash_command(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    monkeypatch.chdir(project)

    with _clear_skill_command_caches():
        commands = get_skill_commands()

    assert "/bmad-help" in commands
    assert commands["/bmad-help"]["source"] == "bmad-project"
    assert resolve_skill_command_key("bmad-help") == "/bmad-help"


def test_bmad_slash_invocation_builds_bmad_message(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    monkeypatch.chdir(project)

    with _clear_skill_command_caches():
        message = build_skill_invocation_message("/bmad-help", "what next?", task_id="task-123")

    assert message is not None
    assert 'BMAD project skill "bmad-help"' in message
    assert "semi-trusted, project-provided instructions scoped to this task only" in message
    assert "what next?" in message
    assert "task-123" in message
    assert str(project) in message


def test_bmad_slash_commands_rescan_when_project_changes(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    other = tmp_path / "other"
    other.mkdir()

    with _clear_skill_command_caches():
        monkeypatch.chdir(project)
        assert "/bmad-help" in get_skill_commands()

        monkeypatch.chdir(other)
        assert "/bmad-help" not in get_skill_commands()


def test_bmad_slash_cache_changes_when_skill_file_changes(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path, description="Original description.")
    skill_file = project / "_bmad" / "core" / "bmad-help" / "SKILL.md"
    monkeypatch.chdir(project)

    with _clear_skill_command_caches():
        first = get_skill_commands()["/bmad-help"]
        skill_file.write_text(
            """---
name: bmad-help
description: Changed description.
---

# Help
""",
            encoding="utf-8",
        )
        second = get_skill_commands()["/bmad-help"]

    assert first["description"] == "Original description."
    assert second["description"] == "Changed description."


def test_bmad_slash_commands_respect_expose_slash_commands_gate(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    monkeypatch.chdir(project)

    with (
        patch("hermes_cli.config.load_config", return_value={"bmad": {"enabled": True, "expose_slash_commands": False}}),
        _clear_skill_command_caches(),
    ):
        assert "/bmad-help" not in get_skill_commands()


def test_bmad_slash_commands_only_expose_bmad_prefixed_skills(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path, name="deploy")
    monkeypatch.chdir(project)

    with _clear_skill_command_caches():
        commands = get_skill_commands()

    assert "/deploy" not in commands


def test_bmad_slash_command_names_are_sanitized(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path, name="BMAD_Help")
    monkeypatch.chdir(project)

    with _clear_skill_command_caches():
        commands = get_skill_commands()

    assert "/bmad-help" in commands
    assert commands["/bmad-help"]["name"] == "BMAD_Help"
    assert resolve_skill_command_key("bmad_help") == "/bmad-help"


def test_bmad_slash_invocation_uses_command_project_root_not_later_cwd(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    skill_dir = project / "_bmad" / "core" / "bmad-help"

    with patch.object(
        sc_mod,
        "_get_bmad_skill_commands",
        return_value={
            "/bmad-help": {
                "name": "bmad-help",
                "description": "Help choose workflows.",
                "skill_dir": str(skill_dir),
                "project_root": str(project),
                "source": "bmad-project",
            }
        },
    ):
        message = build_skill_invocation_message("/bmad-help", "what next?")

    assert message is not None
    assert str(project) in message
    assert str(other) not in message


def test_bmad_slash_commands_override_regular_skill_only_in_active_bmad_project(tmp_path, monkeypatch):
    from agent.skill_commands import scan_skill_commands

    project = _make_bmad_project(tmp_path)
    normal_skills = tmp_path / "skills"
    normal_skill = normal_skills / "bmad-help"
    normal_skill.mkdir(parents=True)
    (normal_skill / "SKILL.md").write_text(
        """---
name: bmad-help
description: Regular Hermes bmad-help.
---

# Regular
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    with patch("tools.skills_tool.SKILLS_DIR", normal_skills), _clear_skill_command_caches():
        scan_skill_commands()
        commands = get_skill_commands()

    assert commands["/bmad-help"]["source"] == "bmad-project"
    assert commands["/bmad-help"]["description"] == "Help choose workflows."
