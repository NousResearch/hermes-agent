"""Regression tests for backend-visible skill-directory rendering (#41541).

``${HERMES_SKILL_DIR}``, the ``[Skill directory: ...]`` header, and the
supporting-file/script hints must resolve to the path the agent can actually
reach on the active terminal backend.  On Docker/Singularity/Modal that is
``/root/.hermes/skills/<name>``; on SSH/Daytona it is
``<remote_home>/.hermes/skills/<name>``; on the local backend the host path is
preserved.  Before the fix all three surfaces emitted the raw HOST path, so
bundled skill scripts (e.g. ``${HERMES_SKILL_DIR}/scripts/todo``) were
unrunnable inside the sandbox.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_skill(hermes_home: Path) -> Path:
    """Create a local skill dir with a bundled script under HERMES_HOME."""
    skill_dir = hermes_home / "skills" / "todo-skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "todo").write_text("#!/usr/bin/env bash\n")
    return skill_dir


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


# ── map_skill_dir_for_backend ──────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["docker", "singularity", "modal"])
def test_container_backends_map_to_root_hermes(backend, hermes_home, monkeypatch):
    from agent import skill_path_mapping

    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", backend)
    # No live environment created yet — TERMINAL_ENV drives the container base.
    monkeypatch.setattr(skill_path_mapping, "_active_terminal_env", lambda task_id: None)

    mapped = skill_path_mapping.map_skill_dir_for_backend(skill_dir)

    assert mapped == "/root/.hermes/skills/todo-skill"
    # The bundled script is reachable under the container-visible path.
    assert f"{mapped}/scripts/todo" == "/root/.hermes/skills/todo-skill/scripts/todo"


@pytest.mark.parametrize("backend", ["ssh", "daytona"])
def test_remote_backends_map_to_remote_home(backend, hermes_home, monkeypatch):
    from agent import skill_path_mapping

    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", backend)
    env = SimpleNamespace(_remote_home="/home/remoteuser")
    monkeypatch.setattr(skill_path_mapping, "_active_terminal_env", lambda task_id: env)

    mapped = skill_path_mapping.map_skill_dir_for_backend(skill_dir)

    assert mapped == "/home/remoteuser/.hermes/skills/todo-skill"


def test_local_backend_preserves_host_path(hermes_home, monkeypatch):
    from agent import skill_path_mapping

    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(skill_path_mapping, "_active_terminal_env", lambda task_id: None)

    mapped = skill_path_mapping.map_skill_dir_for_backend(skill_dir)

    assert mapped == str(skill_dir)


def test_unknown_backend_falls_back_to_host_path(hermes_home, monkeypatch):
    from agent import skill_path_mapping

    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", "some-future-backend")
    monkeypatch.setattr(skill_path_mapping, "_active_terminal_env", lambda task_id: None)

    mapped = skill_path_mapping.map_skill_dir_for_backend(skill_dir)

    assert mapped == str(skill_dir)


def test_dir_outside_any_mount_falls_back_to_host_path(hermes_home, tmp_path, monkeypatch):
    from agent import skill_path_mapping

    # A skill dir that is NOT under HERMES_HOME/skills and not a registered
    # external dir must not be given a bogus container path.
    _make_skill(hermes_home)
    stray = tmp_path / "elsewhere" / "stray-skill"
    stray.mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(skill_path_mapping, "_active_terminal_env", lambda task_id: None)

    mapped = skill_path_mapping.map_skill_dir_for_backend(stray)

    assert mapped == str(stray)


# ── substitute_template_vars uses the mapped path ──────────────────────────


def test_template_var_substitution_uses_container_path(hermes_home, monkeypatch):
    from agent import skill_preprocessing

    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        "agent.skill_path_mapping._active_terminal_env", lambda task_id: None
    )

    out = skill_preprocessing.substitute_template_vars(
        "Run ${HERMES_SKILL_DIR}/scripts/todo", skill_dir, session_id=None
    )

    assert out == "Run /root/.hermes/skills/todo-skill/scripts/todo"


def test_template_var_substitution_local_keeps_host_path(hermes_home, monkeypatch):
    from agent import skill_preprocessing

    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "agent.skill_path_mapping._active_terminal_env", lambda task_id: None
    )

    out = skill_preprocessing.substitute_template_vars(
        "Run ${HERMES_SKILL_DIR}/scripts/todo", skill_dir, session_id=None
    )

    assert out == f"Run {skill_dir}/scripts/todo"


# ── _build_skill_message: all three surfaces agree (the core invariant) ─────


def _build_message(skill_dir: Path):
    from agent import skill_commands

    loaded_skill = {
        "content": "Use ${HERMES_SKILL_DIR}/scripts/todo to manage tasks.",
        "linked_files": {},
    }
    return skill_commands._build_skill_message(
        loaded_skill,
        skill_dir,
        activation_note="[activation]",
        session_id=None,
    )


def test_all_agent_visible_surfaces_agree_on_container_path(hermes_home, monkeypatch):
    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        "agent.skill_path_mapping._active_terminal_env", lambda task_id: None
    )

    msg = _build_message(skill_dir)

    container = "/root/.hermes/skills/todo-skill"
    # The host path must NOT leak into the prompt on a remote backend.
    assert str(skill_dir) not in msg
    # 1) ${HERMES_SKILL_DIR} substitution in the skill body.
    assert f"Use {container}/scripts/todo" in msg
    # 2) [Skill directory: ...] header.
    assert f"[Skill directory: {container}]" in msg
    # 3) supporting-file / script hint.
    assert f"{container}/scripts/todo" in msg
    assert f"node {container}/scripts/foo.js" in msg


def test_all_agent_visible_surfaces_preserve_host_path_locally(hermes_home, monkeypatch):
    skill_dir = _make_skill(hermes_home)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "agent.skill_path_mapping._active_terminal_env", lambda task_id: None
    )

    msg = _build_message(skill_dir)

    assert f"[Skill directory: {skill_dir}]" in msg
    assert f"{skill_dir}/scripts/todo" in msg
