"""Regression: HERMES_PROTECTED_GIT_ROOTS must be injected from the *project*
primary path even when the board has no default_workdir.

Found in supervisor final review of the G chain: ``import subprocess`` lived
inside the ``if _bd_workdir`` branch, so a board without default_workdir but a
task with a project primary_path raised NameError in the project branch, the
broad ``except`` swallowed it, and the guard was silently OFF.
"""
from __future__ import annotations

import subprocess

import pytest


def _make_task(kb, *, assignee: str, project_id: str | None):
    return kb.Task(
        id="t_pg_project",
        title="pg project",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        project_id=project_id,
    )


def _git_init(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path.resolve()


@pytest.fixture
def spawn_env(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "kamiyu"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("toolsets:\n  - hermes-cli\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}
    real_popen = subprocess.Popen

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        # Only intercept the hermes worker spawn; let `git rev-parse` run for real.
        if cmd and str(cmd[0]) == "hermes":
            captured["env"] = dict(kwargs.get("env") or {})
            return FakeProc()
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return kb, captured


def test_project_primary_path_is_protected_when_board_has_no_default_workdir(
    spawn_env, monkeypatch, tmp_path
):
    kb, captured = spawn_env
    repo = _git_init(tmp_path / "proj-repo")

    # Board metadata WITHOUT default_workdir.
    monkeypatch.setattr(kb, "read_board_metadata", lambda board=None: {})

    class _Proj:
        primary_path = str(repo)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import hermes_cli.projects_db as projects_mod

    monkeypatch.setattr(projects_mod, "connect_closing", lambda: _Conn())
    monkeypatch.setattr(projects_mod, "get_project", lambda conn, pid: _Proj())

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb._default_spawn(_make_task(kb, assignee="kamiyu", project_id="p_test"), str(workspace))

    roots = captured["env"].get("HERMES_PROTECTED_GIT_ROOTS", "")
    assert str(repo) in roots.split(__import__("os").pathsep), roots


def test_board_default_workdir_is_protected(spawn_env, monkeypatch, tmp_path):
    kb, captured = spawn_env
    repo = _git_init(tmp_path / "board-repo")
    monkeypatch.setattr(kb, "read_board_metadata", lambda board=None: {"default_workdir": str(repo)})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb._default_spawn(_make_task(kb, assignee="kamiyu", project_id=None), str(workspace))

    roots = captured["env"].get("HERMES_PROTECTED_GIT_ROOTS", "")
    assert str(repo) in roots.split(__import__("os").pathsep), roots
