"""Tests: kanban worker spawn pins TERMINAL_CWD to the task workspace.

Regression coverage for #34619 and #41312 (same root cause): ``_default_spawn``
launched the worker subprocess with ``cwd=workspace`` and set
``HERMES_KANBAN_WORKSPACE``, but did NOT set ``TERMINAL_CWD``. Because
``TERMINAL_CWD`` takes precedence over the process cwd in both
``tools/file_tools.py::_resolve_base_dir`` (relative ``write_file`` paths) and
``agent_init``'s context-file loader (``AGENTS.md`` discovery), workers inherited
the dispatching gateway's cwd — relative writes landed in the gateway user's
home (#41312) and the wrong profile's ``AGENTS.md`` was loaded (#34619).
Pinning ``TERMINAL_CWD`` to the workspace fixes both.
"""

from __future__ import annotations

import subprocess
import os
import sys

import pytest


def _make_task(kb, *, assignee: str = "w"):
    return kb.Task(
        id="t_cwd",
        title="cwd pin",
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
        current_run_id=1,
    )


def _capture_spawn_env(kb, monkeypatch, workspace: str) -> dict:
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kbd._default_spawn(_make_task(kb), workspace)
    return captured


def test_terminal_cwd_pinned_to_workspace(monkeypatch, tmp_path):
    """A real, absolute workspace dir is pinned as TERMINAL_CWD."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == str(workspace)
    # The subprocess cwd and TERMINAL_CWD must agree — both anchor the workspace.
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)


def test_worker_write_safe_root_is_exact_assigned_workspace(monkeypatch, tmp_path):
    """A worker must not inherit a broader generic file-tool write scope."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["HERMES_WRITE_SAFE_ROOT"] == str(workspace)


def test_worker_spawn_rejects_malformed_profile_config(monkeypatch, tmp_path):
    """A broken profile must fail once before a provider-less child is spawned."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        "model:\n  provider: openai-codex\nagent:\n  coding_instructions: bad: scalar\n",
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    def forbidden_popen(*args, **kwargs):
        raise AssertionError("malformed profile reached subprocess spawn")

    monkeypatch.setattr(subprocess, "Popen", forbidden_popen)

    with pytest.raises(ValueError, match="invalid profile config"):
        kbd._default_spawn(_make_task(kb), str(workspace))

def test_worker_path_prefers_workspace_venv_for_terminal_commands(monkeypatch, tmp_path):
    """A worker's literal ``python`` commands use its project interpreter."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    (workspace / ".venv" / "bin").mkdir(parents=True)
    python = workspace / ".venv" / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["PATH"].split(":")[:3] == [
        str(workspace / ".venv" / "bin"),
        "/usr/bin",
        "/bin",
    ]


@pytest.mark.macos_only
def test_protected_control_command_executes_in_spawned_environment(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from tools.kanban_tools import _sanitize_remote_worker_payload

    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    workspace = tmp_path / "project"
    workspace.mkdir()
    python = workspace / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    real_popen = subprocess.Popen
    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    command = _sanitize_remote_worker_payload(
        f'env HERMES_HOME={root} {sys.executable} --version',
        workspace_path=str(workspace), control_home=str(root),
        worker_python=str(python), dispatcher_python=sys.executable,
    )
    # Use the actual terminal environment factory and shell expansion, not a
    # string-only assertion that misses an absent interpreter token (exit 127).
    from tools.environments.local import build_subprocess_env
    monkeypatch.setattr(subprocess, "Popen", real_popen)
    result = subprocess.run(
        ["/bin/sh", "-c", command], cwd=workspace,
        env=build_subprocess_env(captured["env"]),
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Python ")
    assert captured["env"]["VIRTUAL_ENV"] == str(python.parent.parent)
    assert captured["env"]["HERMES_KANBAN_WORKTREE_PYTHON"] == str(python)
    assert captured["env"]["HERMES_KANBAN_HERMES_PYTHON"] == os.path.abspath(sys.executable)


def test_python_workspace_without_interpreter_never_spawns(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    with pytest.raises(RuntimeError, match="no executable"):
        _capture_spawn_env(kb, monkeypatch, str(workspace))
