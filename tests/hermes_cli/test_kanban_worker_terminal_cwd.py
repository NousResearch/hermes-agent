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

#70688: dispatcher-spawned workers also inherit a deployment-wide
``HERMES_WRITE_SAFE_ROOT``, letting sibling tasks write each other's workspaces.
The spawn path replaces inherited roots with a normalized task workspace and
preserves that scope through dotenv reload and delegated children.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest import mock

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

    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(subprocess, "Popen", fake_popen)
        kbd._default_spawn(_make_task(kb), workspace)
    return captured


def _shell_file_operations(workspace: Path):
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    environment = LocalEnvironment(cwd=str(workspace))
    return ShellFileOperations(environment, cwd=str(workspace))


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


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
    child_root = os.path.realpath(workspace)

    assert captured["env"]["TERMINAL_CWD"] == child_root
    assert captured["env"]["HERMES_WRITE_SAFE_ROOT"] == child_root
    # The subprocess cwd and TERMINAL_CWD must agree — both anchor the workspace.
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)


def test_worker_lineage_marker_isolated_from_test_process():
    assert os.environ.get("HERMES_KANBAN_SAFE_ROOT_ACTIVE") is None


def test_narrow_inherited_root_replaced_for_scratch_workspace_write(
    monkeypatch, tmp_path
):
    root = tmp_path / "board"
    workspace = root / "scratch-a"
    sibling = root / "scratch-b"
    inherited = tmp_path / "deployment-root"
    workspace.mkdir(parents=True)
    sibling.mkdir()
    inherited.mkdir()
    (tmp_path / ".hermes" / "profiles" / "w").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_CWD", str(inherited))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(inherited))

    from hermes_cli import kanban_db as kb

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    child_root = os.path.realpath(workspace)
    assert captured["env"]["TERMINAL_CWD"] == child_root

    target = workspace / "own.txt"
    with mock.patch.dict(os.environ, captured["env"], clear=True):
        result = _shell_file_operations(workspace).write_file(str(target), "own")

    assert result.error is None
    assert captured["env"]["HERMES_WRITE_SAFE_ROOT"] == child_root
    assert target.read_text(encoding="utf-8") == "own"


def test_task_safe_root_survives_profile_dotenv_override(monkeypatch, tmp_path):
    root = tmp_path / "board"
    workspace = root / "scratch-a"
    inherited = tmp_path / "deployment-root"
    workspace.mkdir(parents=True)
    inherited.mkdir()
    profile_home = tmp_path / ".hermes" / "profiles" / "w"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text(
        f"HERMES_WRITE_SAFE_ROOT={inherited}\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(inherited))

    from hermes_cli import kanban_db as kb
    from hermes_cli.env_loader import load_hermes_dotenv

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    child_root = os.path.realpath(workspace)
    with mock.patch.dict(os.environ, captured["env"], clear=True):
        load_hermes_dotenv(hermes_home=captured["env"]["HERMES_HOME"])
        target = workspace / "dotenv-own.txt"
        result = _shell_file_operations(workspace).write_file(str(target), "own")

    assert result.error is None
    assert captured["env"]["HERMES_WRITE_SAFE_ROOT"] == child_root
    assert target.read_text(encoding="utf-8") == "own"


def test_delegated_worker_safe_root_survives_profile_dotenv_override(
    monkeypatch, tmp_path
):
    root = tmp_path / "board"
    workspace = root / "scratch-a"
    inherited = tmp_path / "deployment-root"
    workspace.mkdir(parents=True)
    inherited.mkdir()
    profile_home = tmp_path / ".hermes" / "profiles" / "w"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text(
        f"HERMES_WRITE_SAFE_ROOT={inherited}\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(inherited))

    from agent.delegation_context import scrub_kanban_env
    from hermes_cli import kanban_db as kb
    from hermes_cli.env_loader import load_hermes_dotenv

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    delegated_env = scrub_kanban_env(captured["env"])
    child_root = os.path.realpath(workspace)
    assert "HERMES_KANBAN_TASK" not in delegated_env
    assert delegated_env["HERMES_KANBAN_SAFE_ROOT_ACTIVE"] == "1"
    with mock.patch.dict(os.environ, delegated_env, clear=True):
        load_hermes_dotenv(hermes_home=delegated_env["HERMES_HOME"])
        target = workspace / "delegated-own.txt"
        result = _shell_file_operations(workspace).write_file(str(target), "own")

    assert result.error is None
    assert target.read_text(encoding="utf-8") == "own"
    assert delegated_env["HERMES_WRITE_SAFE_ROOT"] == child_root


def test_sibling_and_traversal_mutations_are_denied(monkeypatch, tmp_path):
    root = tmp_path / "board"
    workspace = root / "scratch-a"
    sibling = root / "scratch-b"
    workspace.mkdir(parents=True)
    sibling.mkdir()
    (tmp_path / ".hermes" / "profiles" / "w").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_CWD", str(root))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(root))

    from hermes_cli import kanban_db as kb

    direct = sibling / "direct.txt"
    traversal = sibling / "traversal.txt"
    sibling_write = sibling / "sibling-write.txt"
    sibling_delete = sibling / "sibling-delete.txt"
    own_move = workspace / "own-move.txt"
    moved = sibling / "moved.txt"
    direct.write_text("before", encoding="utf-8")
    traversal.write_text("before", encoding="utf-8")
    sibling_delete.write_text("before", encoding="utf-8")
    own_move.write_text("before", encoding="utf-8")
    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    with mock.patch.dict(os.environ, captured["env"], clear=True):
        ops = _shell_file_operations(workspace)
        write_result = ops.write_file(str(sibling_write), "blocked")
        direct_result = ops.patch_replace(str(direct), "before", "after")
        traversal_result = ops.patch_replace(
            str(workspace / ".." / "scratch-b" / "traversal.txt"),
            "before",
            "after",
        )
        delete_result = ops.delete_file(str(sibling_delete))
        move_result = ops.move_file(str(own_move), str(moved))

    assert write_result.error is not None
    assert "outside HERMES_WRITE_SAFE_ROOT" in write_result.error
    assert direct_result.error is not None
    assert "outside HERMES_WRITE_SAFE_ROOT" in direct_result.error
    assert traversal_result.error is not None
    assert "outside HERMES_WRITE_SAFE_ROOT" in traversal_result.error
    assert delete_result.error is not None
    assert "outside HERMES_WRITE_SAFE_ROOT" in delete_result.error
    assert move_result.error is not None
    assert "outside HERMES_WRITE_SAFE_ROOT" in move_result.error
    assert not sibling_write.exists()
    assert direct.read_text(encoding="utf-8") == "before"
    assert traversal.read_text(encoding="utf-8") == "before"
    assert sibling_delete.read_text(encoding="utf-8") == "before"
    assert own_move.read_text(encoding="utf-8") == "before"
    assert not moved.exists()


def test_symlink_escape_is_denied(monkeypatch, tmp_path):
    if os.name == "nt":
        pytest.skip("symlink boundary is covered on POSIX CI")

    root = tmp_path / "board"
    workspace = root / "scratch-a"
    outside = root / "outside"
    workspace.mkdir(parents=True)
    outside.mkdir()
    try:
        (workspace / "escape").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    (tmp_path / ".hermes" / "profiles" / "w").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_CWD", str(root))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(root))

    from hermes_cli import kanban_db as kb

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    target = workspace / "escape" / "escaped.txt"
    with mock.patch.dict(os.environ, captured["env"], clear=True):
        result = _shell_file_operations(workspace).write_file(str(target), "blocked")

    assert result.error is not None
    assert "outside HERMES_WRITE_SAFE_ROOT" in result.error
    assert not (outside / "escaped.txt").exists()


def test_workspace_variables_share_realpath(monkeypatch, tmp_path):
    target = tmp_path / "real-workspace"
    workspace = tmp_path / "workspace-link"
    target.mkdir()
    try:
        workspace.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    inherited = tmp_path / "inherited-root"
    inherited.mkdir()
    (tmp_path / ".hermes" / "profiles" / "w").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_CWD", str(inherited))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(inherited))

    from hermes_cli import kanban_db as kb

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert captured["env"]["TERMINAL_CWD"] == os.path.realpath(workspace)
    assert captured["env"]["HERMES_WRITE_SAFE_ROOT"] == os.path.realpath(workspace)


def test_invalid_workspace_preserves_inherited_policy(monkeypatch, tmp_path):
    inherited_cwd = tmp_path / "inherited-cwd"
    inherited_root = tmp_path / "inherited-root"
    existing_file = tmp_path / "workspace-file"
    separator_dir = tmp_path / f"workspace{os.pathsep}roots"
    inherited_cwd.mkdir()
    inherited_root.mkdir()
    existing_file.write_text("file", encoding="utf-8")
    separator_dir.mkdir()
    (tmp_path / ".hermes" / "profiles" / "w").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_CWD", str(inherited_cwd))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(inherited_root))

    candidates = [
        "relative-workspace",
        "",
        str(tmp_path / "missing-workspace"),
        str(existing_file),
        os.path.abspath(os.sep),
        str(separator_dir),
    ]
    from hermes_cli import kanban_db as kb

    for candidate in candidates:
        captured = _capture_spawn_env(kb, monkeypatch, candidate)
        assert captured["env"]["TERMINAL_CWD"] == str(inherited_cwd), candidate
        assert captured["env"]["HERMES_WRITE_SAFE_ROOT"] == str(inherited_root), candidate


def test_dispatch_scopes_materialized_scratch_workspace(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    inherited = kanban_home.parent / "deployment-root"
    inherited.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(inherited))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(inherited))
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        captured["cwd"] = kwargs["cwd"]
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="scoped scratch",
            assignee="default",
            workspace_kind="scratch",
        )
        result = kb.dispatch_once(conn, max_spawn=1)
        task = kb.get_task(conn, task_id)

    assert [item[0] for item in result.spawned] == [task_id]
    assert task is not None
    workspace = Path(task.workspace_path)
    assert workspace.is_dir()
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_WRITE_SAFE_ROOT"] == os.path.realpath(workspace)


def test_scoped_spawn_failure_records_existing_failure_flow(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    def failing_popen(*args, **kwargs):
        raise OSError("spawn boom")

    monkeypatch.setattr(subprocess, "Popen", failing_popen)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="spawn failure",
            assignee="default",
            workspace_kind="scratch",
        )
        first = kb.dispatch_once(conn, failure_limit=2)
        after_first = kb.get_task(conn, task_id)
        second = kb.dispatch_once(conn, failure_limit=2)
        after_second = kb.get_task(conn, task_id)
        event_kinds = [event.kind for event in kb.list_events(conn, task_id)]

    assert first.spawned == []
    assert second.spawned == []
    assert after_first is not None
    assert after_first.status == "ready"
    assert after_first.claim_lock is None
    assert after_first.consecutive_failures == 1
    assert after_second is not None
    assert after_second.status == "blocked"
    assert after_second.claim_lock is None
    assert after_second.consecutive_failures == 2
    assert "spawn_failed" in event_kinds
    assert "gave_up" in event_kinds
