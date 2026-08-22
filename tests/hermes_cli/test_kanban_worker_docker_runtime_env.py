"""Dispatcher → worker regression coverage for #91568."""
from __future__ import annotations

import subprocess

import pytest

from hermes_cli.kanban_runtime import (
    KANBAN_TERMINAL_RUNTIME_ENV,
    decode_kanban_terminal_runtime,
)


def _make_task(kb, *, assignee: str = "w", workspace_kind: str = "dir"):
    return kb.Task(
        id="t_runtime",
        title="docker runtime",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind=workspace_kind,
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=1,
    )


def test_default_spawn_emits_task_scoped_runtime(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        kb,
        "_workspace_mount_authority_roots",
        lambda task, workspace, board=None: [str(tmp_path.resolve())],
    )
    kb._default_spawn(_make_task(kb), str(workspace))

    env = captured["env"]
    runtime = decode_kanban_terminal_runtime(
        env[KANBAN_TERMINAL_RUNTIME_ENV],
        expected_task_id="t_runtime",
        expected_workspace=workspace,
    )
    assert runtime["mounts"][0]["target"] == "/workspace"
    assert runtime["authorized_roots"] == [str(tmp_path.resolve())]
    assert env["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert env["TERMINAL_CWD"] == str(workspace)



def test_mount_authority_uses_exact_generated_scratch(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb

    root = tmp_path / "workspaces"
    workspace = root / "t_runtime"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(kb, "workspaces_root", lambda board=None: root)
    monkeypatch.setattr(kb, "read_board_metadata", lambda board: {})
    task = _make_task(kb, workspace_kind="scratch")

    assert kb._workspace_mount_authority_roots(
        task, str(workspace), board="default"
    ) == [str(workspace.resolve())]


def test_mount_authority_rejects_generated_scratch_symlink_escape(
    monkeypatch, tmp_path
):
    from hermes_cli import kanban_db as kb

    root = tmp_path / "workspaces"
    outside = tmp_path / "opt-data"
    root.mkdir()
    outside.mkdir()
    linked = root / "t_runtime"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not available on this platform")

    monkeypatch.setattr(kb, "workspaces_root", lambda board=None: root)
    monkeypatch.setattr(kb, "read_board_metadata", lambda board: {})
    task = _make_task(kb, workspace_kind="scratch")

    assert kb._workspace_mount_authority_roots(
        task, str(linked), board="default"
    ) == []


def test_mount_authority_prefers_project_folders_over_broader_board_default(
    monkeypatch, tmp_path
):
    import contextlib
    from types import SimpleNamespace

    from hermes_cli import kanban_db as kb
    from hermes_cli import projects_db as project_db

    project_root = tmp_path / "project"
    workspace = project_root / "task"
    project_root.mkdir()
    workspace.mkdir()
    task = _make_task(kb, workspace_kind="dir")
    task.project_id = "p_test"

    monkeypatch.setattr(
        kb,
        "read_board_metadata",
        lambda board: {"default_workdir": str(tmp_path)},
    )

    @contextlib.contextmanager
    def fake_connect_closing():
        yield object()

    monkeypatch.setattr(project_db, "connect_closing", fake_connect_closing)
    monkeypatch.setattr(
        project_db,
        "get_project",
        lambda conn, ident: SimpleNamespace(
            primary_path=str(project_root),
            folders=[SimpleNamespace(path=str(project_root))],
        ),
    )

    assert kb._workspace_mount_authority_roots(
        task, str(workspace), board="default"
    ) == [str(project_root.resolve())]
