"""External ACP Kanban executor contracts."""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_conn(tmp_path):
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def test_project_executor_is_snapshotted_onto_linked_task(monkeypatch, kanban_conn, tmp_path):
    """Changing a project later must not reroute an already-created task."""
    projects_path = tmp_path / "projects.db"
    monkeypatch.setattr(pdb, "projects_db_path", lambda: projects_path)
    with pdb.connect_closing() as projects:
        project_id = pdb.create_project(
            projects,
            name="External harness",
            folders=[str(tmp_path / "repo")],
            executor="claude-code",
        )
        project = pdb.get_project(projects, project_id)
        assert project.executor == "claude-code"

    task_id = kb.create_task(kanban_conn, title="Use native ACP", project_id=project_id)
    assert kb.get_task(kanban_conn, task_id).executor == "claude-code"

    with pdb.connect_closing() as projects:
        assert pdb.update_project(projects, project_id, executor="codex")
    assert kb.get_task(kanban_conn, task_id).executor == "claude-code"


def test_external_executor_spawn_uses_dedicated_worker(monkeypatch, tmp_path):
    """Claude tasks must never enter the normal Hermes chat/provider path."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "external").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    task = kb.Task(
        id="t_acp_spawn",
        title="external",
        body=None,
        assignee="external",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="test-lock",
        claim_expires=None,
        tenant=None,
        current_run_id=9,
        executor="claude-code",
    )
    captured = {}

    class FakeProc:
        pid = 1234

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert kb._default_spawn(task, str(workspace)) == 1234
    assert captured["cmd"] == [kb.sys.executable, "-m", "agent.acp_task_executor"]
    assert captured["env"]["HERMES_KANBAN_EXECUTOR"] == "claude-code"
    assert captured["env"]["HERMES_KANBAN_RUN_ID"] == "9"


def test_acp_worker_completes_claimed_task_with_single_session(monkeypatch, kanban_conn, tmp_path):
    """The task worker owns one adapter session and closes the claimed run."""
    from agent import acp_task_executor as executor

    task_id = kb.create_task(kanban_conn, title="External task", assignee="external")
    claimed = kb.claim_task(kanban_conn, task_id, claimer="test-lock")
    assert claimed is not None
    monkeypatch.setattr(kb, "connect_closing", lambda *, board=None: _connection_context(kanban_conn))

    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def _run_prompt(self, prompt, *, timeout_seconds):
            calls.append((prompt, timeout_seconds))
            return "Implemented and tested.", ""

    monkeypatch.setattr(executor, "CopilotACPClient", FakeClient)
    monkeypatch.setattr(executor, "command_for", lambda name: ("fake-acp", ["--stdio"]))

    assert executor.run_task(
        executor="claude-code", task_id=task_id, workspace=str(tmp_path), board="test"
    ) == "Implemented and tested."
    task = kb.get_task(kanban_conn, task_id)
    assert task.status == "done"
    assert len(calls) == 2
    assert calls[0]["acp_command"] == "fake-acp"
    assert "# Kanban task" in calls[1][0]


class _connection_context:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        return False
