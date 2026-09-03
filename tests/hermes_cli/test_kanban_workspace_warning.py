from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

from hermes_cli import kanban


def _create_args(**overrides):
    values = {
        "workspace": "scratch",
        "branch": None,
        "max_runtime": None,
        "max_retries": None,
        "title": "Explicit workspace",
        "body": None,
        "assignee": None,
        "created_by": "test",
        "project": "project-id",
        "tenant": None,
        "priority": 0,
        "parent": [],
        "triage": False,
        "idempotency_key": None,
        "skills": [],
        "model_override": None,
        "provider_override": None,
        "goal_mode": False,
        "goal_max_turns": None,
        "initial_status": "running",
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cli_prints_explicit_workspace_supersession_to_stderr(monkeypatch, capsys):
    task = SimpleNamespace(
        workspace_kind="worktree",
        workspace_path="/repo/.worktrees/t_123",
        project_id="project-id",
        status="todo",
        assignee=None,
    )
    captured_kwargs = {}

    def create_task(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "t_123"

    monkeypatch.setattr(kanban.kb, "connect_closing", lambda: nullcontext(object()))
    monkeypatch.setattr(kanban.kb, "create_task", create_task)
    monkeypatch.setattr(kanban.kb, "get_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(
        kanban.kb, "get_created_requested_workspace", lambda *args: "scratch"
    )

    assert kanban._cmd_create(_create_args()) == 0

    captured = capsys.readouterr()
    assert captured_kwargs["requested_workspace"] == "scratch"
    assert "requested workspace 'scratch'" in captured.err
    assert "project-linked workspace 'worktree:/repo/.worktrees/t_123'" in captured.err


def test_cli_omitted_workspace_inherits_silently(monkeypatch, capsys):
    task = SimpleNamespace(
        workspace_kind="worktree",
        workspace_path="/repo/.worktrees/t_123",
        project_id="project-id",
        status="todo",
        assignee=None,
    )
    captured_kwargs = {}

    def create_task(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "t_123"

    monkeypatch.setattr(kanban.kb, "connect_closing", lambda: nullcontext(object()))
    monkeypatch.setattr(kanban.kb, "create_task", create_task)
    monkeypatch.setattr(kanban.kb, "get_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(
        kanban.kb, "get_created_requested_workspace", lambda *args: None
    )

    assert kanban._cmd_create(_create_args(workspace=None)) == 0

    assert capsys.readouterr().err == ""
    assert captured_kwargs["requested_workspace"] is None


def test_cli_retry_warning_uses_immutable_created_request(monkeypatch, capsys):
    task = SimpleNamespace(
        workspace_kind="worktree",
        workspace_path="/repo/.worktrees/t_123",
        project_id="project-id",
        status="todo",
        assignee=None,
    )
    monkeypatch.setattr(kanban.kb, "connect_closing", lambda: nullcontext(object()))
    monkeypatch.setattr(kanban.kb, "create_task", lambda *args, **kwargs: "t_123")
    monkeypatch.setattr(kanban.kb, "get_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(
        kanban.kb, "get_created_requested_workspace", lambda *args: "scratch"
    )

    assert kanban._cmd_create(_create_args(workspace=None)) == 0

    assert "requested workspace 'scratch'" in capsys.readouterr().err


def test_cli_retry_does_not_invent_warning_from_current_request(monkeypatch, capsys):
    task = SimpleNamespace(
        workspace_kind="worktree",
        workspace_path="/repo/.worktrees/t_123",
        project_id="project-id",
        status="todo",
        assignee=None,
    )
    monkeypatch.setattr(kanban.kb, "connect_closing", lambda: nullcontext(object()))
    monkeypatch.setattr(kanban.kb, "create_task", lambda *args, **kwargs: "t_123")
    monkeypatch.setattr(kanban.kb, "get_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(
        kanban.kb, "get_created_requested_workspace", lambda *args: None
    )

    assert kanban._cmd_create(_create_args(workspace="scratch")) == 0

    assert capsys.readouterr().err == ""
