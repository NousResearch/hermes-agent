from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext

from hermes_cli import kanban
from hermes_cli.kanban import _is_kanban_worker_cli_mutation


def test_running_kanban_worker_cannot_mutate_board_through_cli(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="complete")) is True
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="promote")) is True
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="set-model")) is True


def test_every_declared_mutating_action_is_denied_to_running_workers(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    for action in kanban._DELEGATED_CHILD_DENIED_ACTIONS:
        assert _is_kanban_worker_cli_mutation(Namespace(kanban_action=action)) is True


def test_worker_command_dispatch_blocks_set_model_before_handler(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    called = []
    monkeypatch.setattr(kanban, "_cmd_set_model", lambda _args: called.append(True) or 0)
    assert kanban.kanban_command(Namespace(kanban_action="set-model")) == 1
    assert called == []
    assert "cannot mutate board state through the CLI" in capsys.readouterr().err


def test_running_kanban_worker_can_use_read_only_cli(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="show")) is False
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="list")) is False


def test_interactive_caller_can_mutate_board(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="complete")) is False


def test_worker_list_path_does_not_recompute_or_mutate_ready_state(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setattr(kanban.kb, "connect_closing", lambda: nullcontext(object()))
    monkeypatch.setattr(
        kanban.kb,
        "recompute_ready",
        lambda _conn: (_ for _ in ()).throw(AssertionError("worker list mutated board")),
    )
    monkeypatch.setattr(kanban.kb, "list_tasks", lambda _conn, **_kwargs: [])
    args = Namespace(
        assignee=None,
        mine=False,
        status=None,
        tenant=None,
        session=None,
        archived=False,
        sort=None,
        workflow_template_id=None,
        current_step_key=None,
        json=True,
    )
    assert kanban._cmd_list(args) == 0
    assert capsys.readouterr().out.strip() == "[]"
