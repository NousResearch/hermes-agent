from __future__ import annotations

from argparse import Namespace

from hermes_cli.kanban import _is_kanban_worker_cli_mutation


def test_running_kanban_worker_cannot_mutate_board_through_cli(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="complete")) is True
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="promote")) is True


def test_running_kanban_worker_can_use_read_only_cli(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="show")) is False
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="list")) is False


def test_interactive_caller_can_mutate_board(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    assert _is_kanban_worker_cli_mutation(Namespace(kanban_action="complete")) is False
