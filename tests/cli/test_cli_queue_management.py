from __future__ import annotations

from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.queue_management import ManagedPromptQueue


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = ManagedPromptQueue()
    cli._agent_running = True
    cli.session_id = "cli-session"
    return cli


def _output(mock_print) -> str:
    return "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)


@patch("cli._cprint")
def test_cli_empty_queue_opens_snapshot_and_dequeue_removes_position(mock_print):
    cli = _cli()
    cli._pending_input.put("first")
    cli._pending_input.put("second")

    assert cli.process_command("/queue") is True
    assert "1. first" in _output(mock_print)
    assert "2. second" in _output(mock_print)

    mock_print.reset_mock()
    assert cli.process_command("/dq 2") is True
    assert "removed" in _output(mock_print).lower()
    assert [item.preview for item in cli._pending_input.snapshot_items()] == ["first"]


@patch("cli._cprint")
def test_cli_dequeue_all_uses_frozen_ids_and_preserves_new_arrival(mock_print):
    cli = _cli()
    cli._pending_input.put("old one")
    cli._pending_input.put("old two")
    cli.process_command("/queue")
    cli._pending_input.put("new arrival")

    mock_print.reset_mock()
    cli.process_command("/dequeue all")

    assert "removed 2" in _output(mock_print).lower()
    assert [item.preview for item in cli._pending_input.snapshot_items()] == ["new arrival"]


@patch("cli._cprint")
def test_cli_dequeue_rejects_view_from_previous_session(mock_print):
    cli = _cli()
    cli._pending_input.put("keep me")
    cli.process_command("/queue")
    cli.session_id = "new-cli-session"

    mock_print.reset_mock()
    cli.process_command("/dequeue 1")

    assert "run `/queue`" in _output(mock_print).lower()
    assert [item.preview for item in cli._pending_input.snapshot_items()] == ["keep me"]


@patch("cli._cprint")
def test_cli_queue_view_hides_system_items(mock_print):
    cli = _cli()
    cli._pending_input.put("user turn")
    cli._pending_input.put_system("[Continuing toward your standing goal]")

    cli.process_command("/queue")

    output = _output(mock_print)
    assert "user turn" in output
    assert "Continuing toward" not in output


@patch("cli._cprint")
def test_cli_dequeue_without_snapshot_is_fail_safe(mock_print):
    cli = _cli()
    cli._pending_input.put("keep me")

    cli.process_command("/dequeue 1")

    assert "run `/queue`" in _output(mock_print).lower()
    assert [item.preview for item in cli._pending_input.snapshot_items()] == ["keep me"]