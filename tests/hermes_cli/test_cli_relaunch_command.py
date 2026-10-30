"""Classic CLI /relaunch keeps the current conversation, not another console's (#145427)."""

import queue
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cli import HermesCLI
from hermes_cli import relaunch
from hermes_cli.commands import COMMANDS, gateway_help_lines, resolve_command
from hermes_state import SessionDB


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setattr(sys, "argv", ["hermes", "--cli"])
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="this-console", source="cli", model="test")
    instance = HermesCLI.__new__(HermesCLI)
    instance.session_id = "this-console"
    instance._session_db = db
    instance._agent_running = False
    instance._pending_relaunch = None
    instance._pending_session_relaunch = None
    instance.agent = None
    instance._gateway_session_open = True
    instance._voice_recorder = None
    instance._delete_session_on_exit = False
    instance.conversation_history = []
    instance._pet_stop_anim = Mock()
    instance._persist_active_session_before_close = Mock()
    instance._print_exit_summary = Mock()
    instance._release_active_session = Mock()
    yield instance
    db.close()


@pytest.mark.parametrize("empty", [False, True])
def test_relaunch_after_shutdown_resumes_only_surviving_current_session(cli, monkeypatch, empty):
    import cli as cli_module

    if not empty:
        cli._session_db.append_message(cli.session_id, role="user", content="offline fixture")
        cli._session_db.append_message(cli.session_id, role="assistant", content="saved reply")
    cli._session_db.create_session(session_id="other-console", source="cli", model="test")
    cleanup = Mock()
    monkeypatch.setattr(cli_module, "_run_cleanup", cleanup)
    monkeypatch.setattr(cli_module, "_active_worktree", {"path": "keep-this-worktree"})
    ask = Mock(side_effect=AssertionError("relaunch must keep the current worktree"))
    monkeypatch.setattr("hermes_cli.worktree_exit.ask_exit_choice", ask)
    calls = []

    def replace(args, **kwargs):
        assert threading.current_thread() is threading.main_thread()
        cleanup.assert_called_once()
        cli._release_active_session.assert_called_once()
        calls.append(list(args))
        assert kwargs == {"preserve_inherited": False}

    monkeypatch.setattr(relaunch, "relaunch", replace)
    assert cli._handle_relaunch_command("/relaunch") is False
    assert calls == []  # the command worker must never replace the process
    cli._tui_shutdown()
    cli._tui_relaunch()
    assert len(calls) == 1
    if empty:
        assert cli._session_db.get_session(cli.session_id) is None
        assert "--resume" not in calls[0]
    else:
        assert calls[0][-2:] == ["--resume", "this-console"]
        assert cli._session_db.get_session(cli.session_id) is not None
    assert "other-console" not in calls[0]
    assert "--continue" not in calls[0]
    ask.assert_not_called()


def test_launch_options_cwd_and_home_survive_without_replaying_one_shot_flags(cli, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "hermes", "--safe-mode", "--checkpoints", "--pass-session-id", "--yolo",
        "-m", "old-model", "--provider=old-provider", "--reasoning", "low",
        "-s", "old-skill", "--skills=another-skill",
        "--resume=stale-session", "--continue", "latest", "--worktree", "--in", "old-cwd",
        "chat", "-q", "do not replay this prompt",
    ])
    assert cli._handle_relaunch_command("/relaunch") is False
    # The request owns a snapshot; later argv changes cannot change the relaunch.
    monkeypatch.setattr(sys, "argv", ["hermes", "--provider=unrelated-provider"])
    replacement = Mock()
    monkeypatch.setattr(relaunch, "relaunch", replacement)
    cli._tui_relaunch()
    args = replacement.call_args.args[0]
    assert args == [
        "--safe-mode", "--checkpoints", "--pass-session-id", "--yolo",
        "--cli", "--in", str(tmp_path), "--resume", "this-console",
    ]
    import os
    assert os.environ["HERMES_HOME"] == str(tmp_path / "profile")
    replacement.assert_called_once_with(args, preserve_inherited=False)


def test_normal_quit_still_offers_worktree_exit(cli, monkeypatch):
    import cli as cli_module

    worktree = {"path": "current-worktree"}
    monkeypatch.setattr(cli_module, "_active_worktree", worktree)
    monkeypatch.setattr(cli_module, "_signal_watchdog_armed", False)
    monkeypatch.setattr(cli_module, "_run_cleanup", Mock())
    ask, apply = Mock(), Mock()
    monkeypatch.setattr("hermes_cli.worktree_exit.ask_exit_choice", ask)
    monkeypatch.setattr("hermes_cli.worktree_exit.apply_exit_choice", apply)
    cli._tui_shutdown()
    ask.assert_called_once_with(worktree)
    apply.assert_called_once_with(worktree, cli_module._cprint)


def test_pruned_empty_session_keeps_original_launch_picks(cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hermes", "-m", "launch-model", "-s", "launch-skill"])
    replacement = Mock()
    monkeypatch.setattr(relaunch, "relaunch", replacement)
    assert cli._handle_relaunch_command("/relaunch") is False
    cli._session_db.end_session(cli.session_id, "cli_close")
    assert cli._discard_session_if_empty(cli.session_id)
    cli._tui_relaunch()
    args = replacement.call_args.args[0]
    assert args[:4] == ["-m", "launch-model", "-s", "launch-skill"]
    assert "--resume" not in args


def test_busy_relaunch_is_dispatched_immediately_and_refused(cli, capsys):
    cli._agent_running = True
    cli._should_exit = False
    cli._pending_input = queue.Queue()
    cli._should_handle_model_command_inline = Mock(return_value=False)
    cli.process_command = Mock(side_effect=cli._handle_relaunch_command)
    app = SimpleNamespace(current_buffer=Mock(), invalidate=Mock(), exit=Mock(), is_running=True)
    assert cli._tui_enter_inline_command(SimpleNamespace(app=app), "/relaunch", False)
    cli.process_command.assert_called_once_with("/relaunch")
    assert cli._pending_input.empty()
    assert cli._pending_relaunch is None
    assert cli._pending_session_relaunch is None
    assert not cli._should_exit
    app.exit.assert_not_called()
    assert "turn is running" in capsys.readouterr().out


@pytest.mark.parametrize("command,store_available", [("/relaunch extra", True), ("/relaunch", False)])
def test_invalid_or_unavailable_relaunch_keeps_cli_open(cli, command, store_available):
    if not store_available:
        cli._session_db = None
    assert cli._handle_relaunch_command(command) is True
    assert cli._pending_relaunch is None
    assert cli._pending_session_relaunch is None


def test_relaunch_is_cli_only_and_dispatched_by_convention():
    command = resolve_command("relaunch")
    assert command.cli_only
    assert "/relaunch" in COMMANDS
    assert not any("/relaunch" in line for line in gateway_help_lines())
    assert HermesCLI._slash_handler("relaunch") == ("_handle_relaunch_command", True)
    assert resolve_command("restart").gateway_only


def test_completed_shutdown_disarms_watchdog_before_waiting_for_replacement(cli, monkeypatch):
    from hermes_cli import cli_shutdown

    event = threading.Event()
    monkeypatch.setattr(cli_shutdown, "_exit_watchdog_cancelled", event)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    targets = []
    monkeypatch.setattr(cli_shutdown.threading, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: targets.append(kw["target"])))
    exit_process = Mock(side_effect=AssertionError("the replaced CLI must not be killed"))
    monkeypatch.setattr(cli_shutdown.os, "_exit", exit_process)
    cli_shutdown._arm_exit_watchdog(timeout_s=60)

    def replace(*args, **kwargs):
        assert event.is_set()
        targets[0]()  # cancelled watchdog returns immediately, without ending the process

    monkeypatch.setattr(relaunch, "relaunch", replace)
    assert cli._handle_relaunch_command("/relaunch") is False
    cli._tui_relaunch()
    exit_process.assert_not_called()
