"""CLI heartbeats must remain tied to the process until the queued turn starts."""

import queue
from types import SimpleNamespace
from unittest.mock import Mock

from cli import HermesCLI
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import ProcessHeartbeatNotification


def _queued_heartbeat(monkeypatch):
    registry = ProcessRegistry()
    process = ProcessSession(
        id="proc_cli_heartbeat", command="dev server", session_key="cli-session",
        started_at=123.0, heartbeat_seconds=60,
    )
    registry._running[process.id] = process
    registry._emit_heartbeat(process, 183.0)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-session"
    cli._pending_input = queue.Queue()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *args: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *args: None)
    cli._drain_process_notifications("cli-idle")
    pending = cli._pending_input.get_nowait()
    assert isinstance(pending, ProcessHeartbeatNotification)
    return cli, process, pending


def test_cli_discards_heartbeat_when_process_exits_after_queue_admission(monkeypatch):
    cli, process, pending = _queued_heartbeat(monkeypatch)
    cli.chat = Mock()
    cli._print_user_message_preview = Mock()
    process.mark_exited(-15)

    cli._tui_process_one_input(pending)

    cli.chat.assert_not_called()
    cli._print_user_message_preview.assert_not_called()


def test_cli_rechecks_heartbeat_just_before_starting_turn(monkeypatch):
    cli, process, pending = _queued_heartbeat(monkeypatch)
    cli._pending_resume_sessions = []
    cli._typed_voice_stop = lambda text: process.mark_exited(-15) or False
    cli.handle_bang_shell = lambda text: False
    cli._print_user_message_preview = Mock()
    cli.chat = Mock()
    cli._app = SimpleNamespace(invalidate=lambda: None)

    cli._tui_process_one_input(pending)

    cli.chat.assert_not_called()
    cli._print_user_message_preview.assert_not_called()


def test_cli_delivers_current_heartbeat(monkeypatch):
    cli, _process, pending = _queued_heartbeat(monkeypatch)
    cli._pending_resume_sessions = []
    cli._typed_voice_stop = lambda text: False
    cli.handle_bang_shell = lambda text: False
    cli._print_user_message_preview = Mock()
    cli._turn_summary_begin = lambda: None
    cli._tui_after_turn = lambda: None
    cli.chat = Mock()
    cli._app = SimpleNamespace(invalidate=lambda: None)

    cli._tui_process_one_input(pending)

    cli.chat.assert_called_once()
    assert "still running" in cli.chat.call_args.args[0]
