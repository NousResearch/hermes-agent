"""Explicit backlog disposal must work even while a foreground turn runs."""
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cli import HermesCLI
from tools.process_registry import ProcessRegistry


def fixture(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "owner"
    cli._session_db = None
    cli._pending_input = queue.Queue()
    cli._agent_running = True
    cli._should_handle_model_command_inline = lambda *a, **kw: False
    cli._process_unregistered_slash = Mock(return_value=True)
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr("cli.CLI_CONFIG", {"display": {"ctrl_c_purge_notifications": False}})
    return cli, registry


def notice(sid="proc_owned", owner="owner"):
    return {"type": "completion", "session_id": sid, "session_key": owner,
            "command": "fixture", "exit_code": 0, "output": "retained result"}


def test_purge_dispatch_discards_only_owned_reports_even_with_ctrl_c_opt_out(monkeypatch):
    cli, registry = fixture(monkeypatch)
    own, foreign = notice(), notice("proc_foreign", "other")
    registry.completion_queue.put(own)
    registry.completion_queue.put(foreign)
    cli._pending_input.put("user prompt must survive")
    assert cli.process_command("/purge")
    assert registry.drain_notifications(session_key="owner") == []
    assert registry.is_completion_consumed("proc_owned")
    assert [e for e, _ in registry.drain_notifications(session_key="other")] == [foreign]
    assert cli._pending_input.get_nowait() == "user prompt must survive"
    assert cli._agent_running
    cli._process_unregistered_slash.assert_not_called()


def test_purge_enter_bypasses_busy_queue(monkeypatch):
    cli, registry = fixture(monkeypatch)
    registry.completion_queue.put(notice())
    event = SimpleNamespace(app=Mock())
    assert cli._tui_enter_inline_command(event, "/purge", False)
    assert registry.completion_queue.empty()
    assert cli._pending_input.empty()
    event.app.current_buffer.reset.assert_called_once_with(append_to_history=True)
    event.app.invalidate.assert_called_once()


@pytest.mark.parametrize("text", ["/purge kill", "/purge all", "/purge unknown", "/purge all extra"])
def test_unsupported_purge_modes_are_not_silently_destructive(monkeypatch, capsys, text):
    cli, registry = fixture(monkeypatch)
    own = notice()
    registry.completion_queue.put(own)
    assert cli.process_command(text)
    assert registry.completion_queue.get_nowait() == own
    assert "Usage:" in capsys.readouterr().out


def test_explicit_empty_purge_reports_zero(monkeypatch, capsys):
    cli, registry = fixture(monkeypatch)
    assert cli.process_command("/purge")
    assert "Discarded 0" in capsys.readouterr().out


def test_purge_failure_keeps_event_and_reports_failure(monkeypatch, capsys):
    cli, registry = fixture(monkeypatch)
    registry.completion_queue.put(notice())
    monkeypatch.setattr(registry, "purge_notifications", Mock(side_effect=RuntimeError("fixture")))
    assert cli.process_command("/purge")
    assert not registry.completion_queue.empty()
    assert "Could not discard" in capsys.readouterr().out


def test_purge_is_not_inline_when_idle_or_attached(monkeypatch):
    cli, _ = fixture(monkeypatch)
    assert not cli._tui_enter_inline_command(SimpleNamespace(app=Mock()), "/purge", True)
    cli._agent_running = False
    assert not cli._tui_enter_inline_command(SimpleNamespace(app=Mock()), "/purge", False)
