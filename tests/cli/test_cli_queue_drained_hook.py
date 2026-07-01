from __future__ import annotations

import io
import queue
import sys
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.plugins import VALID_HOOKS


def _make_cli_for_queue_drained():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()
    cli._last_turn_interrupted = False
    cli.bell_on_complete = False
    cli.agent = SimpleNamespace(session_id="agent-session")
    cli.session_id = "cli-session"
    cli.platform = "cli"
    return cli


def test_cli_queue_drained_hook_is_registered():
    assert "cli_queue_drained" in VALID_HOOKS


def test_queue_drained_hook_waits_for_pending_queue(monkeypatch):
    cli = _make_cli_for_queue_drained()
    cli.bell_on_complete = True
    cli._pending_input.put("next queued turn")

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    with patch.object(cli, "_emit_cli_queue_drained_hook") as emit_hook:
        emitted = cli._emit_completion_cues_if_idle(
            user_input="first turn",
            response="finished",
        )

    assert emitted is False
    assert stdout.getvalue() == ""
    emit_hook.assert_not_called()


def test_queue_drained_hook_emits_bell_and_payload_when_idle(monkeypatch):
    cli = _make_cli_for_queue_drained()
    cli.bell_on_complete = True

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    emitted_payloads = []

    def fake_invoke_hook(name, **kwargs):
        emitted_payloads.append((name, kwargs))
        return [{"started": True}]

    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "cli_queue_drained")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    emitted = cli._emit_completion_cues_if_idle(
        user_input="Implement queue drained hook",
        response="finished",
    )

    assert emitted is True
    assert stdout.getvalue() == "\a"
    assert len(emitted_payloads) == 1
    name, payload = emitted_payloads[0]
    assert name == "cli_queue_drained"
    assert payload["cli"] is cli
    assert payload["agent"] is cli.agent
    assert payload["session_id"] == "cli-session"
    assert payload["platform"] == "cli"
    assert payload["user_input"] == "Implement queue drained hook"
    assert payload["response"] == "finished"
    assert payload["queued_work_drained"] is True
    assert payload["bell_emitted"] is True
    assert payload["interrupted"] is False
    assert isinstance(payload["timestamp"], float)


def test_queue_drained_hook_can_emit_without_bell(monkeypatch):
    cli = _make_cli_for_queue_drained()
    cli.bell_on_complete = False

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "cli_queue_drained")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [{"started": True}])

    emitted = cli._emit_completion_cues_if_idle(
        user_input="Run tests",
        response="finished",
    )

    assert emitted is True
    assert stdout.getvalue() == ""


def test_queue_drained_hook_suppressed_for_interrupted_turn(monkeypatch):
    cli = _make_cli_for_queue_drained()
    cli._last_turn_interrupted = True

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    with patch.object(cli, "_emit_cli_queue_drained_hook") as emit_hook:
        emitted = cli._emit_completion_cues_if_idle(
            user_input="run tests",
            response="finished",
        )

    assert emitted is False
    assert stdout.getvalue() == ""
    emit_hook.assert_not_called()


def test_queue_drained_hook_suppressed_for_empty_or_error_response(monkeypatch):
    cli = _make_cli_for_queue_drained()
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    with patch.object(cli, "_emit_cli_queue_drained_hook") as emit_hook:
        assert cli._emit_completion_cues_if_idle(user_input="x", response="") is False
        assert cli._emit_completion_cues_if_idle(user_input="x", response="Error: failed") is False

    assert stdout.getvalue() == ""
    emit_hook.assert_not_called()
