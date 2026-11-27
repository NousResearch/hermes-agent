"""A plugin slash command asks the classic CLI's user to pick an option (``ctx.ask_choice``) with the
CLI's own clarify panel, and gets the picked choice's value back:

* in process (no gateway), the handler's ``ask_choice`` opens the panel directly;
* on the gateway, the handler runs there and asks with a ``clarify`` server request; the CLI answers
  it with the same panel while it still waits for its ``slash.exec`` reply.
"""
from __future__ import annotations

import json
import queue
import threading
import types
from pathlib import Path

import pytest

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager

_CHOICES = [{"value": "default", "label": "Default", "description": "delegation.model / parent model"},
            {"value": "model_map", "label": "Model routes"}, "profile_routes"]
_LABELS = ["Default — delegation.model / parent model", "Model routes", "profile_routes"]


def _panel(pick):
    """The CLI's clarify panel answering with the label at index *pick*; records what it showed."""
    shown = []

    def clarify_callback(questions):
        shown.append(questions)
        return {"answers": {questions[0]["qid"]: questions[0]["choices"][pick]}, "outcome": "submitted"}
    return clarify_callback, shown


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    plugin = home / "plugins" / "pick"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(json.dumps({"name": "pick", "version": "0.1.0"}))
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    choices = {_CHOICES!r}\n"
        "    ctx.register_command('delegation', lambda arg: 'MODE=%s' % ctx.ask_choice('Delegation mode', choices))\n")
    (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["pick"]}}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.discover_plugins()
    yield home
    plugins_mod._reset_plugin_managers_for_tests()


def test_an_in_process_plugin_command_picks_with_the_cli_clarify_panel(plugin_home, monkeypatch, capsys):
    from cli import HermesCLI
    from hermes_cli.plugin_command_dispatch import invoke_plugin_command
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda _cli: None)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "s1"
    cli._clarify_callback, shown = _panel(1)
    cli._run_plugin_slash_command("/delegation", "")
    assert "MODE=model_map" in capsys.readouterr().out
    assert shown == [[{"qid": "q0", "question": "Delegation mode", "choices": _LABELS, "multi_select": False}]]
    # Called where no surface binds an asker (a hook, a cron job): None, and the handler says so itself.
    assert invoke_plugin_command(plugins_mod.get_plugin_command_handler("delegation"), "") == "MODE=None"


class _GatewayConnection:
    """The CLI's gateway connection while the gateway runs the command: its handler asks with a
    ``clarify`` server request on the session's event stream, and its reply waits for the answer."""
    closed = final = False
    epoch = 1

    def __init__(self, session_id):
        self.events: queue.Queue = queue.Queue()
        self.session_id = session_id
        self.answers: queue.Queue = queue.Queue()
        self.session = types.SimpleNamespace(request_listeners=[])

    def state(self, session_id):
        assert session_id == self.session_id
        return self.session

    def call_until(self, method, wake, **params):
        assert (method, params["command"]) == ("slash.exec", "delegation")
        self.events.put({"jsonrpc": "2.0", "method": "event", "params": {
            "session_id": self.session_id, "type": "message.delta", "payload": {"text": "later turn"}}})
        self.events.put({"jsonrpc": "2.0", "id": "srq-1", "method": "clarify", "params": {
            "session_id": self.session_id,
            "questions": [{"qid": "q0", "question": "Delegation mode", "choices": _LABELS, "multi_select": False}]}})
        request_id, result = self.answers.get(timeout=10)
        assert request_id == "srq-1"
        return {"type": "exec", "output": "MODE=" + result["answers"]["q0"]}

    def reply(self, request_id, result):
        self.answers.put((request_id, result))


def test_a_gateway_plugin_command_choice_is_answered_while_the_cli_waits(monkeypatch, capsys):
    from cli import HermesCLI
    from hermes_cli import gateway_plugin_commands
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "s1"
    cli._clarify_callback, shown = _panel(2)
    connection = _GatewayConnection("s1")
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda _cli: connection)
    result: list = []
    runner = threading.Thread(target=lambda: result.append(gateway_plugin_commands.run(cli, "delegation", "")))
    runner.start()
    runner.join(15)
    assert result == [True]
    assert "MODE=profile_routes" in capsys.readouterr().out
    assert shown and shown[0][0]["choices"] == _LABELS
    # The session's other frames stay queued, in order, for the turn pump that owns them.
    assert [f["params"]["type"] for f in list(connection.events.queue)] == ["message.delta"]


def test_a_settled_choice_closes_the_cli_picker(monkeypatch, capsys):
    """Another viewer answers the command's picker (the gateway's ``request.cancel``): the CLI's panel closes
    instead of keeping the input for an already-settled request."""
    from cli import HermesCLI
    from hermes_cli import gateway_plugin_commands
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "s1"
    opened, closed = threading.Event(), threading.Event()

    def panel(questions):
        opened.set()
        closed.wait(10)  # a withdrawn panel returns nothing
    cli._clarify_callback = panel
    cli._withdraw_gateway_modal = lambda method: closed.set() if method == "clarify" else None
    connection = _GatewayConnection("s1")

    def other_viewer_answers(method, wake, **params):
        connection.events.put({"jsonrpc": "2.0", "id": "srq-1", "method": "clarify", "params": {
            "session_id": "s1", "questions": [{"qid": "q0", "question": "Delegation mode", "choices": _LABELS}]}})
        assert opened.wait(10)
        for listener in list(connection.session.request_listeners):
            listener({"id": "srq-1", "method": "clarify"})
        connection.answers.get(timeout=10)  # the panel's late reply, dropped by the gateway
        return {"type": "exec", "output": "MODE=model_map"}
    connection.call_until = other_viewer_answers
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda _cli: connection)
    result: list = []
    runner = threading.Thread(target=lambda: result.append(gateway_plugin_commands.run(cli, "delegation", "")))
    runner.start()
    runner.join(15)
    assert closed.is_set() and result == [True]
    assert "MODE=model_map" in capsys.readouterr().out


def test_a_described_choice_still_accepts_its_plain_label():
    from hermes_cli.plugin_choices import ask_choice, bind_asker
    with bind_asker(lambda question, labels: "model routes"):
        assert ask_choice("Delegation mode", [
            {"value": "model_map", "label": "Model routes", "description": "a model per task"}]) == "model_map"
