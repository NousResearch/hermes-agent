"""Fail-closed plugin command dispatch across TUI entry points."""

from hermes_cli import plugins
from tui_gateway import server


def _secure_entry(seen):
    def handler(raw_args, *, command_context):
        seen.append((raw_args, command_context))
        return "unsafe"

    return {"handler": handler, "authenticated_context": True}


def test_command_dispatch_denies_authenticated_plugin_without_invocation(monkeypatch):
    seen = []
    monkeypatch.setattr(plugins, "_get_plugin_command_entry", lambda _name: _secure_entry(seen))

    response = server.handle_request({
        "id": "secure-command-dispatch",
        "method": "command.dispatch",
        "params": {"name": "secure", "arg": "payload", "session_id": ""},
    })

    assert response["error"]["code"] == 4013
    assert "authenticated gateway request" in response["error"]["message"]
    assert seen == []


def test_tui_secondary_plugin_route_uses_central_dispatcher(monkeypatch):
    seen = []
    monkeypatch.setattr(plugins, "_get_plugin_command_entry", lambda _name: _secure_entry(seen))

    dispatched = server._dispatch_tui_plugin_command("secure", "payload")

    assert dispatched.found is True
    assert dispatched.denied is True
    assert seen == []


def test_command_dispatch_does_not_fall_through_after_plugin_failure(monkeypatch):
    def handler(_raw_args):
        raise RuntimeError("ambiguous external outcome")

    monkeypatch.setattr(
        plugins,
        "_get_plugin_command_entry",
        lambda _name: {"handler": handler, "authenticated_context": False},
    )

    response = server.handle_request({
        "id": "failed-command-dispatch",
        "method": "command.dispatch",
        "params": {"name": "plugin-send", "arg": "payload", "session_id": ""},
    })

    assert response["error"]["code"] == 5030
    assert response["error"]["message"] == "Plugin command failed."


def test_command_dispatch_contains_dispatch_infrastructure_failure(monkeypatch):
    def broken_dispatch(_name, _arg):
        raise RuntimeError("sensitive infrastructure detail")

    monkeypatch.setattr(server, "_dispatch_tui_plugin_command", broken_dispatch)

    response = server.handle_request({
        "id": "broken-command-dispatch",
        "method": "command.dispatch",
        "params": {"name": "possibly-secure", "arg": "secret args", "session_id": ""},
    })

    assert response["error"]["code"] == 5030
    assert response["error"]["message"] == "Plugin command failed."
