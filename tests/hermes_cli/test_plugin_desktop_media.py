"""Public PluginContext desktop requests validate input and use the scoped gateway seam."""

import json
import threading

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.plugins import PluginContext
from tui_gateway import server
from tui_gateway.transport import StdioTransport


class _Agent:
    session_id = "stored-sid"

    def __init__(self, callback):
        self.player_agent_callback = callback


def _capability():
    return PluginContext.__new__(PluginContext).desktop


def test_public_desktop_accessor_routes_one_valid_request_to_live_link(monkeypatch):
    delivered = []
    record = {"agent": _Agent(lambda operation: delivered.append(operation) or json.dumps({"ok": True}))}
    monkeypatch.setattr(server, "_sessions", {"runtime-sid": record})
    monkeypatch.setattr("tools.desktop_ui.available", lambda: True)
    transport = StdioTransport(lambda: None, threading.Lock())
    record["transport"] = transport
    vars_token = set_session_vars(ui_session_id="runtime-sid")
    transport_token = server.bind_transport(transport)
    runtime_token = server._current_runtime_session_record.set(record)
    try:
        capability = _capability()
        assert capability.available()
        assert capability.media_request(session_id="stored-sid", operation={"type": "context"}) == {"ok": True}
    finally:
        server._current_runtime_session_record.reset(runtime_token)
        server.reset_transport(transport_token)
        clear_session_vars(vars_token)
    assert delivered == [{"type": "context"}]


def test_public_desktop_accessor_fails_closed_without_sink_or_for_invalid_input(monkeypatch):
    capability = _capability()
    monkeypatch.setattr("tools.desktop_ui.available", lambda: False)
    assert not capability.available()
    assert capability.media_request(session_id="stored-sid", operation={"type": "context"}) == {
        "error": "desktop-player-unavailable"
    }
    monkeypatch.setattr("tools.desktop_ui.available", lambda: True)
    assert capability.media_request(session_id="", operation={"type": "context"}) == {
        "error": "session-unavailable"
    }
    assert capability.media_request(session_id="stored-sid", operation={"type": "context", "url": "forged"}) == {
        "error": "invalid-operation"
    }
