"""Desktop window bridges for a local viewer's turn on the session authority.

A GUI session is offered the ``desktop_ui`` tools; on the authority their emitter, GUI callbacks
and process-registry sinks must reach the session's viewers, and be released with the turn.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway import local_desktop_ui, local_viewer_events, session_server_requests
from gateway.session_context import clear_session_vars, set_session_vars
from tools import desktop_ui
from tools.process_registry import process_registry

_ANSWERS = {
    "preview.read": {"value": "page text"},
    "terminal.read": {"value": "tick 1\ntick 2"},
    "window.read": {"value": json.dumps({"app": "Finder"})},
    "preview.act": {"value": json.dumps({"success": True, "acted": "clicked a"})},
    "tour": {"value": json.dumps({"success": True})},
    "theme.apply": {"value": json.dumps({"applied": {"accent": "violet"}})},
    "theme.pick": {"picked": {"accent": "violet"}},
    "setup_beat": {"choice": "next"},
}


class _Fanout:
    """A viewer that answers every request it knows, unless told to stay silent."""

    def __init__(self, requests, silent):
        self.requests, self.silent = requests, silent

    def transports(self):
        return [object()]

    def write(self, frame, **_kw):
        self.requests.append(frame)
        if frame.get("method") in _ANSWERS and frame["method"] not in self.silent:
            session_server_requests.resolve({"id": frame["id"], "result": _ANSWERS[frame["method"]]})
        return True


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(local_desktop_ui, "_turns", {})
    monkeypatch.setattr(local_desktop_ui, "_preview_bridge", {})
    monkeypatch.setattr(local_desktop_ui, "_tour_bridge", {})
    monkeypatch.setattr(local_desktop_ui, "_process_routes", {})
    monkeypatch.setattr(local_desktop_ui, "_emitter_installed", False)
    monkeypatch.setattr(local_desktop_ui, "_process_sinks_installed", False)
    monkeypatch.setattr(desktop_ui, "_emit", None)
    monkeypatch.setattr(process_registry, "on_output", None)
    monkeypatch.setattr(process_registry, "on_close", None)

    published, streamed, requests, silent = [], [], [], set()
    stream = SimpleNamespace(
        fanout=_Fanout(requests, silent),
        publish=lambda session_id, payload, *, event_type: streamed.append((session_id, event_type, payload)))
    authority = SimpleNamespace(sessions={"canon": SimpleNamespace(event_stream=stream)})
    turn = SimpleNamespace(
        _ctx=SimpleNamespace(session_id="phys", session_key="agent:main:local:dm:1"),
        _approval_owner=(authority, "canon", 1),
        _publish_execution=lambda kind, payload: published.append((kind, payload)) or True)
    agent = SimpleNamespace(_interrupt_requested=False)
    tokens = set_session_vars(session_id="phys")
    yield SimpleNamespace(turn=turn, agent=agent, published=published, streamed=streamed,
                          requests=requests, silent=silent)
    clear_session_vars(tokens)


def _emit(event, payload):
    return desktop_ui.emit_or_error(event, payload, "failed: ", "desktop only", {"success": True})


def test_window_events_reach_the_turns_stream(window):
    sinks = local_viewer_events.wire(window.turn, window.agent)
    assert json.loads(_emit("preview.open", {"url": "https://example.com"})) == {"success": True}
    assert window.published[-1] == ("preview.open", {"url": "https://example.com"})
    local_viewer_events.restore(window.agent, sinks)
    assert "failed: " in _emit("preview.open", {"url": "https://example.com"})


def test_gui_reads_are_server_requests_to_the_session_viewers(window):
    local_viewer_events.wire(window.turn, window.agent)
    assert window.agent.read_preview_callback() == "page text"
    assert window.agent.read_terminal_callback(start=0, count=2) == "tick 1\ntick 2"
    assert json.loads(window.agent.read_window_below_callback()) == {"app": "Finder"}
    assert json.loads(window.agent.drive_preview_callback({"action": "click"}))["success"] is True
    terminal = next(f for f in window.requests if f["method"] == "terminal.read")
    assert terminal["params"] == {"session_id": "canon", "start": 0, "count": 2}


def test_set_theme_applies_and_picks(window):
    local_viewer_events.wire(window.turn, window.agent)
    assert window.agent.theme_callback({"accent": "violet"}) == {"applied": {"accent": "violet"}}
    assert window.agent.theme_callback({"picker": "accent"}) == {"success": True, "picked": {"accent": "violet"}}


def test_tour_fails_fast_once_no_window_answered(window, monkeypatch):
    monkeypatch.setattr(local_desktop_ui, "_TOUR_PROBE_TIMEOUT_S", 0.3)
    window.silent.add("tour")
    local_viewer_events.wire(window.turn, window.agent)
    assert "No Hermes Desktop window answered the tour" in window.agent.tour_callback({"action": "start"})
    sent = len(window.requests)
    assert "No Hermes Desktop window answered the tour" in window.agent.tour_callback({"action": "start"})
    assert len(window.requests) == sent


def test_background_process_output_follows_its_session_after_the_turn(window):
    sinks = local_viewer_events.wire(window.turn, window.agent)
    local_viewer_events.restore(window.agent, sinks)
    proc = SimpleNamespace(id="proc_1", session_key="agent:main:local:dm:1")
    process_registry.on_output(proc, "tick 1\n")
    process_registry.on_close(proc, "proc_1")
    assert window.streamed == [
        ("canon", "agent.terminal.output", {"process_id": "proc_1", "chunk": "tick 1\n"}),
        ("canon", "terminal.close", {"process_id": "proc_1"})]


def test_restore_puts_the_agents_callbacks_back(window):
    sinks = local_viewer_events.wire(window.turn, window.agent)
    local_viewer_events.restore(window.agent, sinks)
    for attr in local_desktop_ui.CALLBACK_ATTRS:
        assert getattr(window.agent, attr) is None
    assert local_desktop_ui._turns == {}
