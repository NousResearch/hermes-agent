"""Plugin player calls use the canonical GUI turn, never the legacy runtime table."""
import contextvars
import json
import threading
from types import SimpleNamespace

import pytest

from gateway import local_desktop_ui, session_server_requests
from hermes_cli.plugin_desktop import DesktopPluginCapability


@pytest.fixture
def player(monkeypatch):
    requests = []

    class Fanout:
        def transports(self):
            return [self]

        def write(self, frame, **kwargs):
            requests.append(frame)
            session_server_requests.resolve({
                "id": frame["id"], "result": {"value": json.dumps({"operation": "context", "ok": True})},
            })
            return True

    policy = SimpleNamespace(source="gui")
    monkeypatch.setattr("gateway.session_policy.active_policy", lambda: policy)
    monkeypatch.setattr(local_desktop_ui, "_install_emitter", lambda: None)
    monkeypatch.setattr(local_desktop_ui, "_install_process_sinks", lambda: None)
    monkeypatch.setattr(local_desktop_ui, "_install_connection_hook", lambda: None)
    monkeypatch.setattr(local_desktop_ui, "_turns", {})
    monkeypatch.setattr(local_desktop_ui, "_process_routes", {})
    stream = SimpleNamespace(fanout=Fanout(), lock=threading.RLock(), publish=lambda *a, **kw: None)
    live = SimpleNamespace(event_stream=stream)
    agent = SimpleNamespace(session_id="stored", _interrupt_requested=False)
    authority = SimpleNamespace(sessions={"canonical": live}, executing={"canonical": 4},
                                adopted={"canonical": (4, agent)})
    turn = SimpleNamespace(_ctx=SimpleNamespace(session_id="stored", session_key="route"),
                           _approval_owner=(authority, "canonical", 4),
                           _publish_execution=lambda *a, **kw: True)
    capability = DesktopPluginCapability()
    yield SimpleNamespace(turn=turn, agent=agent, authority=authority, live=live,
                          requests=requests, capability=capability, policy=policy)


def test_plugin_call_routes_through_canonical_viewer_and_releases_with_turn(player):
    sinks = local_desktop_ui.wire(player.turn, player.agent)
    copied = contextvars.copy_context()
    try:
        assert player.capability.media_request(session_id="stored", operation={"type": "context"}) == {
            "operation": "context", "ok": True,
        }
        assert player.requests[-1]["method"] == "player.agent.request"
        assert player.requests[-1]["params"] == {"session_id": "canonical", "operation": {"type": "context"}}
    finally:
        local_desktop_ui.release(sinks)
    assert copied.run(player.capability.media_request, session_id="stored", operation={"type": "context"}) == {
        "error": "desktop-player-unavailable",
    }
    assert len(player.requests) == 1


@pytest.mark.parametrize("denial", ["wrong-session", "replaced-live", "new-generation", "wrong-agent", "not-gui", "wrong-policy"])
def test_plugin_call_rejects_foreign_or_expired_canonical_turn(player, monkeypatch, denial):
    sinks = local_desktop_ui.wire(player.turn, player.agent)
    session_id = "another" if denial == "wrong-session" else "stored"
    mutations = {
        "wrong-session": lambda: None,
        "replaced-live": lambda: player.authority.sessions.update(
            canonical=SimpleNamespace(event_stream=player.live.event_stream)),
        "new-generation": lambda: player.authority.executing.update(canonical=5),
        "wrong-agent": lambda: player.authority.adopted.update(canonical=(4, SimpleNamespace(session_id="stored"))),
        "not-gui": lambda: setattr(player.policy, "source", "tui"),
        "wrong-policy": lambda: monkeypatch.setattr(
            "gateway.session_policy.active_policy", lambda: SimpleNamespace(source="gui")),
    }
    mutations[denial]()
    try:
        assert player.capability.media_request(session_id=session_id, operation={"type": "context"}) == {
            "error": "desktop-player-unavailable",
        }
        assert player.requests == []
    finally:
        local_desktop_ui.release(sinks)
