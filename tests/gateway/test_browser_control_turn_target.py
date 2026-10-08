"""The broker names the user's turn tab on every command instead of leaving it to the extension."""

import asyncio
import json

from gateway.browser_control_broker import BrowserControlBroker, ControllerScope, browser_turn_target
from gateway.platforms.api_server_browser_control import bind_browser_turn_target, run_controller_keepalive


def _envelope(**control):
    base = {"route": "extension-controller", "availability": "available", "controller_id": "ctrl-1",
            "tab_id": 42, "frame_id": 0, "document_generation": 4}
    base.update(control)
    return json.dumps({"protocol": "hermes.browser.turn.v2", "human_input": {"text": "hi"},
                       "browser_control": base}, separators=(",", ":"))


def test_turn_target_parsed_from_envelope():
    assert browser_turn_target(_envelope()) == {"controller_id": "ctrl-1", "tab_id": 42, "frame_id": 0}


def test_turn_target_ignores_unusable_envelopes():
    assert browser_turn_target("plain chat") is None
    assert browser_turn_target(None) is None
    assert browser_turn_target(_envelope(availability="unavailable")) is None
    assert browser_turn_target(_envelope(route="isolated")) is None
    assert browser_turn_target(_envelope(tab_id=0)) is None
    assert browser_turn_target(_envelope(tab_id=True)) is None
    assert browser_turn_target(_envelope(controller_id="")) is None
    assert browser_turn_target('{"protocol":"hermes.browser.turn.v2", broken') is None


def _dispatch_frame(broker, scope):
    frames = []

    def send(frame):
        frames.append(frame)
        broker.complete(frame["params"]["command_id"], scope=scope, ok=True, result={"ok": True})

    broker.attach(scope, send, owner=object())
    broker.dispatch(scope, action="browser_snapshot", arguments={})
    return frames[-1]["params"]


def test_dispatch_carries_turn_tab_for_matching_controller():
    broker = BrowserControlBroker(command_timeout=2.0)
    scope = ControllerScope(principal_id="p", session_id="s1", controller_id="ctrl-1",
                            browser_profile_id="b", transport_family="api",
                            capabilities=frozenset({"browser_snapshot"}))
    target = browser_turn_target(_envelope())
    assert target is not None
    broker.set_turn_target("s1", controller_id=target["controller_id"], tab_id=target["tab_id"],
                           frame_id=target["frame_id"])
    params = _dispatch_frame(broker, scope)
    assert params["tab_id"] == 42 and params["frame_id"] == 0
    assert "document_generation" not in params


def test_dispatch_omits_tab_for_other_controller_or_cleared_target():
    broker = BrowserControlBroker(command_timeout=2.0)
    scope = ControllerScope(principal_id="p", session_id="s1", controller_id="ctrl-2",
                            browser_profile_id="b", transport_family="api",
                            capabilities=frozenset({"browser_snapshot"}))
    broker.set_turn_target("s1", controller_id="ctrl-1", tab_id=42)
    assert "tab_id" not in _dispatch_frame(broker, scope)
    broker.set_turn_target("s1", controller_id="ctrl-2", tab_id=7)
    broker.clear_turn_target("s1")
    assert "tab_id" not in _dispatch_frame(broker, scope)


def test_bind_turn_target_sets_and_clears():
    broker = BrowserControlBroker()
    bind_browser_turn_target(broker, "s1", _envelope(tab_id=9))
    scope = ControllerScope(session_id="s1", controller_id="ctrl-1")
    assert broker._turn_target_for(scope) == {"controller_id": "ctrl-1", "tab_id": 9, "frame_id": 0}
    bind_browser_turn_target(broker, "s1", "no envelope this turn")
    assert broker._turn_target_for(scope) is None


def test_keepalive_sends_noop_heartbeats_until_closed():
    class WS:
        closed = False

        def __init__(self):
            self.sent = []

        async def send_json(self, frame):
            self.sent.append(frame)
            if len(self.sent) == 3:
                self.closed = True

    ws = WS()
    asyncio.run(asyncio.wait_for(run_controller_keepalive(ws, interval=0.001), timeout=2))
    assert len(ws.sent) == 3
    assert all(f["method"] == "browser.controller.heartbeat" and f["params"]["ok"] is True for f in ws.sent)
    assert len({f["params"]["nonce"] for f in ws.sent}) == 3
