"""Cron agents drive the desktop preview pane (#120361).

A cron job with the ``desktop_ui`` toolset must be able to read and drive the
preview pane through the same preview.read / preview.act bridge desktop
sessions use, routed to the most recently active live desktop window.
Outside the desktop gateway (no emitter) nothing changes: no callbacks, and
the tools keep their 'desktop app only' error.
"""

import json
import threading

import pytest

from cron.scheduler import _construct_cron_agent, _CronAgentSetup, _cron_preview_callbacks
from tools import desktop_ui
from tools.drive_preview_tool import drive_preview_tool
from tools.read_preview_tool import read_preview_tool
from tui_gateway import agent_callbacks as acb


@pytest.fixture()
def no_emitter():
    desktop_ui.set_emitter(None)
    try:
        yield
    finally:
        desktop_ui.set_emitter(None)


@pytest.fixture()
def fake_bridge(monkeypatch):
    """Pretend to be the desktop gateway: emitter installed, two live windows."""
    desktop_ui.set_emitter(lambda sid, event, payload: None)
    calls = []
    sessions = {
        "sid-old": {"session_key": "old", "last_active": 100.0, "live_transport": True},
        "sid-new": {"session_key": "new", "last_active": 200.0, "live_transport": True},
        "sid-dead": {"session_key": "dead", "last_active": 300.0, "live_transport": False},
    }

    class FakeTransports:
        @staticmethod
        def _session_has_live_transport(session):
            return bool(session.get("live_transport"))

    def fake_ask(method, sid, params, timeout=None):
        calls.append((method, sid, params, timeout))
        return json.dumps({"success": True, "method": method, "sid": sid})

    monkeypatch.setattr(acb, "_sessions", sessions, raising=False)
    monkeypatch.setattr(acb, "_sessions_lock", threading.RLock(), raising=False)
    monkeypatch.setattr(acb, "_session_transports", FakeTransports, raising=False)
    monkeypatch.setattr(acb, "_ask", fake_ask, raising=False)
    try:
        yield calls
    finally:
        desktop_ui.set_emitter(None)


def _setup():
    setup = _CronAgentSetup()
    setup.model = "test-model"
    setup.runtime = {}
    setup.prefill_messages = []
    setup.max_iterations = 5
    setup.reasoning_config = None
    setup.fallback_model = None
    setup.credential_pool = None
    return setup


def _construct(caught):
    def fake_agent_cls(**kwargs):
        caught.update(kwargs)
        return object()

    return _construct_cron_agent(
        fake_agent_cls, {"id": "j", "name": "t"}, {}, _setup(),
        workdir="", session_id="cron_j_x", session_db=None)


def test_no_bridge_outside_desktop_gateway(no_emitter):
    assert _cron_preview_callbacks() == {}
    caught = {}
    _construct(caught)
    assert caught["read_preview_callback"] is None
    assert caught["drive_preview_callback"] is None
    # Today's error text is preserved headless.
    assert "desktop" in json.loads(drive_preview_tool(action="elements", callback=None))["error"]


def test_cron_agent_gets_preview_callbacks(fake_bridge):
    cbs = _cron_preview_callbacks()
    assert set(cbs) == {"read_preview_callback", "drive_preview_callback"}
    caught = {}
    _construct(caught)
    assert callable(caught["read_preview_callback"])
    assert callable(caught["drive_preview_callback"])
    # Wired into the same bridge: both route to the live window.
    assert "sid-new" in caught["drive_preview_callback"]({"action": "elements"})


def test_read_and_drive_route_to_most_recent_live_window(fake_bridge):
    cbs = _cron_preview_callbacks()
    out = json.loads(read_preview_tool(start=10, count=50, callback=cbs["read_preview_callback"]))
    assert out["sid"] == "sid-new"
    assert fake_bridge[-1][0] == "preview.read"
    assert fake_bridge[-1][1] == "sid-new"
    assert fake_bridge[-1][2] == {"start": 10, "count": 50}

    out = json.loads(drive_preview_tool(action="elements", callback=cbs["drive_preview_callback"]))
    assert out["sid"] == "sid-new"
    assert fake_bridge[-1][0] == "preview.act"
    assert fake_bridge[-1][1] == "sid-new"
    assert fake_bridge[-1][2]["action"] == "elements"


def test_no_live_window_returns_empty_without_blocking(fake_bridge, monkeypatch):
    monkeypatch.setattr(acb, "_sessions", {
        "sid-dead": {"session_key": "dead", "last_active": 300.0, "live_transport": False},
    }, raising=False)
    cbs = _cron_preview_callbacks()
    assert cbs["read_preview_callback"]() == ""
    assert cbs["drive_preview_callback"]({"action": "elements"}) == ""
    assert fake_bridge == []
    assert "timed out" in json.loads(
        read_preview_tool(callback=cbs["read_preview_callback"]))["error"]
    assert "timed out" in json.loads(
        drive_preview_tool(action="elements", callback=cbs["drive_preview_callback"]))["error"]
