"""The preview-action and tour probes of a local viewer's desktop window.

A renderer that never answers costs one short probe, then the call fails fast; for the preview
action that lasts a cooldown, after which one more probe is sent. A renderer that answered once
is always asked again, with the full deadline.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway import local_desktop_ui
from tests.gateway.test_local_desktop_ui import _Fanout

_UNAVAILABLE = "No Hermes Desktop window answered"


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setattr(local_desktop_ui, "_preview_bridge", {})
    monkeypatch.setattr(local_desktop_ui, "_tour_bridge", {})
    monkeypatch.setattr(local_desktop_ui, "_PREVIEW_ACTION_PROBE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(local_desktop_ui, "_PREVIEW_ACTION_TIMEOUT_S", 0.3)
    monkeypatch.setattr(local_desktop_ui, "_TOUR_PROBE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(local_desktop_ui, "_TOUR_TIMEOUT_S", 0.3)
    clock = [1000.0]
    monkeypatch.setattr(local_desktop_ui, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    requests, silent = [], set()
    stream = SimpleNamespace(fanout=_Fanout(requests, silent), publish=lambda *_a, **_kw: None)
    authority = SimpleNamespace(sessions={"canon": SimpleNamespace(event_stream=stream)})
    turn = SimpleNamespace(_approval_owner=(authority, "canon", 1))
    agent = SimpleNamespace(_interrupt_requested=False)
    return SimpleNamespace(turn=turn, agent=agent, requests=requests, silent=silent, clock=clock)


def _act(viewer):
    return local_desktop_ui._preview_action(viewer.turn, viewer.agent, {"action": "click"})


def _sent(viewer, method):
    return sum(1 for frame in viewer.requests if frame["method"] == method)


def test_an_unanswered_preview_probe_fails_fast_until_the_cooldown_ends(viewer):
    viewer.silent.add("preview.act")
    assert _UNAVAILABLE in _act(viewer)
    assert _UNAVAILABLE in _act(viewer)
    assert _sent(viewer, "preview.act") == 1

    viewer.clock[0] += local_desktop_ui._PREVIEW_ACTION_REPROBE_COOLDOWN_S + 1
    viewer.silent.discard("preview.act")
    assert json.loads(_act(viewer))["success"] is True
    assert _sent(viewer, "preview.act") == 2


def test_an_answered_preview_bridge_is_always_asked_again(viewer):
    assert json.loads(_act(viewer))["success"] is True
    viewer.silent.add("preview.act")
    assert _UNAVAILABLE in _act(viewer)
    assert _UNAVAILABLE in _act(viewer)
    assert _sent(viewer, "preview.act") == 3


def test_an_interrupted_preview_probe_leaves_the_bridge_unknown(viewer):
    viewer.silent.add("preview.act")
    viewer.agent._interrupt_requested = True
    assert _UNAVAILABLE in _act(viewer)
    viewer.agent._interrupt_requested = False
    assert _UNAVAILABLE in _act(viewer)
    assert _sent(viewer, "preview.act") == 2


def test_an_answered_tour_keeps_being_asked_after_a_miss(viewer):
    def tour():
        return local_desktop_ui._tour(viewer.turn, viewer.agent, {"action": "start"})

    assert json.loads(tour())["success"] is True
    viewer.silent.add("tour")
    assert _UNAVAILABLE in tour()
    assert _UNAVAILABLE in tour()
    assert _sent(viewer, "tour") == 3
