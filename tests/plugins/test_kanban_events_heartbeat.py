"""Heartbeat on the kanban events WebSocket (#118147).

A quiet board used to send nothing at all, so a half-open connection (NAT
timeout, VPN switch, laptop sleep) was undetectable and the board silently
froze. ``stream_events`` now sends a heartbeat frame after
``_EVENT_HEARTBEAT_SECONDS`` of silence; the dashboard's watchdog uses any
frame as a liveness signal.

The tests drive ``stream_events`` with a fake WebSocket and a scripted
``_EventTail.poll()``. The module's ``time`` is replaced by a fake clock that
advances per poll (never slept); asyncio's own clock is untouched.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_heartbeat_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(monkeypatch, poll_outcomes, clock_step=0.0):
    """Run ``stream_events`` against fakes.

    poll_outcomes: ``(cursor, events)`` tuples served by successive polls; once
    exhausted the tail raises ``WebSocketDisconnect`` to end the stream.
    clock_step: seconds the fake monotonic clock advances per poll.
    Returns ``[(poll_count, frame), ...]`` for every frame sent, where
    poll_count is the number of polls completed when the frame went out.
    """
    mod = _load_plugin_module()
    records = []
    clock = [1000.0]
    polls = [0]

    class FakeWebSocket:
        query_params = {}

        async def accept(self):
            pass

        async def receive(self):
            await asyncio.Event().wait()  # never a client message

        async def send_json(self, payload):
            records.append((polls[0], payload))

        async def close(self, code=1000):
            pass

    class ScriptedTail:
        def __init__(self, board):
            pass

        async def poll(self, cursor):
            idx = polls[0]
            polls[0] += 1
            clock[0] += clock_step
            if idx < len(poll_outcomes):
                return poll_outcomes[idx]
            raise mod.WebSocketDisconnect()

        async def shutdown(self):
            pass

    monkeypatch.setattr(mod, "_EventTail", ScriptedTail)
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    monkeypatch.setattr(mod, "_ws_board", lambda param: "default")
    monkeypatch.setattr(mod, "_int_param", lambda ws, name: 0)
    monkeypatch.setattr(mod, "_EVENT_POLL_SECONDS", 0.0)
    monkeypatch.setattr(
        mod, "time", types.SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000.0),
    )
    asyncio.run(mod.stream_events(FakeWebSocket()))
    return records


def test_heartbeat_after_interval_of_silence(monkeypatch):
    # 5 s per poll: the third idle poll reaches 15 s of silence.
    records = _run(monkeypatch, [(0, [])] * 3, clock_step=5.0)
    assert records == [
        (3, {"type": "heartbeat", "cursor": 0, "server_time": 1700000000.0}),
    ]


def test_no_frame_before_interval(monkeypatch):
    records = _run(monkeypatch, [(0, [])] * 2, clock_step=5.0)
    assert records == []


def test_events_frame_resets_the_silence_timer(monkeypatch):
    # Without the reset the heartbeat would fire at poll 3 (15 s after connect);
    # the events frame at poll 2 pushes it to poll 5 (15 s after that frame).
    ev = {"id": 1, "task_id": "t1"}
    outcomes = [(0, []), (5, [ev]), (5, []), (5, []), (5, [])]
    records = _run(monkeypatch, outcomes, clock_step=5.0)
    assert records == [
        (2, {"events": [ev], "cursor": 5}),
        (5, {"type": "heartbeat", "cursor": 5, "server_time": 1700000000.0}),
    ]


def test_heartbeat_spacing_on_a_quiet_board(monkeypatch):
    # 40 idle polls at 5 s = 200 s: a heartbeat exactly every 3rd poll.
    records = _run(monkeypatch, [(0, [])] * 40, clock_step=5.0)
    assert all(frame["type"] == "heartbeat" for _, frame in records)
    assert [n for n, _ in records] == list(range(3, 41, 3))
