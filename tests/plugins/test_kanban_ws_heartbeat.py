"""Regression for #118147: the kanban ``/events`` WebSocket must prove liveness while idle.

``stream_events`` wrote to the socket only when the tail returned events, so a quiet board was
indistinguishable from a half-open connection (NAT/conntrack timeout, VPN reconnect, laptop
sleep): the browser fires neither ``onclose`` nor ``onerror`` and keeps rendering its last
snapshot, counters included. The stream now emits a heartbeat frame while idle — the shipped
board bundle ignores any frame without ``events`` — and the send gives the server its only
chance to notice a peer that is gone instead of polling SQLite forever for a dead tab.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time as _real_time
from pathlib import Path

import pytest


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


class _ClockShim:
    """``plugin_api.time`` proxy with a settable ``monotonic``.

    Everything else (``time.time``, ``time.strftime``) delegates to the real module, so the
    fake clock stays confined to the heartbeat window arithmetic.
    """

    def __init__(self, now: float = 1000.0):
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(_real_time, name)


class _ScriptedWebSocket:
    def __init__(self, since: str = "0"):
        self.accepted = False
        self.sent: list[dict] = []
        self.query_params = {"since": since}

    async def accept(self):
        self.accepted = True

    async def receive(self):
        await asyncio.sleep(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=None):
        pass


class _TrackingConnection:
    """Stand-in for the tail's SQLite connection: one row set per ``execute``."""

    def __init__(self, rows_by_poll=None):
        self.rows_by_poll = list(rows_by_poll or [])
        self.execute_calls = 0
        self.close_calls = 0
        self._rows: list[dict] = []

    def execute(self, sql, params):
        self.execute_calls += 1
        index = self.execute_calls - 1
        self._rows = self.rows_by_poll[index] if index < len(self.rows_by_poll) else []
        return self

    def fetchall(self):
        return self._rows

    def close(self):
        self.close_calls += 1


def _drive_stream(monkeypatch, mod, clock, rounds, connect):
    """Drive ``stream_events`` deterministically instead of sleeping between polls.

    Each ``("idle", seconds)`` entry makes the ``receive()`` race time out — advancing the fake
    clock by *seconds* before the DB poll that follows — and ``("disconnect", 0)`` ends the loop
    the way a closed tab does.
    """
    script = list(rounds)
    calls = {"n": 0}

    async def _wait_for(awaitable, timeout):
        awaitable.close()  # the receive() coroutine is discarded, never awaited
        step = script[calls["n"]] if calls["n"] < len(script) else None
        calls["n"] += 1
        if step is None or step[0] == "disconnect":
            return {"type": "websocket.disconnect"}
        clock.now += step[1]
        raise asyncio.TimeoutError

    monkeypatch.setattr(mod.asyncio, "wait_for", _wait_for)
    monkeypatch.setattr(mod.kbc, "connect", connect)


@pytest.mark.asyncio
async def test_idle_stream_heartbeats_once_per_interval(monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    monkeypatch.setattr(mod, "_EVENT_HEARTBEAT_SECONDS", 15.0, raising=False)
    clock = _ClockShim(1000.0)
    monkeypatch.setattr(mod, "time", clock)

    conn = _TrackingConnection(rows_by_poll=[[], [], [], []])
    ws = _ScriptedWebSocket(since="12")

    _drive_stream(monkeypatch, mod, clock, [
        ("idle", 0.3),    # just connected: silence
        ("idle", 5.0),    # inside the window
        ("idle", 10.1),   # window elapsed — the stream proves liveness
        ("idle", 1.0),    # the heartbeat re-arms it
        ("idle", 15.1),   # elapsed again
        ("disconnect", 0),
    ], connect=lambda *, board=None: conn)

    await mod.stream_events(ws)

    assert ws.accepted
    assert [frame.get("type") for frame in ws.sent] == ["heartbeat", "heartbeat"]
    assert all("events" not in frame for frame in ws.sent)
    for frame in ws.sent:
        assert frame["cursor"] == 12
        assert isinstance(frame["server_time"], (int, float))
    assert conn.close_calls == 1


@pytest.mark.asyncio
async def test_events_frame_resets_the_heartbeat_window(monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    monkeypatch.setattr(mod, "_EVENT_HEARTBEAT_SECONDS", 15.0, raising=False)
    clock = _ClockShim(1000.0)
    monkeypatch.setattr(mod, "time", clock)

    event_row = {
        "id": 7,
        "task_id": "task-1",
        "run_id": None,
        "kind": "updated",
        "payload": '{"status": "running"}',
        "created_at": 1234,
    }
    conn = _TrackingConnection(rows_by_poll=[[event_row], [], []])
    ws = _ScriptedWebSocket(since="0")

    _drive_stream(monkeypatch, mod, clock, [
        ("idle", 0.3),   # events: the events frame wins, and re-arms the window
        ("idle", 10.0),  # inside the window measured from that frame
        ("idle", 6.0),   # past it — heartbeat
        ("disconnect", 0),
    ], connect=lambda *, board=None: conn)

    await mod.stream_events(ws)

    assert [frame.get("type") for frame in ws.sent] == [None, "heartbeat"]
    events_frame, heartbeat = ws.sent
    assert events_frame["cursor"] == 7
    assert events_frame["events"] == [{**event_row, "payload": {"status": "running"}}]
    assert heartbeat["cursor"] == 7
