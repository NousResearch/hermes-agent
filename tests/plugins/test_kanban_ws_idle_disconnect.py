"""Regressions for the kanban events WebSocket connection lifecycle.

Before the fix (#77833), ``stream_events`` only awaited ``asyncio.sleep``
between DB polls, so a disconnect was detected solely when ``send_json``
raised — which never happens on a board with no new events. Every closed
dashboard tab therefore left a zombie poll task querying SQLite forever.

The event tail must also reuse one SQLite connection instead of opening and
closing the last WAL connection on every idle poll. On Windows that repeatedly
deletes and recreates ``kanban.db-wal`` and ``kanban.db-shm``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path

import pytest


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_ws_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _IdleDisconnectingWebSocket:
    """Accepts, then reports a client disconnect on the first receive()."""

    def __init__(self):
        self.accepted = False
        self.sent: list[dict] = []
        self.query_params: dict[str, str] = {}
        self.receive_calls = 0

    async def accept(self):
        self.accepted = True

    async def receive(self):
        self.receive_calls += 1
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=None):
        pass


class _PollingWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent: list[dict] = []
        self.query_params: dict[str, str] = {}
        self._disconnect = asyncio.Event()

    async def accept(self):
        self.accepted = True

    async def receive(self):
        await self._disconnect.wait()
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=None):
        pass


class _TrackingConnection:
    def __init__(self, rows_by_poll=None, on_execute=None):
        self.rows_by_poll = list(rows_by_poll or [])
        self.on_execute = on_execute
        self.execute_calls = 0
        self.close_calls = 0
        self.thread_ids: list[int] = []
        self._rows: list[dict] = []

    def execute(self, sql, params):
        self.execute_calls += 1
        self.thread_ids.append(threading.get_ident())
        if self.on_execute is not None:
            self.on_execute()
        poll_index = self.execute_calls - 1
        self._rows = (
            self.rows_by_poll[poll_index]
            if poll_index < len(self.rows_by_poll)
            else []
        )
        return self

    def fetchall(self):
        return self._rows

    def close(self):
        self.close_calls += 1
        self.thread_ids.append(threading.get_ident())


@pytest.mark.asyncio
async def test_stream_events_exits_on_idle_disconnect(monkeypatch, tmp_path):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)

    ws = _IdleDisconnectingWebSocket()

    # The disconnect must terminate the handler even though the board is idle
    # and no event is ever sent. Before the fix this call never returned
    # (the loop only slept between polls), so bound it with a timeout.
    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert ws.accepted
    assert ws.receive_calls == 1
    assert ws.sent == []  # returned before any poll, no zombie loop


@pytest.mark.asyncio
async def test_stream_events_reuses_connection_and_closes_after_disconnect(
    monkeypatch,
):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)

    event_row = {
        "id": 7,
        "task_id": "task-1",
        "run_id": None,
        "kind": "updated",
        "payload": '{"status": "running"}',
        "created_at": 1234,
    }
    conn = _TrackingConnection(rows_by_poll=[[], [event_row]])
    connect_threads: list[int] = []

    def _connect(*, board=None):
        connect_threads.append(threading.get_ident())
        return conn

    monkeypatch.setattr(mod.kanban_db, "connect", _connect)

    wait_calls = 0

    async def _poll_twice_then_disconnect(awaitable, timeout):
        nonlocal wait_calls
        wait_calls += 1
        awaitable.close()
        if wait_calls <= 2:
            raise asyncio.TimeoutError
        return {"type": "websocket.disconnect"}

    monkeypatch.setattr(mod.asyncio, "wait_for", _poll_twice_then_disconnect)
    ws = _PollingWebSocket()

    await mod.stream_events(ws)

    assert ws.accepted
    assert len(connect_threads) == 1
    assert conn.execute_calls == 2
    assert conn.close_calls == 1
    assert len(set(connect_threads + conn.thread_ids)) == 1
    assert ws.sent == [{
        "events": [{
            "id": 7,
            "task_id": "task-1",
            "run_id": None,
            "kind": "updated",
            "payload": {"status": "running"},
            "created_at": 1234,
        }],
        "cursor": 7,
    }]


@pytest.mark.asyncio
async def test_stream_events_closes_connection_when_cancelled(monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)

    loop = asyncio.get_running_loop()
    first_fetch_done = asyncio.Event()
    conn = _TrackingConnection(
        on_execute=lambda: loop.call_soon_threadsafe(first_fetch_done.set),
    )
    connect_threads: list[int] = []

    def _connect(*, board=None):
        connect_threads.append(threading.get_ident())
        return conn

    monkeypatch.setattr(mod.kanban_db, "connect", _connect)

    real_wait_for = asyncio.wait_for
    wait_calls = 0

    async def _poll_once_then_wait(awaitable, timeout):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(mod.asyncio, "wait_for", _poll_once_then_wait)
    ws = _PollingWebSocket()
    task = asyncio.create_task(mod.stream_events(ws))

    await real_wait_for(first_fetch_done.wait(), timeout=5)
    task.cancel()
    await task

    assert len(connect_threads) == 1
    assert conn.execute_calls == 1
    assert conn.close_calls == 1
    assert len(set(connect_threads + conn.thread_ids)) == 1
