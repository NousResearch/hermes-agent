"""The kanban events stream starts at the board's tail unless the client names a cursor.

``/events`` opened without ``since`` used to start at cursor 0 and replay every ``task_events``
row, 200 per 300 ms poll — and the Desktop invalidates the whole board on every frame, so each
open, board switch and reconnect became a storm of full-board refetches lasting as long as the
board's history (#81537). The ``/board`` snapshot already holds the past; a client that wants
history asks for it with ``since``.
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
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_kanban_events_tail_test", plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Ws:
    def __init__(self, query_params: dict[str, str]):
        self.query_params = query_params
        self.sent: list[dict] = []
        self._disconnect = asyncio.Event()

    async def accept(self):
        pass

    async def receive(self):
        await self._disconnect.wait()
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=None):
        pass


def _event(event_id: int, kind: str = "created") -> dict:
    return {"id": event_id, "task_id": "t", "run_id": None, "kind": kind, "payload": None, "created_at": event_id}


class _Conn:
    """A board with events 1..450 already on disk; a fresh event 451 lands on the second poll."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.thread_ids: list[int] = []
        self._rows: list[dict] = []

    def execute(self, sql, params):
        self.calls.append((sql, tuple(params)))
        self.thread_ids.append(threading.get_ident())
        if "MAX(id)" in sql:
            self._rows = [{"m": 450}]
            return self
        (cursor,) = params
        polls = sum(1 for seen_sql, _ in self.calls if "MAX(id)" not in seen_sql)
        on_disk = [_event(i) for i in range(1, 451)] + ([_event(451, "spawned")] if polls >= 2 else [])
        self._rows = [row for row in on_disk if row["id"] > cursor][:200]
        return self

    def fetchall(self):
        return self._rows

    def close(self):
        pass


async def _drive(monkeypatch, mod, ws, conn, polls: int) -> None:
    """Run ``stream_events`` for ``polls`` DB polls, then let the client disconnect."""
    waits = 0

    async def _wait_for(awaitable, timeout):
        nonlocal waits
        waits += 1
        awaitable.close()
        if waits <= polls:
            raise asyncio.TimeoutError
        return {"type": "websocket.disconnect"}

    monkeypatch.setattr(mod.asyncio, "wait_for", _wait_for)
    monkeypatch.setattr(mod.kbc, "connect", lambda *, board=None: conn)
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    await mod.stream_events(ws)


def _poll_cursors(conn: _Conn) -> list[tuple]:
    return [params for sql, params in conn.calls if "MAX(id)" not in sql]


@pytest.mark.asyncio
async def test_without_since_the_stream_starts_at_the_boards_tail(monkeypatch):
    mod = _load_plugin_module()
    conn, ws = _Conn(), _Ws({})
    await _drive(monkeypatch, mod, ws, conn, polls=2)
    assert _poll_cursors(conn)[0] == (450,), "the first poll must start where the board is, not at 0"
    # Only the event that landed after the connect reaches the client — never the 450 old rows.
    assert [event["id"] for frame in ws.sent for event in frame["events"]] == [451]
    assert ws.sent[-1]["cursor"] == 451
    assert len(set(conn.thread_ids)) == 1  # the tail read and the polls share the tail's one thread


@pytest.mark.asyncio
@pytest.mark.parametrize("since", ["0", "100"], ids=["replay-all", "catch-up"])
async def test_an_explicit_since_replays_from_exactly_there(monkeypatch, since):
    mod = _load_plugin_module()
    conn, ws = _Conn(), _Ws({"since": since})
    await _drive(monkeypatch, mod, ws, conn, polls=1)
    assert not any("MAX(id)" in sql for sql, _ in conn.calls), "a client with a cursor never pays the tail read"
    assert _poll_cursors(conn)[0] == (int(since),)
    assert ws.sent[0]["events"][0]["id"] == int(since) + 1


@pytest.mark.asyncio
async def test_garbage_since_is_treated_as_no_cursor(monkeypatch):
    mod = _load_plugin_module()
    conn, ws = _Conn(), _Ws({"since": "latest"})
    await _drive(monkeypatch, mod, ws, conn, polls=1)
    assert _poll_cursors(conn)[0] == (450,)
