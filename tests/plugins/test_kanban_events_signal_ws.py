"""Kanban dashboard plugin: ``/events`` WS ``view=signal`` mode (#86808).

Covers the strict, redacted read-model contract layered onto the
existing live-events WebSocket:

* legacy (``view`` absent) behaviour is untouched — a malformed
  ``since`` still silently coerces to 0
* ``view=signal`` rejects unknown/duplicate query params and a
  malformed/negative ``since`` by closing the upgrade instead of
  coercing
* signal-mode event frames carry only the allowlisted fields (no
  ``payload``) plus a ``generation`` marker
* a cursor below the current retention floor gets an explicit
  ``reset`` frame instead of being treated as "nothing happened"
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_signal_ws_test", plugin_file,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _QueryParams:
    """Minimal stand-in for Starlette's multidict ``QueryParams``."""

    def __init__(self, items):
        self._items = list(items)

    def get(self, key, default=None):
        for k, v in self._items:
            if k == key:
                return v
        return default

    def multi_items(self):
        return list(self._items)


class _FakeWebSocket:
    """Accepts (unless rejected), replays canned client messages, then
    reports disconnect; records everything sent/closed."""

    def __init__(self, query_items, messages=()):
        self.query_params = _QueryParams(query_items)
        self.accepted = False
        self.closed_with = None
        self.sent: list[dict] = []
        self._messages = list(messages) + [{"type": "websocket.disconnect"}]

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return self._messages.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=None):
        self.closed_with = code


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _insert_event(conn, task_id, kind="task_created", payload='{"secret": "nope"}'):
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, NULL, ?, ?, ?)",
        (task_id, kind, payload, int(time.time())),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


@pytest.mark.asyncio
async def test_legacy_mode_still_coerces_malformed_since(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    ws = _FakeWebSocket([("since", "not-a-number")])

    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert ws.accepted
    assert ws.closed_with is None
    assert ws.sent == []  # no events on a fresh board; legacy path never rejects


@pytest.mark.asyncio
async def test_signal_mode_rejects_unknown_param(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    ws = _FakeWebSocket([("view", "signal"), ("board", "default"), ("color", "blue")])

    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert not ws.accepted
    assert ws.closed_with == mod.http_status.WS_1008_POLICY_VIOLATION


@pytest.mark.asyncio
async def test_signal_mode_rejects_duplicate_param(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    ws = _FakeWebSocket([("view", "signal"), ("board", "default"), ("board", "default")])

    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert not ws.accepted
    assert ws.closed_with == mod.http_status.WS_1008_POLICY_VIOLATION


@pytest.mark.asyncio
async def test_signal_mode_requires_board(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    ws = _FakeWebSocket([("view", "signal")])

    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert not ws.accepted
    assert ws.closed_with == mod.http_status.WS_1008_POLICY_VIOLATION


@pytest.mark.asyncio
async def test_signal_mode_rejects_nonexistent_board(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    ws = _FakeWebSocket([("view", "signal"), ("board", "no-such-board")])

    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert not ws.accepted
    assert ws.closed_with == mod.http_status.WS_1008_POLICY_VIOLATION


@pytest.mark.asyncio
async def test_signal_mode_rejects_malformed_since_instead_of_coercing(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)
    ws = _FakeWebSocket([("view", "signal"), ("board", "default"), ("since", "-1")])

    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert not ws.accepted
    assert ws.closed_with == mod.http_status.WS_1008_POLICY_VIOLATION


@pytest.mark.asyncio
async def test_signal_mode_frame_is_redacted_and_versioned(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)

    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(conn, title="do it", board="default")
        # create_task already logs its own "created" event; clear it so
        # the only surviving event is the one this test controls.
        conn.execute("DELETE FROM task_events")
        conn.commit()
        _insert_event(conn, task_id, payload='{"leak": "should-not-appear"}')
    finally:
        conn.close()

    ws = _FakeWebSocket(
        [("view", "signal"), ("board", "default")],
        messages=[{"type": "websocket.receive", "text": ""}],
    )
    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert ws.accepted
    assert len(ws.sent) == 1
    frame = ws.sent[0]
    assert "generation" in frame and "cursor" in frame
    assert len(frame["events"]) == 1
    event = frame["events"][0]
    assert set(event.keys()) == {"id", "task_id", "run_id", "kind", "created_at"}
    assert "payload" not in event


@pytest.mark.asyncio
async def test_signal_mode_gap_below_retention_floor_sends_reset(kanban_home, monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod, "_ws_upgrade_authorized", lambda ws: True)

    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(conn, title="do it", board="default")
        # create_task already logs its own "created" event; clear it so
        # the ids below are fully under this test's control.
        conn.execute("DELETE FROM task_events")
        conn.commit()
        stale_id = _insert_event(conn, task_id, kind="stale")
        # Simulate retention (kanban_db.gc_events) reclaiming the old
        # event: the client's stored cursor (stale_id - 1) now points
        # below the surviving history's floor.
        conn.execute("DELETE FROM task_events WHERE id = ?", (stale_id,))
        conn.commit()
        fresh_id = _insert_event(conn, task_id, kind="fresh")
    finally:
        conn.close()

    ws = _FakeWebSocket(
        [("view", "signal"), ("board", "default"), ("since", str(stale_id - 1))],
        messages=[{"type": "websocket.receive", "text": ""}],
    )
    await asyncio.wait_for(mod.stream_events(ws), timeout=5)

    assert ws.accepted
    assert ws.sent[0].get("reset") is True
    assert ws.sent[0]["generation"] == fresh_id
    # The surviving event still arrives — nothing is silently dropped,
    # the client is just told a gap happened.
    assert any(f.get("events") for f in ws.sent[1:])
