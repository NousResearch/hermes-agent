"""Regression: kanban events WS must notice client disconnect on an idle board.

Before the fix (#77833), ``stream_events`` only awaited ``asyncio.sleep``
between DB polls, so a disconnect was detected solely when ``send_json``
raised — which never happens on a board with no new events. Every closed
dashboard tab therefore left a zombie poll task querying SQLite forever.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
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
