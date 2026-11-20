"""Regression tests for the TUI gateway's ``session.list`` handler.

History:
- The original implementation hardcoded an allow-list of known gateway
  sources (``tui, cli, telegram, discord, slack, ...``). New or unlisted
  sources (``acp``, ``webhook``, user-defined ``HERMES_SESSION_SOURCE``
  values, newly-added platforms) were silently dropped from the resume
  picker — users reported "lots of sessions are missing from browse
  but exist in .hermes/sessions."
- The handler now deny-lists only the internal/noisy source ``tool``
  (sub-agent runs) and surfaces every other source to the picker.
- The default ``limit`` raised from 20 to 200 so longer-running users
  can scroll through their history without hitting an artificial cap.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

from tui_gateway import server


class _StubStore:
    """Storage handle double: ``sessions.list_sessions`` answers ``rows`` as listing entries."""

    def __init__(self, rows, home: Path):
        self.rows = rows
        self.calls: list[tuple] = []
        self.scope = SimpleNamespace(profile_dir=home)
        self.sessions = SimpleNamespace(list_sessions=self._list_sessions)

    def _list_sessions(self, query, window, *, compact_rows=False):
        self.calls.append((query, window, compact_rows))
        return tuple(SimpleNamespace(as_list_row=lambda row=row: dict(row)) for row in self.rows)


def _call(limit: int | None = None):
    params: dict = {}
    if limit is not None:
        params["limit"] = limit
    return server.handle_request({
        "id": "1",
        "method": "session.list",
        "params": params,
    })


def test_session_list_surfaces_all_user_facing_sources(monkeypatch, tmp_path):
    """acp / webhook / custom sources should all appear; only ``tool`` is hidden."""
    rows = [
        {"id": "tui-1", "source": "tui", "started_at": 9},
        {"id": "tool-1", "source": "tool", "started_at": 8},
        {"id": "tg-1", "source": "telegram", "started_at": 7},
        {"id": "acp-1", "source": "acp", "started_at": 6},
        {"id": "cli-1", "source": "cli", "started_at": 5},
        {"id": "webhook-1", "source": "webhook", "started_at": 4},
        {"id": "custom-1", "source": "my-custom-source", "started_at": 3},
    ]
    store = _StubStore(rows, tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: store)
    monkeypatch.setattr(server, "_persistence_for", lambda _db: contextlib.nullcontext(store))

    resp = _call(limit=10)
    ids = [s["id"] for s in resp["result"]["sessions"]]

    # Every human-facing source — including previously-hidden acp, webhook,
    # and custom sources — must surface in the picker now.
    assert "tg-1" in ids
    assert "tui-1" in ids
    assert "cli-1" in ids
    assert "acp-1" in ids, "acp sessions were being hidden by the old allow-list"
    assert "webhook-1" in ids, "webhook sessions were being hidden by the old allow-list"
    assert "custom-1" in ids, "custom HERMES_SESSION_SOURCE values were being hidden"

    # Only internal sub-agent runs stay hidden.
    assert "tool-1" not in ids


