"""Regression tests for the TUI gateway's ``session.list`` handler.

Reported during TUI v2 blitz retest: the ``/resume`` modal inside a TUI
session only surfaced ``tui``/``cli`` rows, hiding telegram sessions users
could still resume directly via ``hermes --tui --resume <id>``.

The fix widens the picker to a curated allowlist of user-facing sources
(tui/cli + chat adapters) while still filtering internal/system sources.
"""

from __future__ import annotations

from tui_gateway import server


class _StubDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[dict] = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.rows)


def _call(limit: int = 20):
    return server.handle_request({
        "id": "1",
        "method": "session.list",
        "params": {"limit": limit},
    })


def test_session_list_includes_telegram_but_filters_internal_sources(monkeypatch):
    rows = [
        {"id": "tui-1", "source": "tui", "started_at": 9},
        {"id": "tool-1", "source": "tool", "started_at": 8},
        {"id": "tg-1", "source": "telegram", "started_at": 7},
        {"id": "acp-1", "source": "acp", "started_at": 6},
        {"id": "cli-1", "source": "cli", "started_at": 5},
    ]
    db = _StubDB(rows)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    resp = _call(limit=10)
    sessions = resp["result"]["sessions"]
    ids = [s["id"] for s in sessions]

    assert "tg-1" in ids and "tui-1" in ids and "cli-1" in ids, ids
    assert "tool-1" not in ids and "acp-1" not in ids, ids


def test_session_list_fetches_wider_window_before_filtering(monkeypatch):
    db = _StubDB([{"id": "x", "source": "cli", "started_at": 1}])
    monkeypatch.setattr(server, "_get_db", lambda: db)

    _call(limit=10)

    assert len(db.calls) == 1
    assert db.calls[0].get("source") is None, db.calls[0]
    assert db.calls[0].get("limit") == 100, db.calls[0]


def test_session_list_preserves_ordering_after_filter(monkeypatch):
    rows = [
        {"id": "newest", "source": "telegram", "started_at": 5},
        {"id": "internal", "source": "tool", "started_at": 4},
        {"id": "middle", "source": "tui", "started_at": 3},
        {"id": "oldest", "source": "discord", "started_at": 1},
    ]
    monkeypatch.setattr(server, "_get_db", lambda: _StubDB(rows))

    resp = _call()
    ids = [s["id"] for s in resp["result"]["sessions"]]

    assert ids == ["newest", "middle", "oldest"]
