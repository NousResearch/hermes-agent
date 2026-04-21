"""Regression test for the TUI gateway's ``session.list`` handler.

Reported during the TUI v2 blitz retest: the ``/resume`` modal inside a
TUI session only surfaced ``tui``/``cli`` rows — telegram/discord/whatsapp
sessions stayed hidden even though the user could still paste the id
directly into ``hermes --tui --resume <id>`` and get a working session.

The fix removes the adapter-kind filter so every session the DB surfaces
appears in the picker, sorted by ``started_at`` like before.
"""

from __future__ import annotations

import types

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


def test_session_list_does_not_filter_by_source(monkeypatch):
    rows = [
        {"id": "tui-1", "source": "tui", "title": "a", "preview": "", "started_at": 3, "message_count": 1},
        {"id": "tg-1", "source": "telegram", "title": "b", "preview": "", "started_at": 2, "message_count": 1},
        {"id": "cli-1", "source": "cli", "title": "c", "preview": "", "started_at": 1, "message_count": 1},
    ]
    db = _StubDB(rows)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    resp = _call(limit=10)

    assert "result" in resp, resp
    assert len(db.calls) == 1
    assert db.calls[0].get("source") is None, db.calls[0]
    assert db.calls[0].get("limit") == 10

    kinds = [s["source"] for s in resp["result"]["sessions"]]
    assert "telegram" in kinds and "tui" in kinds and "cli" in kinds, kinds


def test_session_list_preserves_ordering(monkeypatch):
    rows = [
        {"id": "newest", "source": "telegram", "title": "", "preview": "", "started_at": 5, "message_count": 1},
        {"id": "middle", "source": "tui", "title": "", "preview": "", "started_at": 3, "message_count": 1},
        {"id": "oldest", "source": "discord", "title": "", "preview": "", "started_at": 1, "message_count": 1},
    ]
    monkeypatch.setattr(server, "_get_db", lambda: _StubDB(rows))

    resp = _call()
    ids = [s["id"] for s in resp["result"]["sessions"]]

    assert ids == ["newest", "middle", "oldest"]


def test_session_list_surfaces_missing_fields_as_empty(monkeypatch):
    rows = [{"id": "bare", "source": "whatsapp"}]
    monkeypatch.setattr(server, "_get_db", lambda: _StubDB(rows))

    sess = _call()["result"]["sessions"][0]

    assert sess == {
        "id": "bare",
        "title": "",
        "preview": "",
        "started_at": 0,
        "message_count": 0,
        "source": "whatsapp",
    }
