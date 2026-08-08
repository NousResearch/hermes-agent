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

from tui_gateway import server


class _StubDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[dict] = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.rows)


def _call(limit: int | None = None):
    params: dict = {}
    if limit is not None:
        params["limit"] = limit
    return server.handle_request({
        "id": "1",
        "method": "session.list",
        "params": params,
    })


def test_session_list_surfaces_all_user_facing_sources(monkeypatch):
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
    db = _StubDB(rows)
    monkeypatch.setattr(server, "_get_db", lambda: db)

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


def test_session_list_clamps_negative_limit(monkeypatch):
    """Negative limit must not reverse-slice (``rows[:-n]``) the result set."""
    rows = [
        {"id": f"s{i}", "source": "cli", "started_at": i}
        for i in range(10)
    ]
    db = _StubDB(rows)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    resp = _call(limit=-3)
    sessions = resp["result"]["sessions"]
    # Clamped to 1 — never drops the tail via negative slicing.
    assert len(sessions) == 1
    # Stub returns rows in insertion order; first surviving row is s0.
    assert sessions[0]["id"] == "s0"


def test_session_list_clamps_huge_limit(monkeypatch):
    """Huge limits must not request unbounded DB fetches."""
    rows = [{"id": "only", "source": "cli", "started_at": 1}]
    db = _StubDB(rows)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    resp = _call(limit=99999)
    assert "error" not in resp
    # fetch_limit = max(clamped_limit * 2, 200) with clamp=500 → 1000
    assert db.calls[0]["limit"] == 1000


def test_session_list_invalid_limit_falls_back(monkeypatch):
    rows = [{"id": "a", "source": "cli", "started_at": 1}]
    db = _StubDB(rows)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    resp = server.handle_request({
        "id": "1",
        "method": "session.list",
        "params": {"limit": "not-a-number"},
    })
    assert "error" not in resp
    assert len(resp["result"]["sessions"]) == 1


