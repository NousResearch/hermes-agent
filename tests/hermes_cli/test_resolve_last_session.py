"""Verify `hermes -c` picks the session the user most recently used."""

from __future__ import annotations

from hermes_cli.main import _resolve_last_session


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def search_sessions(self, source=None, limit=20, **_kw):
        rows = [r for r in self._rows if r.get("source") == source] if source else list(self._rows)
        return rows[:limit]

    def close(self):
        self.closed = True


def test_resolve_last_session_prefers_last_active_over_started_at(monkeypatch):
    # `search_sessions` returns in started_at DESC order, but the most recently
    # *touched* session may have been started earlier. -c should pick by
    # last_active so closing the active session and typing `hermes -c` resumes
    # that one, not an older-but-newer-started session from another window.
    rows = [
        {
            "id": "new_started_old_active",
            "source": "cli",
            "started_at": 1000.0,
            "last_active": 100.0,
        },
        {
            "id": "old_started_recently_active",
            "source": "cli",
            "started_at": 500.0,
            "last_active": 999.0,
        },
    ]

    fake_db = _FakeDB(rows)
    monkeypatch.setattr("hermes_state.SessionDB", lambda: fake_db)

    assert _resolve_last_session("cli") == "old_started_recently_active"
    assert fake_db.closed


def test_search_sessions_exposes_last_active_column(tmp_path, monkeypatch):
    # End-to-end: the actual SessionDB must surface a last_active column so
    # _resolve_last_session's sort works. A previous bug had last_active=None
    # on every row because search_sessions used `SELECT *` with no computed
    # column, silently breaking the -c resume behavior.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state

    from pathlib import Path

    db = hermes_state.SessionDB(db_path=Path(tmp_path / "state.db"))
    try:
        db.create_session("s_started_later", source="cli")
        db.create_session("s_active_later", source="cli")
        # Force started_at ordering so the test is deterministic regardless
        # of how quickly the two inserts land.
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (2000.0, "s_started_later"))
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (1000.0, "s_active_later"))
            db._conn.commit()

        db.append_message("s_active_later", role="user", content="hi")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (3000.0, "s_active_later"),
            )
            db._conn.commit()

        rows = db.search_sessions(source="cli", limit=5)
        ids = {r["id"]: r.get("last_active") for r in rows}

        assert ids["s_started_later"] == 2000.0
        assert ids["s_active_later"] == 3000.0
    finally:
        db.close()


def test_resolve_last_session_returns_none_when_empty(monkeypatch):
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _FakeDB([]))
    assert _resolve_last_session("cli") is None


def test_resolve_last_session_falls_back_to_started_at(monkeypatch):
    # When last_active is missing entirely (legacy row), fall back to
    # started_at so the helper still picks the newest session.
    rows = [
        {"id": "older", "source": "cli", "started_at": 10.0},
        {"id": "newer", "source": "cli", "started_at": 20.0},
    ]
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _FakeDB(rows))
    assert _resolve_last_session("cli") == "newer"
