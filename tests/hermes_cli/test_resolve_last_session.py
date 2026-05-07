"""Verify `hermes -c` picks the session the user most recently used."""

from __future__ import annotations

from hermes_cli.main import _resolve_last_session


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def search_sessions(self, source=None, limit=20, **_kw):
        rows = [r for r in self._rows if r.get("source") == source] if source else list(self._rows)
        rows.sort(
            key=lambda r: float(r.get("last_active") or r.get("started_at") or 0),
            reverse=True,
        )
        return rows[:limit]

    def close(self):
        self.closed = True


def test_resolve_last_session_prefers_last_active_over_started_at(monkeypatch):
    # `search_sessions` should return in MRU order, so -c can trust row 0.
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
    # End-to-end: SessionDB must surface last_active and order by MRU.
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
        assert rows[0]["id"] == "s_active_later"
    finally:
        db.close()


def test_resolve_last_session_returns_none_when_empty(monkeypatch):
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _FakeDB([]))
    assert _resolve_last_session("cli") is None


def test_resolve_last_session_closes_db_on_search_error(monkeypatch):
    class _FailingDB:
        def __init__(self):
            self.closed = False

        def search_sessions(self, source=None, limit=20, **_kw):
            raise RuntimeError("boom")

        def close(self):
            self.closed = True

    db = _FailingDB()
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    assert _resolve_last_session("cli") is None
    assert db.closed is True


def test_resolve_last_session_falls_back_to_started_at(monkeypatch):
    # When last_active is missing entirely (legacy row), fall back to
    # started_at so the helper still picks the newest session.
    rows = [
        {"id": "older", "source": "cli", "started_at": 10.0},
        {"id": "newer", "source": "cli", "started_at": 20.0},
    ]
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _FakeDB(rows))
    assert _resolve_last_session("cli") == "newer"


def test_resolve_last_session_not_limited_to_newest_started_20(tmp_path, monkeypatch):
    # Regression: when sampling by started_at, -c could miss the true MRU if
    # it was older than the newest 20 started sessions.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state

    from pathlib import Path

    state_db = Path(tmp_path / "state.db")
    real_session_db = hermes_state.SessionDB
    db = real_session_db(db_path=state_db)
    try:
        for i in range(25):
            sid = f"s_{i:02d}"
            db.create_session(sid, source="cli")
            with db._lock:
                db._conn.execute(
                    "UPDATE sessions SET started_at=? WHERE id=?",
                    (10_000.0 - i, sid),
                )
                db._conn.commit()

        target = "s_24"
        db.append_message(target, role="user", content="latest activity")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (20_000.0, target),
            )
            db._conn.commit()
    finally:
        db.close()

    monkeypatch.setattr("hermes_state.SessionDB", lambda: real_session_db(db_path=state_db))
    assert _resolve_last_session("cli") == target
