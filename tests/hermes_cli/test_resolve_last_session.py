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


# ---------------------------------------------------------------------------
# cwd-scoped resume: -c prefers the last session in the current workspace.
# ---------------------------------------------------------------------------


class _WorkspaceAwareDB:
    """Fake SessionDB whose ``search_sessions`` honors ``workspace_key`` the
    same way the real ``_workspace_key_clause`` does: a row matches the key
    when its ``git_repo_root`` equals it, or (no repo root recorded) when its
    ``cwd`` is at or under it."""

    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def search_sessions(self, source=None, limit=20, workspace_key=None, **_kw):
        rows = [r for r in self._rows if r.get("source") == source] if source else list(self._rows)
        if workspace_key:
            key = workspace_key.rstrip("/")
            def _in_ws(r):
                grr = (r.get("git_repo_root") or "").rstrip("/")
                if grr:
                    return grr == key
                cwd = (r.get("cwd") or "").rstrip("/")
                return cwd == key or cwd.startswith(key + "/")
            rows = [r for r in rows if _in_ws(r)]
        rows.sort(
            key=lambda r: float(r.get("last_active") or r.get("started_at") or 0),
            reverse=True,
        )
        return rows[:limit]

    def close(self):
        self.closed = True


def test_resolve_last_session_prefers_workspace_session_over_global_mru(monkeypatch, tmp_path):
    # Two sessions: a globally-recent one in repo B, an older one in repo A.
    # CWD is repo A → -c must pick repo A's session, not the global MRU.
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    monkeypatch.chdir(repo_a)
    # Pretend both are git repo roots so _resolve_workspace_key returns the dir.
    monkeypatch.setattr(
        "hermes_cli.main.subprocess.run",
        lambda cmd, **kw: __import__("subprocess").CompletedProcess(
            cmd, 0, stdout=str(repo_a), stderr=""
        ),
    )

    rows = [
        {"id": "global_recent", "source": "cli", "started_at": 9000.0,
         "last_active": 9000.0, "cwd": str(repo_b), "git_repo_root": str(repo_b)},
        {"id": "repo_a_older", "source": "cli", "started_at": 1000.0,
         "last_active": 1000.0, "cwd": str(repo_a), "git_repo_root": str(repo_a)},
    ]
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _WorkspaceAwareDB(rows))
    assert _resolve_last_session("cli") == "repo_a_older"


def test_resolve_last_session_falls_back_to_global_when_workspace_empty(monkeypatch, tmp_path):
    # CWD has no matching session → fall back to the global MRU.
    repo_a = tmp_path / "repo-a"
    repo_a.mkdir()
    monkeypatch.chdir(repo_a)
    monkeypatch.setattr(
        "hermes_cli.main.subprocess.run",
        lambda cmd, **kw: __import__("subprocess").CompletedProcess(
            cmd, 0, stdout=str(repo_a), stderr=""
        ),
    )
    rows = [
        {"id": "elsewhere", "source": "cli", "started_at": 9000.0,
         "last_active": 9000.0, "cwd": "/other/repo", "git_repo_root": "/other/repo"},
    ]
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _WorkspaceAwareDB(rows))
    assert _resolve_last_session("cli") == "elsewhere"


def test_resolve_last_session_workspace_matches_cwd_subdir_without_git_root(monkeypatch, tmp_path):
    # No git repo root on the session (legacy row) — match via cwd prefix so a
    # session started in repo/src still resumes from repo/.
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        "hermes_cli.main.subprocess.run",
        lambda cmd, **kw: __import__("subprocess").CompletedProcess(
            cmd, 1, stdout="", stderr="not a repo"
        ),
    )
    rows = [
        {"id": "subdir_session", "source": "cli", "started_at": 500.0,
         "last_active": 500.0, "cwd": str(repo / "src"), "git_repo_root": ""},
    ]
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _WorkspaceAwareDB(rows))
    assert _resolve_last_session("cli") == "subdir_session"


def test_resolve_last_session_workspace_key_filters_by_source(monkeypatch, tmp_path):
    # A TUI session in repo A is newer than a CLI session in repo A; asking for
    # source="cli" must skip the TUI row even within the same workspace.
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        "hermes_cli.main.subprocess.run",
        lambda cmd, **kw: __import__("subprocess").CompletedProcess(
            cmd, 0, stdout=str(repo), stderr=""
        ),
    )
    rows = [
        {"id": "tui_recent", "source": "tui", "started_at": 9000.0,
         "last_active": 9000.0, "cwd": str(repo), "git_repo_root": str(repo)},
        {"id": "cli_older", "source": "cli", "started_at": 1000.0,
         "last_active": 1000.0, "cwd": str(repo), "git_repo_root": str(repo)},
    ]
    monkeypatch.setattr("hermes_state.SessionDB", lambda: _WorkspaceAwareDB(rows))
    assert _resolve_last_session("cli") == "cli_older"


def test_search_sessions_workspace_key_filters_at_sql_level(tmp_path, monkeypatch):
    # End-to-end: search_sessions(workspace_key=...) must filter by
    # git_repo_root, and fall back to cwd-prefix for rows without one.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state
    from pathlib import Path

    db = hermes_state.SessionDB(db_path=Path(tmp_path / "state.db"))
    try:
        # repo A: two sessions — one with git_repo_root, one legacy (cwd only).
        db.create_session("repo_a_rooted", source="cli", cwd=str(tmp_path / "repo-a"),
                          git_repo_root=str(tmp_path / "repo-a"))
        db.create_session("repo_a_legacy", source="cli", cwd=str(tmp_path / "repo-a" / "src"))
        # repo B: one session, globally most recent.
        db.create_session("repo_b", source="cli", cwd=str(tmp_path / "repo-b"),
                          git_repo_root=str(tmp_path / "repo-b"))
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (100.0, "repo_a_rooted"))
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (50.0, "repo_a_legacy"))
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (9000.0, "repo_b"))
            db._conn.commit()

        key = str(tmp_path / "repo-a")
        rows = db.search_sessions(source="cli", limit=10, workspace_key=key)
        ids = sorted(r["id"] for r in rows)
        assert ids == ["repo_a_legacy", "repo_a_rooted"]
        assert "repo_b" not in ids

        # No workspace_key → global MRU includes repo_b.
        all_rows = db.search_sessions(source="cli", limit=10)
        assert all_rows[0]["id"] == "repo_b"
    finally:
        db.close()


def test_resolve_last_session_real_db_prefers_workspace(monkeypatch, tmp_path):
    # End-to-end through the real SessionDB + _resolve_last_session: -c from
    # repo A picks repo A's session even though repo B is globally newer.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state
    from pathlib import Path

    repo_a = tmp_path / "repo-a"
    repo_a.mkdir()
    state_db = Path(tmp_path / "state.db")
    real_db = hermes_state.SessionDB
    db = real_db(db_path=state_db)
    try:
        db.create_session("repo_a", source="cli", cwd=str(repo_a), git_repo_root=str(repo_a))
        db.create_session("repo_b", source="cli", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (100.0, "repo_a"))
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (9000.0, "repo_b"))
            db._conn.commit()
    finally:
        db.close()

    monkeypatch.chdir(repo_a)
    monkeypatch.setattr(
        "hermes_cli.main.subprocess.run",
        lambda cmd, **kw: __import__("subprocess").CompletedProcess(
            cmd, 0, stdout=str(repo_a), stderr=""
        ),
    )
    monkeypatch.setattr("hermes_state.SessionDB", lambda: real_db(db_path=state_db))
    assert _resolve_last_session("cli") == "repo_a"
