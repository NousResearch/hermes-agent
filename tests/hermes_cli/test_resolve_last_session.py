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


# ---------------------------------------------------------------------------
# Family fallback (-c across cli/webui/tui): workspace-first ordering.
# A preferred-source session in another repository must never beat an
# alternate-interface session in the current workspace (#47214).
# ---------------------------------------------------------------------------


def _real_db_fixture(tmp_path, monkeypatch):
    """Seed a real SessionDB in tmp_path and point _resolve_last_session at it.

    Returns (db, repo_a) with repo_a created on disk; callers create rows with
    git_repo_root/cwd and SQL-bump started_at for deterministic MRU ordering.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    repo_a = tmp_path / "repo-a"
    repo_a.mkdir(exist_ok=True)
    state_db = tmp_path / "state.db"

    import hermes_state

    real_db = hermes_state.SessionDB
    db = real_db(db_path=state_db)

    def _stamp(sid, at):
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (at, sid))
            db._conn.commit()

    def _use_repo_a():
        monkeypatch.chdir(repo_a)
        monkeypatch.setattr(
            "hermes_cli.main.subprocess.run",
            lambda cmd, **kw: __import__("subprocess").CompletedProcess(
                cmd, 0, stdout=str(repo_a), stderr=""
            ),
        )
        monkeypatch.setattr("hermes_state.SessionDB", lambda: real_db(db_path=state_db))

    return db, repo_a, _stamp, _use_repo_a


def test_family_prefers_current_workspace_over_foreign_preferred_source(tmp_path, monkeypatch):
    """Reviewer case: local webui in /repo-a must beat a foreign cli in /repo-b."""
    db, repo_a, _stamp, _use_repo_a = _real_db_fixture(tmp_path, monkeypatch)
    try:
        db.create_session("local_webui", source="webui", cwd=str(repo_a), git_repo_root=str(repo_a))
        db.create_session("foreign_cli", source="cli", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        _stamp("local_webui", 1000.0)
        _stamp("foreign_cli", 9000.0)
        _use_repo_a()
        from hermes_cli.main import _resolve_last_session

        assert _resolve_last_session(("cli", "webui", "tui")) == "local_webui"
    finally:
        db.close()


def test_family_tui_first_variant_prefers_local_webui(tmp_path, monkeypatch):
    db, repo_a, _stamp, _use_repo_a = _real_db_fixture(tmp_path, monkeypatch)
    try:
        db.create_session("local_webui", source="webui", cwd=str(repo_a), git_repo_root=str(repo_a))
        db.create_session("foreign_tui", source="tui", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        _stamp("local_webui", 1000.0)
        _stamp("foreign_tui", 9000.0)
        _use_repo_a()
        from hermes_cli.main import _resolve_last_session

        assert _resolve_last_session(("tui", "webui", "cli")) == "local_webui"
    finally:
        db.close()


def test_family_interface_preference_within_workspace(tmp_path, monkeypatch):
    """Inside one workspace, launching CLI prefers the cli session even when a
    webui session is newer (interface preference preserved inside the scope)."""
    db, repo_a, _stamp, _use_repo_a = _real_db_fixture(tmp_path, monkeypatch)
    try:
        db.create_session("ws_cli", source="cli", cwd=str(repo_a), git_repo_root=str(repo_a))
        db.create_session("ws_webui", source="webui", cwd=str(repo_a), git_repo_root=str(repo_a))
        _stamp("ws_cli", 100.0)
        _stamp("ws_webui", 5000.0)
        _use_repo_a()
        from hermes_cli.main import _resolve_last_session

        assert _resolve_last_session(("cli", "webui", "tui")) == "ws_cli"
        assert _resolve_last_session(("tui", "webui", "cli")) == "ws_webui"
    finally:
        db.close()


def test_family_falls_back_globally_when_workspace_empty(tmp_path, monkeypatch):
    db, repo_a, _stamp, _use_repo_a = _real_db_fixture(tmp_path, monkeypatch)
    try:
        db.create_session("global_cli", source="cli", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        db.create_session("global_webui", source="webui", cwd="/third/repo-c", git_repo_root="/third/repo-c")
        _stamp("global_cli", 9000.0)
        _stamp("global_webui", 7000.0)
        _use_repo_a()
        from hermes_cli.main import _resolve_last_session

        assert _resolve_last_session(("cli", "webui", "tui")) == "global_cli"
    finally:
        db.close()


def test_family_never_picks_automation_sessions(tmp_path, monkeypatch):
    """A cron session in the current workspace is not a resume candidate."""
    db, repo_a, _stamp, _use_repo_a = _real_db_fixture(tmp_path, monkeypatch)
    try:
        db.create_session("ws_cron", source="cron", cwd=str(repo_a), git_repo_root=str(repo_a))
        _stamp("ws_cron", 9000.0)
        _use_repo_a()
        from hermes_cli.main import _resolve_last_session

        assert _resolve_last_session(("cli", "webui", "tui")) is None
        assert _resolve_last_session("cli") is None
    finally:
        db.close()


def test_latest_session_id_uses_interface_family(tmp_path, monkeypatch):
    db, repo_a, _stamp, _use_repo_a = _real_db_fixture(tmp_path, monkeypatch)
    try:
        db.create_session("older_cli", source="cli", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        db.create_session("newer_tui", source="tui", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        _stamp("older_cli", 100.0)
        _stamp("newer_tui", 9000.0)
        _use_repo_a()
        from hermes_cli.main import _latest_session_id

        assert _latest_session_id(use_tui=False) == "older_cli"  # cli preferred in CLI mode
        assert _latest_session_id(use_tui=True) == "newer_tui"  # tui preferred in TUI mode
    finally:
        db.close()
