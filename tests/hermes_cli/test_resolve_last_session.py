"""Verify `hermes -c` picks the session the user most recently used."""

from __future__ import annotations

from hermes_cli.main import _resolve_last_session


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
    monkeypatch.setattr("hermes_state.SessionDB", lambda **kw: real_db(db_path=state_db, **kw))
    assert _resolve_last_session("cli") == "repo_a"


def test_resolve_last_session_cli_continues_a_oneshot(monkeypatch, tmp_path):
    """`hermes -z … --resume latest` / `hermes -c` chain on the previous one-shot: its distinct `oneshot`
    source hides it from pickers but it is still CLI history (#112550)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state
    from pathlib import Path

    state_db = Path(tmp_path / "state.db")
    real_db = hermes_state.SessionDB
    db = real_db(db_path=state_db)
    try:
        db.create_session("interactive", source="cli")
        db.create_session("oneshot_run", source="oneshot")
        db.create_session("tui_chat", source="tui")
        with db._lock:
            for sid, started in (("interactive", 100.0), ("oneshot_run", 200.0), ("tui_chat", 300.0)):
                db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (started, sid))
            db._conn.commit()
    finally:
        db.close()

    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: None)
    monkeypatch.setattr("hermes_state.SessionDB", lambda **kw: real_db(db_path=state_db, **kw))
    assert _resolve_last_session("cli") == "oneshot_run"
    assert _resolve_last_session("tui") == "tui_chat"
