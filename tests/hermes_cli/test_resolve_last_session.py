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


# ---------------------------------------------------------------------------
# Cross-surface resume: `hermes -c` also continues the workspace's Desktop chat.
# ---------------------------------------------------------------------------


def _cross_surface_db(tmp_path, monkeypatch, rows):
    """Install a real SessionDB with the given (id, source, started_at) rows."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import hermes_state
    from pathlib import Path

    state_db = Path(tmp_path / "state.db")
    real_db = hermes_state.SessionDB
    db = real_db(db_path=state_db)
    try:
        for sid, source, started in rows:
            db.create_session(sid, source=source, cwd=str(tmp_path))
            with db._lock:
                db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (started, sid))
                db._conn.commit()
    finally:
        db.close()

    monkeypatch.setattr("hermes_state.SessionDB", lambda **kw: real_db(db_path=state_db, **kw))
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: str(tmp_path))
    return real_db, state_db


def test_cli_continue_reaches_a_newer_desktop_session(monkeypatch, tmp_path):
    """A Desktop conversation newer than the CLI's is what `hermes -c` continues.

    Both surfaces write into the same workspace of the same state.db, and the Desktop
    session is the one every picker shows, so a bare `-c` that skipped it looked like
    the CLI had "lost" the recent conversation.
    """
    from hermes_cli.main import _latest_session_id

    _cross_surface_db(tmp_path, monkeypatch, [
        ("cli_session", "cli", 100.0),
        ("desktop_session", "desktop", 9000.0),
    ])
    assert _latest_session_id(use_tui=False) == "desktop_session"


def test_cli_continue_prefers_the_cli_family_over_an_older_desktop_session(monkeypatch, tmp_path):
    """Control: the CLI family still wins when it is the newer one.

    Without this, a fix that simply appended `desktop` to the source list would pass the
    test above while regressing ordinary `-c`.
    """
    from hermes_cli.main import _latest_session_id

    _cross_surface_db(tmp_path, monkeypatch, [
        ("cli_session", "cli", 9000.0),
        ("desktop_session", "desktop", 100.0),
    ])
    assert _latest_session_id(use_tui=False) == "cli_session"


def test_cli_continue_keeps_chaining_a_oneshot_over_a_desktop_session(monkeypatch, tmp_path):
    """Control: #112550's `hermes -z … --resume latest` chain is not displaced.

    A one-shot run is CLI history; when it is the newest thing in the workspace it must
    still be what `-c` picks, even with a Desktop session present.
    """
    from hermes_cli.main import _latest_session_id

    _cross_surface_db(tmp_path, monkeypatch, [
        ("oneshot_run", "oneshot", 9000.0),
        ("desktop_session", "desktop", 5000.0),
    ])
    assert _latest_session_id(use_tui=False) == "oneshot_run"


def test_cli_continue_ignores_a_desktop_session_from_another_workspace(monkeypatch, tmp_path):
    """Control: another project's Desktop chat must not win a bare `-c`.

    The UI lookup is workspace-only, so it cannot fall back to the global MRU and drag
    an unrelated workspace's conversation into this one. The second project is a sibling
    directory, not a subdirectory — a workspace key matches its own subtree by design.
    """
    from hermes_cli.main import _latest_session_id

    real_db, state_db = _cross_surface_db(tmp_path, monkeypatch, [
        ("cli_session", "cli", 100.0),
    ])
    other = tmp_path.parent / (tmp_path.name + "-other-project")
    other.mkdir(exist_ok=True)
    db = real_db(db_path=state_db)
    try:
        db.create_session("other_desktop", source="desktop", cwd=str(other))
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (9999.0, "other_desktop"))
            db._conn.commit()
    finally:
        db.close()

    assert _latest_session_id(use_tui=False) == "cli_session"


def test_tui_continue_does_not_adopt_the_desktop_session(monkeypatch, tmp_path):
    """Control: a TUI launch keeps its own order and never hijacks the Desktop window.

    TUI and Desktop are two live surfaces of the same transport family; `hermes --tui -c`
    stealing the Desktop's conversation would move it out from under that window.
    """
    from hermes_cli.main import _latest_session_id

    _cross_surface_db(tmp_path, monkeypatch, [
        ("cli_session", "cli", 100.0),
        ("tui_chat", "tui", 200.0),
        ("desktop_session", "desktop", 9000.0),
    ])
    assert _latest_session_id(use_tui=True) == "tui_chat"
