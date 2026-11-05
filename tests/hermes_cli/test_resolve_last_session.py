"""Verify `hermes -c` picks the session the user most recently used."""

from __future__ import annotations

from hermes_cli.main import _resolve_last_session


def test_search_sessions_exposes_last_active_column(tmp_path, monkeypatch):
    # End-to-end: the store must surface last_active and order by MRU.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    from persistence.conversation.sessions import SessionWindow
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_holder import raw_holder
    from tests.fixtures.storage_seed import seed_messages, seed_session

    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "s_started_later", "cli")
        seed_session(store, "s_active_later", "cli")
        # Force started_at ordering so the test is deterministic regardless
        # of how quickly the two inserts land.
        raw_holder(store)._execute_write(lambda conn: conn.executemany(
            "UPDATE sessions SET started_at=? WHERE id=?",
            ((2000.0, "s_started_later"), (1000.0, "s_active_later")),
        ))

        seed_messages(store, "s_active_later", {"role": "user", "content": "hi"})
        raw_holder(store)._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET timestamp=? WHERE session_id=?",
            (3000.0, "s_active_later"),
        ))

        rows = store.sessions.search_sessions(sources=("cli",), window=SessionWindow(limit=5))
        ids = {r.session.session_id: r.last_active for r in rows}

        assert ids["s_started_later"] == 2000.0
        assert ids["s_active_later"] == 3000.0
        assert rows[0].session.session_id == "s_active_later"




# ---------------------------------------------------------------------------
# cwd-scoped resume: -c prefers the last session in the current workspace.
# ---------------------------------------------------------------------------


def test_resolve_last_session_real_db_prefers_workspace(monkeypatch, tmp_path):
    # End-to-end through the real store + _resolve_last_session: -c from
    # repo A picks repo A's session even though repo B is globally newer.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_holder import raw_holder
    from tests.fixtures.storage_seed import seed_session

    repo_a = tmp_path / "repo-a"
    repo_a.mkdir()
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "repo_a", "cli", cwd=str(repo_a), git_repo_root=str(repo_a))
        seed_session(store, "repo_b", "cli", cwd="/other/repo-b", git_repo_root="/other/repo-b")
        # One message each: ``-c`` never resumes an empty launch stub.
        raw_holder(store)._execute_write(lambda conn: conn.executemany(
            "UPDATE sessions SET started_at=?, message_count=1 WHERE id=?",
            ((100.0, "repo_a"), (9000.0, "repo_b")),
        ))

    monkeypatch.chdir(repo_a)
    monkeypatch.setattr(
        "hermes_cli.main.subprocess.run",
        lambda cmd, **kw: __import__("subprocess").CompletedProcess(
            cmd, 0, stdout=str(repo_a), stderr=""
        ),
    )
    monkeypatch.setattr("hermes_cli.main.get_hermes_home", lambda: tmp_path)
    assert _resolve_last_session("cli") == "repo_a"


def test_resolve_last_session_cli_continues_a_oneshot(monkeypatch, tmp_path):
    """`hermes -z … --resume latest` / `hermes -c` chain on the previous one-shot: its distinct `oneshot`
    source hides it from pickers but it is still CLI history (#112550)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_holder import raw_holder
    from tests.fixtures.storage_seed import seed_session

    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "interactive", "cli")
        seed_session(store, "oneshot_run", "oneshot")
        seed_session(store, "tui_chat", "tui")
        raw_holder(store)._execute_write(lambda conn: conn.executemany(
            "UPDATE sessions SET started_at=?, message_count=1 WHERE id=?",
            ((100.0, "interactive"), (200.0, "oneshot_run"), (300.0, "tui_chat")),
        ))

    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: None)
    # The lookup opens the active home's store by explicit path (a served-profile process has no
    # single default home), so the home is the seam — same as the workspace test above.
    monkeypatch.setattr("hermes_cli.main.get_hermes_home", lambda: tmp_path)
    assert _resolve_last_session("cli") == "oneshot_run"
    assert _resolve_last_session("tui") == "tui_chat"
