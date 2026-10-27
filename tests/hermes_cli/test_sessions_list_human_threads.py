"""``hermes sessions`` lists the conversations you resume, not the automation around them.

A cron tick writes a ``local-<sha>`` row every run (most stay empty), and a compacted thread's
title can live on a middle segment of its lineage (and a root that rotated before its first flush is
empty while its tip is not). The list must show each human lineage once, as the tip you resume,
under its title, ranked by its last message; bare ``hermes sessions`` off a terminal prints that
same list.
"""

import io
import sqlite3
import time
from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd


@pytest.fixture(autouse=True)
def _no_workspace_scope(monkeypatch):
    """These rows carry no cwd; this file is about lineages and sources, not the workspace scope."""
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: None)


@pytest.fixture
def home(tmp_path, monkeypatch):
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session
    msg = {"role": "user", "content": "hi"}
    with open_persistence(profile_scope(home=tmp_path)) as store:
        for i in range(5):  # newest-started rows: empty cron ticks
            seed_session(store, f"local-cron{i}", "cron")
        seed_session(store, "local-root", "local")
        seed_messages(store, "local-root", msg)
        store.sessions.end("local-root", "compression")
        seed_session(store, "mid", "local", parent_session_id="local-root", title="Fix sessions UX")
        seed_messages(store, "mid", msg)
        store.sessions.end("mid", "compression")
        seed_session(store, "tip", "local", parent_session_id="mid")
        seed_messages(store, "tip", msg)
        seed_session(store, "sub", "subagent", parent_session_id="tip", title="Subagent: task")
        seed_messages(store, "sub", msg)
        seed_session(store, "old-cli", "cli", title="Old chat")
        seed_messages(store, "old-cli", msg)
        seed_session(store, "local-empty", "local", end_reason="compression")  # rotated before any flush
        seed_session(store, "saved-tip", "local", parent_session_id="local-empty", title="Saved thread")
        seed_messages(store, "saved-tip", msg)
        seed_session(store, "tool-run", "tool")
        seed_messages(store, "tool-run", msg)
        store.sessions.set_pinned("local-cron0", True)
    now = time.time()
    db = sqlite3.connect(tmp_path / "state.db")
    # started_at says the cron ticks are newest; the last message says the lineage tip is.
    for sid, started, last in [("local-root", now - 9000, now - 8900), ("mid", now - 8000, now - 7900),
                               ("tip", now - 7000, now - 60), ("sub", now - 50, now - 10),
                               ("old-cli", now - 6000, now - 5000), ("local-empty", now - 4500, None),
                               ("saved-tip", now - 4000, now - 3000), ("tool-run", now - 40, now - 20)]:
        db.execute("UPDATE sessions SET started_at=?, last_activity_at=NULL WHERE id=?", (started, sid))
        db.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (last, sid))
    db.execute("UPDATE sessions SET started_at=? WHERE source='cron'", (now - 30,))
    db.commit()
    db.close()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _rows(out):
    return [line.split() for line in out.splitlines()[2:] if line.strip()]


def test_list_shows_lineage_tips_with_inherited_titles_not_cron(home, capsys):
    from persistence_assembly import open_persistence_reader
    with open_persistence_reader(home / "state.db") as store:
        sessions_cmd._cmd_list(Namespace(limit=4, source=None, workspace=None), persistence=store)
        rows = _rows(capsys.readouterr().out)
        # No cron tick, tool run, subagent or closed segment; the empty root still lists as its tip.
        assert [r[-1] for r in rows] == ["tip", "saved-tip", "old-cli"]
        assert rows[0][:3] == ["Fix", "sessions", "UX"]  # the middle segment's title, on the tip
        sessions_cmd._cmd_list(Namespace(limit=50, source=None, workspace=None, all=True), persistence=store)
        listed = {r[-1] for r in _rows(capsys.readouterr().out)}
        assert {"tool-run", "local-cron1"} <= listed  # --all is every source, tool included
        sessions_cmd._cmd_pinned(Namespace(source=None, json=True), persistence=store)
        assert "local-cron0" in capsys.readouterr().out  # a pin outlives the automated-source filter


def test_bare_sessions_off_a_terminal_prints_the_list(home, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO())  # not a terminal
    sessions_cmd.cmd_sessions(Namespace(sessions_action=None))
    assert [r[-1] for r in _rows(capsys.readouterr().out)] == ["tip", "saved-tip", "old-cli"]


def test_deep_untitled_lineages_list_under_sqlite_variable_limit(home, capsys, monkeypatch):
    """Six 180-deep untitled chains put 1080 ancestors in the tip-title lookup; a build whose
    variable limit is the historical 999 must still list them (one IN per ``_id_chunks`` batch)."""
    from persistence.sqlite import _holder
    from persistence_assembly import open_persistence_reader
    db = sqlite3.connect(home / "state.db")
    now = time.time()
    for c in range(6):
        ids = [f"deep{c}-{i:03d}" for i in range(181)]
        for i, sid in enumerate(ids):
            db.execute("INSERT INTO sessions (id, source, parent_session_id, started_at, end_reason, message_count)"
                       " VALUES (?, 'local', ?, ?, ?, 1)",
                       (sid, ids[i - 1] if i else None, now - 2000 + i, None if i == 180 else "compression"))
    db.commit()
    db.close()
    connect = _holder._connect_tracked_db

    def low_limit(*a, **k):
        conn = connect(*a, **k)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        return conn
    monkeypatch.setattr(_holder, "_connect_tracked_db", low_limit)
    with open_persistence_reader(home / "state.db") as store:
        sessions_cmd._cmd_list(Namespace(limit=20, source=None, workspace=None), persistence=store)
    assert {f"deep{c}-180" for c in range(6)} <= {r[-1] for r in _rows(capsys.readouterr().out)}
