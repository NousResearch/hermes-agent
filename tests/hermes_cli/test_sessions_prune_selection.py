"""`sessions prune --newer-than --source` selects by recent activity, not creation date."""

import sys
import time
from pathlib import Path

import pytest

from hermes_cli import main
from hermes_state import SessionDB

DAY = 86400


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    now = time.time()
    # (id, source, age in days of the latest message); every session started 100 days ago.
    rows = (("cli-recent", "cli", 1), ("cli-stale", "cli", 10), ("tg-recent", "telegram", 1))
    with SessionDB() as db:
        for session_id, source, idle_days in rows:
            active = now - idle_days * DAY
            db.create_session(session_id, source)
            db.append_message(session_id, "user", f"hello {session_id}", timestamp=active - 60)
            db.append_message(session_id, "assistant", f"reply {session_id}", timestamp=active)
            db.end_session(session_id, "done")
            db._conn.execute(
                "UPDATE sessions SET started_at=?, last_activity_at=?, ended_at=? WHERE id=?",
                (now - 100 * DAY, active, active, session_id),
            )
        db._conn.commit()
    return home


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "prune", "--newer-than", "2d", "--source", "cli", *args])
    return main.main()


def _snapshot(db, ids):
    return {i: (db.get_session(i), db.get_messages(i)) for i in ids}


def test_prune_newer_than_selects_recently_active_old_session_of_source(store, tmp_path, monkeypatch, capsys):
    ids = ("cli-recent", "cli-stale", "tg-recent")
    with SessionDB() as db:
        assert Path(db.db_path).resolve().is_relative_to(tmp_path.resolve())
        before = _snapshot(db, ids)
    assert all(session and messages for session, messages in before.values())

    _run(monkeypatch, "--dry-run")
    out = capsys.readouterr().out
    assert "cli-recent" in out
    assert "cli-stale" not in out and "tg-recent" not in out
    with SessionDB() as db:
        assert _snapshot(db, ids) == before

    # This fixture is the sole writer; bypass the unrelated host-wide holder inventory.
    _run(monkeypatch, "--yes", "--force")
    assert "Pruned 1 session(s)." in capsys.readouterr().out
    with SessionDB() as db:
        assert db.get_session("cli-recent") is None
        assert db.get_messages("cli-recent") == []
        after = _snapshot(db, ids)
    assert {i: after[i] for i in ("cli-stale", "tg-recent")} == {i: before[i] for i in ("cli-stale", "tg-recent")}
