"""A session id becomes a filename under ``sessions/``; an id that is not a single safe path
component must never reach the filesystem.

``_remove_session_files`` builds ``sessions_dir / f"{id}.json"`` (and ``.jsonl``,
``session_<id>.json``), so an imported or API-created row carrying ``../x`` deleted a file outside
``sessions/`` on delete/prune. The import validator now rejects such ids at the door, and the sweep
skips file cleanup for any that are already in the DB (the row itself is still deleted).
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "a\\b", "..", "C:escape", "name:stream", "bad\x00id"])
def test_import_rejects_path_unsafe_ids(db, bad_id):
    result = db.import_sessions([{"id": bad_id, "messages": []}])

    assert result["ok"] is False
    assert result["errors"][0]["error"] == "session id is not a safe filename"
    assert db.get_session(bad_id) is None


def test_import_accepts_glob_metacharacters(db):
    # The dump sweep glob.escape()s the id, so these are only odd names, not a widening risk.
    result = db.import_sessions([{"id": "sess-[ab]-*?", "messages": []}])

    assert result["ok"] is True
    assert db.get_session("sess-[ab]-*?") is not None


@pytest.mark.parametrize("bad_id", ["../keep-me", "sub/../../keep-me", "C:keep-me", "name:stream"])
def test_delete_sweep_never_touches_files_outside_sessions_dir(db, tmp_path, bad_id):
    sessions_dir = tmp_path / "sessions"
    (sessions_dir / "sub").mkdir(parents=True)
    outside = tmp_path / "keep-me.json"
    outside.write_text("{}", encoding="utf-8")
    db.create_session(session_id=bad_id, source="cli")
    db.create_session(session_id="s1", source="cli")
    (sessions_dir / "s1.json").write_text("{}", encoding="utf-8")

    assert db.delete_sessions([bad_id, "s1"], sessions_dir=sessions_dir) == 2

    assert db.get_session(bad_id) is None
    assert outside.exists()
    assert not (sessions_dir / "s1.json").exists()
