"""``hermes sessions recover`` keeps session outcomes (#138653): the full copy, the partial-salvage orphan
cleanup, and the page-level lost_and_found lane all list child tables by name."""

import sqlite3

from hermes_state import SessionDB
from hermes_cli.session_lost_and_found import _copy_direct_tables, stub_missing_parent_sessions
from hermes_cli.session_recovery import _cleanup_partial_orphans, recover_session_database


def _source(path, *, orphan=False):
    with SessionDB(db_path=path) as db:
        for sid in ("s", "gone"):
            db.create_session(sid, source="cli")
            db.append_message(sid, role="user", content="q")
            db.record_session_outcome(sid, signal="merged", source="github", recorder="cli", ref=f"pr/{sid}")
        if orphan:  # the parent row is what corruption took
            db._conn.execute("PRAGMA foreign_keys=OFF")
            db._conn.execute("DELETE FROM messages WHERE session_id = 'gone'")
            db._conn.execute("DELETE FROM sessions WHERE id = 'gone'")
            db._conn.commit()


def _refs(path):
    with sqlite3.connect(path) as conn:
        return sorted(r[0] for r in conn.execute("SELECT ref FROM session_outcomes"))


def test_recover_copies_outcomes(tmp_path):
    _source(tmp_path / "state.db")
    report = recover_session_database(tmp_path / "state.db", tmp_path / "out.db", work_dir=tmp_path)

    assert report["complete"] is True and report["verified"] is True
    assert report["copy"]["session_outcomes"]["copied_rows"] == 2
    assert _refs(tmp_path / "out.db") == ["pr/gone", "pr/s"]


def test_partial_cleanup_removes_outcomes_of_unsalvageable_sessions(tmp_path):
    _source(tmp_path / "state.db", orphan=True)
    conn = sqlite3.connect(tmp_path / "state.db", isolation_level=None)
    try:
        result = _cleanup_partial_orphans(conn)
        assert result["session_outcomes_removed"] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
    assert _refs(tmp_path / "state.db") == ["pr/s"]


def test_lost_and_found_lane_copies_outcomes_and_stubs_their_missing_session(tmp_path):
    _source(tmp_path / "lf.db", orphan=True)
    SessionDB(db_path=tmp_path / "out.db").close()
    lf = sqlite3.connect(tmp_path / "lf.db", isolation_level=None)
    dest = sqlite3.connect(tmp_path / "out.db", isolation_level=None)
    try:
        dest.execute("PRAGMA foreign_keys=OFF")
        assert _copy_direct_tables(lf, dest)["session_outcomes"] == 2
        assert stub_missing_parent_sessions(dest)["sessions_stubbed"] == 1
        assert dest.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        lf.close()
        dest.close()
    assert _refs(tmp_path / "out.db") == ["pr/gone", "pr/s"]
