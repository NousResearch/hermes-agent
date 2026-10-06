"""list_gateway_sessions after /resume onto an older row of the same routing key.

/resume ends the current row and reopens an older one without touching its
started_at. The listing must still report the live row for that key.
"""

import pytest

from hermes_state import SessionDB

KEY = "agent:main:telegram:dm:c1"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _resumed_older_row(db):
    for sid, started in (("old", 1_700_000_000.0), ("new", 1_700_000_500.0)):
        db.create_session(sid, "telegram", session_key=KEY, chat_id="c1", chat_type="dm")
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (started, sid))
            db._conn.commit()
    db.end_session("old", "session_switch")
    # /resume old: end the current row, reopen the older target.
    db.end_session("new", "session_switch")
    db.reopen_session("old")


@pytest.mark.parametrize("active_only", [True, False])
def test_resumed_older_row_is_the_listed_mapping_for_its_key(db, active_only):
    _resumed_older_row(db)

    rows = [r for r in db.list_gateway_sessions(active_only=active_only) if r["session_key"] == KEY]

    assert [(r["id"], r["ended_at"]) for r in rows] == [("old", None)]


def test_key_with_no_live_row_still_lists_its_newest_row(db):
    _resumed_older_row(db)
    db.end_session("old", "session_reset")

    assert [r["id"] for r in db.list_gateway_sessions(active_only=True) if r["session_key"] == KEY] == []
    assert [r["id"] for r in db.list_gateway_sessions(active_only=False) if r["session_key"] == KEY] == ["new"]
