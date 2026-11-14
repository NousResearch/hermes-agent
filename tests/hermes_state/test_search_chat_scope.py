"""Chat-origin filters and provenance on the session search / browse / title store APIs."""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _session(db, sid, chat_id, key=None, title=None, content=None):
    db.create_session(session_id=sid, source="signal")
    db._conn.execute("UPDATE sessions SET chat_id=?, chat_type='group', display_name=?, session_key=?, title=? "
                     "WHERE id=?", (chat_id, f"Group {chat_id}", key, title, sid))
    if content:
        db.append_message(sid, role="user", content=content)
    db._conn.commit()


def test_search_rows_carry_session_origin(db):
    _session(db, "s1", "chat-a", key="k-a", content="Origin tagged Python question")

    row = db.search_messages("Python")[0]

    assert (row["chat_id"], row["chat_type"], row["display_name"], row["session_key"]) == (
        "chat-a", "group", "Group chat-a", "k-a")


@pytest.mark.parametrize("scope", [{"chat_id": "chat-a"}, {"session_key": "k-a"}])
def test_search_filters_by_chat_origin(db, scope):
    _session(db, "same", "chat-a", key="k-a", content="Needle context isolation")
    _session(db, "foreign", "chat-b", key="k-b", content="Needle context isolation")

    assert [r["session_id"] for r in db.search_messages("Needle", **scope)] == ["same"]


def test_session_key_wins_over_chat_id(db):
    _session(db, "same_user", "chat-a", key="k-a-user-1", content="Needle per user")
    _session(db, "other_user", "chat-a", key="k-a-user-2", content="Needle per user")

    assert [r["session_id"] for r in db.search_messages("Needle", session_key="k-a-user-1", chat_id="chat-a")] == [
        "same_user"]


def test_bounded_browse_filters_by_chat_origin(db):
    _session(db, "same", "chat-a", key="k-a", content="hi")
    _session(db, "foreign", "chat-b", key="k-b", content="hi")

    rows = db.list_recent_sessions_bounded(limit=5, session_key="k-a")

    assert [r["id"] for r in rows] == ["same"]
    assert rows[0]["chat_id"] == "chat-a" and rows[0]["display_name"] == "Group chat-a"


def test_title_resolution_can_be_scoped_to_one_chat(db):
    _session(db, "same", "chat-a", key="k-a", title="Plan")
    _session(db, "foreign", "chat-b", key="k-b", title="Plan #2")

    assert db.resolve_session_by_title("Plan") == "foreign"  # unscoped: latest continuation wins
    assert db.resolve_session_by_title("Plan", session_key="k-a") == "same"
    assert db.resolve_session_by_title("Plan", chat_id="chat-b") == "foreign"
    assert db.resolve_session_by_title("Plan", session_key="k-none") is None
