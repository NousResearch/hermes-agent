"""#84870: session_reset chains project to the live tip in the session list."""

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _chain(db: SessionDB, *, reason: str, stamp_reset_from: bool = False):
    base = time.time() - 200
    db.create_session("root", source="cli")
    db.set_session_title("root", "Creative Skills Overview")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.set_session_title("tip", "Live tip after /new")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = ?, "
        "message_count = 2, last_activity_at = ? WHERE id = 'root'",
        (base, base + 10, reason, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 4, last_activity_at = ? "
        "WHERE id = 'tip'",
        (base + 80, base + 80),
    )
    if stamp_reset_from:
        db._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = 'tip'",
            ('{"_reset_from": "root"}',),
        )
    db._conn.commit()


def test_session_reset_list_projects_to_live_tip(db):
    _chain(db, reason="session_reset")

    rows = db.list_sessions_rich(order_by_last_active=True)
    ids = [s["id"] for s in rows]
    assert ids == ["tip"], ids
    assert rows[0]["title"] == "Live tip after /new"
    assert rows[0].get("_lineage_root_id") == "root"


def test_new_session_end_reason_also_projects(db):
    _chain(db, reason="new_session")

    rows = db.list_sessions_rich(order_by_last_active=True)
    assert [s["id"] for s in rows] == ["tip"]


def test_marked_reset_child_stays_its_own_conversation(db):
    """A `_reset_from` child is a new user-visible conversation, not a tip."""
    _chain(db, reason="session_reset", stamp_reset_from=True)

    rows = db.list_sessions_rich(order_by_last_active=True)
    by_id = {s["id"]: s for s in rows}
    assert set(by_id) == {"root", "tip"}
    assert by_id["root"]["title"] == "Creative Skills Overview"
    assert by_id["root"].get("_lineage_root_id") is None
    assert by_id["tip"]["title"] == "Live tip after /new"


def test_compression_projection_still_walks(db):
    _chain(db, reason="compression")

    rows = db.list_sessions_rich(order_by_last_active=True)
    assert [s["id"] for s in rows] == ["tip"]


def test_same_key_legacy_reset_keeps_both_rows(db):
    """Gateway DM reset children with the same session_key stay listable."""
    db.create_session("parent", source="telegram", session_key="agent:main:telegram:dm:lane")
    db.end_session("parent", "session_reset")
    db.create_session(
        "child",
        source="telegram",
        session_key="agent:main:telegram:dm:lane",
        parent_session_id="parent",
    )
    listed = {row["id"] for row in db.list_sessions_rich(source="telegram")}
    assert listed == {"parent", "child"}
    assert db.get_compression_tip("parent") == "parent"
    assert db.get_list_surface_tip("parent") == "parent"
