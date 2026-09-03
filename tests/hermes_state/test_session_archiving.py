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


def _compression_pair(db: SessionDB):
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def test_archiving_compression_tip_archives_projected_root(db):
    _compression_pair(db)

    assert db.set_session_archived("tip", True) is True

    assert db.get_session("root")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["tip"]


def test_unarchiving_compression_tip_unarchives_projected_root(db):
    _compression_pair(db)
    db.set_session_archived("tip", True)

    assert db.set_session_archived("tip", False) is True

    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


def test_bulk_archive_matches_unended_title_and_preserves_title(db):
    db.create_session("open", source="cli")
    db.set_session_title("open", "Purple Elephant Test")
    db.create_session("pinned-open", source="cli")
    db.set_session_title("pinned-open", "Purple Elephant Pinned")
    db.set_session_pinned("pinned-open", True)

    assert db.list_prune_candidates(title_like="Purple Elephant") == []
    assert [
        row["id"]
        for row in db.list_archive_candidates(title_like="Purple Elephant")
    ] == ["open"]
    assert {
        row["id"]
        for row in db.list_archive_candidates(
            title_like="Purple Elephant", include_pinned=True
        )
    } == {"open", "pinned-open"}
    assert db.prune_sessions(
        older_than_days=None, title_like="Purple Elephant"
    ) == 0
    with pytest.raises(TypeError):
        db.prune_sessions(
            older_than_days=None,
            title_like="Purple Elephant",
            include_unended=True,
        )

    assert db.archive_sessions(title_like="Purple Elephant") == 1

    session = db.get_session("open")
    assert session["ended_at"] is None
    assert session["archived"] == 1
    assert session["title"] == "Purple Elephant Test"
    assert db.get_session("pinned-open")["archived"] == 0
