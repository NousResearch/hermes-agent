import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def test_hidden_excluded_by_default_included_on_request(db):
    db.create_session("visible", source="cli")
    db.create_session("secret", source="cli")
    # Give both a message so the default min_message_count filter keeps them.
    for sid in ("visible", "secret"):
        db._conn.execute(
            "UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,)
        )
    db._conn.commit()

    # Flip the hidden flag on one session.
    assert db.set_session_hidden("secret", True) is True
    assert db.get_session("secret")["hidden"] == 1
    assert db.get_session("visible")["hidden"] == 0

    # Default listing drops the hidden row; include_hidden=True surfaces it.
    default_ids = {s["id"] for s in db.list_sessions_rich(min_message_count=1)}
    assert default_ids == {"visible"}

    all_ids = {
        s["id"]
        for s in db.list_sessions_rich(min_message_count=1, include_hidden=True)
    }
    assert all_ids == {"visible", "secret"}

    # Unhiding brings it back into the default listing.
    assert db.set_session_hidden("secret", False) is True
    assert db.get_session("secret")["hidden"] == 0
    unhidden_ids = {s["id"] for s in db.list_sessions_rich(min_message_count=1)}
    assert unhidden_ids == {"visible", "secret"}


def test_pinning_clears_hidden(db):
    """Pinning means "keep visible": pinning a hidden session unhides it (#106171)."""
    db.create_session("botsession", source="cli")
    assert db.set_session_hidden("botsession", True) is True

    assert db.set_session_pinned("botsession", True) is True

    row = db.get_session("botsession")
    assert row["pinned"] == 1
    assert row["hidden"] == 0


def test_unpinning_leaves_hidden_alone(db):
    """Unpinning restores no visibility: the hidden flag survives an unpin."""
    db.create_session("s", source="cli")
    assert db.set_session_pinned("s", True) is True
    assert db.set_session_hidden("s", True) is True

    assert db.set_session_pinned("s", False) is True

    row = db.get_session("s")
    assert row["pinned"] == 0
    assert row["hidden"] == 1


def test_pinned_backfill_lists_hidden_pinned_rows(db):
    """The pinned back-fill ignores the hidden filter: a pinned-but-hidden row
    (e.g. pinned before pinning cleared hidden) stays listable via
    ``include_pinned`` while remaining out of the default listing (#106171)."""
    db.create_session("legacy", source="cli")
    db.create_session("plain_hidden", source="cli")
    assert db.set_session_pinned("legacy", True) is True
    assert db.set_session_hidden("legacy", True) is True
    assert db.set_session_hidden("plain_hidden", True) is True

    default_ids = {s["id"] for s in db.list_sessions_rich()}
    assert "legacy" not in default_ids
    assert "plain_hidden" not in default_ids

    pinned_ids = {s["id"] for s in db.list_sessions_rich(include_pinned=True)}
    assert "legacy" in pinned_ids
    assert "plain_hidden" not in pinned_ids
