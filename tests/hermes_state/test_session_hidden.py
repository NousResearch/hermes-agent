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


def test_session_count_matches_paged_listing_with_hidden(db):
    """Regression for #123612: session_count() must mirror list_sessions_rich's hidden
    filter, so a paired pagination 'total' equals the distinct rows the pages serve."""
    ids = [f"sess_{i}" for i in range(5)]
    for sid in ids:
        db.create_session(sid, source="cli")
        db._conn.execute("UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,))
    db._conn.commit()
    for sid in ids[:2]:
        assert db.set_session_hidden(sid, True) is True

    # Page the default listing and collect every distinct row served.
    served: list = []
    offset = 0
    while True:
        rows = db.list_sessions_rich(limit=2, offset=offset)
        served += [r["id"] for r in rows]
        if len(rows) < 2:
            break
        offset += 2
    distinct_served = set(served)

    assert len(served) == len(distinct_served) == 3
    assert db.session_count(min_message_count=1) == len(distinct_served)

    # Opt-in parity with include_hidden listings.
    assert db.session_count(min_message_count=1, include_hidden=True) == 5


def test_session_count_archived_only_keeps_hidden_rows(db):
    """The archived-only view is the recovery surface for rows dropped from every default
    list (#90946): a session that is archived AND hidden must still be counted there."""
    db.create_session("gone", source="cli")
    db._conn.execute("UPDATE sessions SET message_count = 1, archived = 1 WHERE id = ?", ("gone",))
    db._conn.commit()
    assert db.set_session_hidden("gone", True) is True

    assert db.session_count() == 0
    assert db.session_count(archived_only=True) == 1
    assert db.list_sessions_rich(archived_only=True) != []
