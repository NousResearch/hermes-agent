"""A ``min_messages`` floor must not sever a compression chain from its listing.

The listable row of a compression chain is its ROOT. After every turn moved to the
live tip, the ancestors hold ``message_count = 0`` — so the sidebar's
``min_messages=1`` floor filtered out every row of the chain and the conversation
vanished from the desktop sidebar (#126841). The floor must admit a chain root
whose forward compression chain still holds a row meeting the floor, while truly
empty chains stay filtered.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _empty_segment(db: SessionDB, session_id: str, parent: str = None) -> None:
    db.create_session(session_id, source="webui", parent_session_id=parent)
    db.end_session(session_id, "compression")


def _listing(db: SessionDB, floor: int = 1, **kw):
    return [
        s["id"]
        for s in db.list_sessions_rich(min_message_count=floor, **kw)
    ]


def test_floor_admits_chain_root_whose_tip_holds_messages(db: SessionDB) -> None:
    """Seven 0-message ancestors + a live tip with the transcript: the chain must still
    surface (projected to the tip), not vanish from every ``min_messages >= 1`` listing."""
    _empty_segment(db, "seg1")
    _empty_segment(db, "seg2", parent="seg1")
    _empty_segment(db, "seg3", parent="seg2")
    db.create_session("tip", source="webui", parent_session_id="seg3")
    db.append_message("tip", "user", "hello")
    db.append_message("tip", "assistant", "hi there")

    listed = _listing(db)

    assert "tip" in listed, "conversation dropped despite a live, non-empty tip"


def test_floor_admits_chain_root_in_recency_order(db: SessionDB) -> None:
    """The desktop sidebar lists with ``order_by_last_active`` (the recursive CTE path);
    the same conversation must surface there too."""
    _empty_segment(db, "seg1")
    db.create_session("tip", source="webui", parent_session_id="seg1")
    db.append_message("tip", "user", "turn after compaction")

    listed = _listing(db, order_by_last_active=True)

    assert "tip" in listed


def test_truly_empty_chain_stays_filtered(db: SessionDB) -> None:
    """No row of the chain ever held a message: the floor must keep filtering."""
    _empty_segment(db, "seg1")
    db.create_session("tip", source="webui", parent_session_id="seg1")

    assert _listing(db) == []


def test_floor_is_checked_across_the_chain(db: SessionDB) -> None:
    """A higher floor reads the TIP's message_count, not the root's: a 2-message tip
    passes ``min_messages=3`` never, a 3-message tip passes it."""
    _empty_segment(db, "seg1")
    db.create_session("tip", source="webui", parent_session_id="seg1")
    db.append_message("tip", "user", "one")
    db.append_message("tip", "assistant", "two")

    assert _listing(db, floor=3) == []

    db.append_message("tip", "user", "three")

    assert _listing(db, floor=3) == ["tip"]


def test_floor_count_lines_up_with_pairing_total(db: SessionDB) -> None:
    """``session_count`` shares the WHERE builder: the "load more" total must agree
    with the listed rows (no phantom total for a chain the listing admits)."""
    _empty_segment(db, "seg1")
    db.create_session("tip", source="webui", parent_session_id="seg1")
    db.append_message("tip", "user", "hello")
    plain = db.create_session("plain", source="webui")
    db.append_message(plain, "user", "ordinary")

    total = db.session_count(min_message_count=1, exclude_children=True)
    listed = _listing(db)

    assert total == len(listed) == 2
