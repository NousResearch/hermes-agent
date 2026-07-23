"""pop_side_session must be scopable to the calling client's side session."""

from hermes_state import SessionDB


def test_pop_side_session_is_scoped_to_the_calling_side_session(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    for s in ("PA", "SA", "PB", "SB"):
        db.create_session(s, source="tui")

    # Two clients share the same process-wide source; B pushed most recently.
    db.push_side_session("tui", "PA", "SA")
    db.push_side_session("tui", "PB", "SB")

    # Client A, currently in side session SA, returns. It must get its own
    # entry, not B's newer one.
    popped = db.pop_side_session(source="tui", side_session_id="SA")
    assert popped is not None
    assert popped["parent_session_id"] == "PA"
    assert popped["side_session_id"] == "SA"

    # B's parked session is untouched and still the active entry.
    active = db.get_active_side_session(source="tui")
    assert active is not None
    assert active["side_session_id"] == "SB"


def test_pop_side_session_returns_none_for_unknown_side_session(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    for s in ("P", "S"):
        db.create_session(s, source="tui")
    db.push_side_session("tui", "P", "S")

    assert db.pop_side_session(source="tui", side_session_id="other") is None
    assert db.get_active_side_session(source="tui")["side_session_id"] == "S"


def test_pop_side_session_without_scope_still_pops_newest(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    for s in ("P", "S1", "S2"):
        db.create_session(s, source="tui")
    db.push_side_session("tui", "P", "S1")
    db.push_side_session("tui", "P", "S2")
    assert db.pop_side_session(source="tui")["side_session_id"] == "S2"
