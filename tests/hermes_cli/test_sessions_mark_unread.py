"""`hermes sessions mark-unread` against a real SessionDB: a never-opened session reads as read
(NULL last_read_at) until its creator marks it unread."""

import sys

import pytest

SID = "20260315_092437_c9a6ff"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(SID, source="cli")
    db.close()
    return tmp_path / "state.db"


def _unread(store):
    from hermes_state import SessionDB
    db = SessionDB(db_path=store, read_only=True)
    try:
        return db.session_unread(db.get_session(SID))
    finally:
        db.close()


def test_mark_unread_surfaces_a_never_opened_session(store, monkeypatch):
    import hermes_cli.main as main_mod
    assert _unread(store) is False
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "mark-unread", SID[:8]])
    try:
        main_mod.main()
    except SystemExit as exc:
        assert exc.code in (0, None)
    assert _unread(store) is True
