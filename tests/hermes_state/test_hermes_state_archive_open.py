"""`hermes sessions archive --include-open`: archive (reversible) may reach never-ended sessions (#139597)."""
import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


def test_archive_include_open_reaches_never_ended_sessions(db):
    """Desktop sessions never set ended_at, so the shared ended-only selector hid them from
    bulk archive entirely; archive is reversible, so include_open lifts the guard (#139597)."""
    db.create_session(session_id="never-ended", source="desktop")
    db.create_session(session_id="ended", source="desktop")
    db.end_session("ended", "completed")

    assert db.archive_sessions(source="desktop") == 1
    assert db.get_session("ended")["archived"] == 1
    assert db.get_session("never-ended")["archived"] == 0

    assert db.archive_sessions(source="desktop", include_open=True) == 1
    assert db.get_session("never-ended")["archived"] == 1
    assert db.count_open_prune_matches(
        source="desktop", archived=False, include_open=True) == 0


def test_prune_rejects_include_open_fail_closed(db):
    """include_open is the reversible-archive escape hatch; the delete path must refuse it
    rather than silently drop the ended-session guard."""
    db.create_session(session_id="open", source="cli")
    with pytest.raises(ValueError, match="archive-only"):
        db.prune_sessions(include_open=True)
    assert db.get_session("open") is not None
