"""The desktop session cost dot turns a session's status dot red once its cumulative
cache-READ tokens cross a threshold. That signal is driven entirely by the
``cache_read_tokens`` field on each compact session-list row, so the list projection
that feeds the desktop sidebar MUST carry it. This is a contract test between the
backend row shape and the frontend feature — not a snapshot of any particular value.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _row(db, sid, **kwargs):
    rows = db.list_sessions_rich(min_message_count=0, **kwargs)
    return next(s for s in rows if s["id"] == sid)


def test_compact_list_row_carries_cache_read_tokens(db):
    """The compact projection the desktop sidebar consumes (compact_rows=True) must
    include cache_read_tokens, and it must reflect the accounted value."""
    db.create_session(session_id="s1", source="cli", model="claude-opus")
    db.queue_token_counts(
        "s1", input_tokens=10, output_tokens=5, cache_read_tokens=40_000_000, model="claude-opus"
    )
    db.flush_token_counts()

    row = _row(db, "s1", compact_rows=True)

    assert "cache_read_tokens" in row
    assert row["cache_read_tokens"] == 40_000_000


def test_cache_read_tokens_defaults_to_zero_without_usage(db):
    """A brand-new session has no cache re-reads: the field is present and zero, so
    the frontend's `>= threshold` check never flags it (and never misreads a
    missing key as truthy)."""
    db.create_session(session_id="s1", source="cli")

    row = _row(db, "s1", compact_rows=True)

    assert row.get("cache_read_tokens") == 0
