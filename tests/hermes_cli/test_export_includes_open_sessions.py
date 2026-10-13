"""Regression tests: a filtered export includes sessions that are still open.

`hermes sessions export` selects sessions with `list_prune_candidates`, which
exists for prune/archive and therefore only ever returned *ended* sessions
(``s.ended_at IS NOT NULL``). An export is a read, not a delete, so that guard
silently truncated it: on the real store, 109 of 122 telegram sessions have no
``ended_at`` and `--source telegram` exported 13 rows. Pinned rows were dropped
for the same reason — a "do not prune me" marker is not a "do not export me".

These drive the real SessionDB (temp file DB) and assert the two halves of the
contract: export sees everything, prune still refuses to touch open or pinned
rows. Run with ``python -m pytest tests/hermes_cli/test_export_includes_open_sessions.py``.
"""

import time

import pytest

from hermes_state import SessionDB

@pytest.fixture()
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass

def _mk(db, sid, title, *, ended=True, pinned=False, source="telegram"):
    db.create_session(sid, source=source)
    db.set_session_title(sid, title)
    stamp = time.time() - 100 * 86400
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, started_at=?, message_count=1 WHERE id=?",
            (stamp if ended else None, stamp, sid),
        )
        db._conn.commit()
    if pinned:
        db.set_session_pinned(sid, True)

def test_export_selection_includes_open_sessions(db):
    """The bug: an open session was invisible to a filtered export."""
    _mk(db, "20260101_000000_open01", "live one", ended=False)
    _mk(db, "20260101_000001_done01", "ended one", ended=True)

    prune_view = {c["id"] for c in db.list_prune_candidates(source="telegram")}
    assert prune_view == {"20260101_000001_done01"}, "prune must still skip the open row"

    export_view = {c["id"] for c in db.list_prune_candidates(
        source="telegram", include_open=True, include_pinned=True, archived=None)}
    assert export_view == {"20260101_000000_open01", "20260101_000001_done01"}

def test_export_selection_includes_pinned_sessions(db):
    """Pinned is "do not prune", not "do not export"."""
    _mk(db, "20260101_000000_pin001", "pinned one", pinned=True)
    _mk(db, "20260101_000001_free01", "unpinned one")

    assert {c["id"] for c in db.list_prune_candidates(source="telegram")} == {
        "20260101_000001_free01"}
    assert {c["id"] for c in db.list_prune_candidates(
        source="telegram", include_open=True, include_pinned=True, archived=None)} == {
        "20260101_000000_pin001", "20260101_000001_free01"}

def test_include_open_alone_leaves_the_pin_guard_on(db):
    """The two opt-ins are independent; passing one must not drop the other."""
    _mk(db, "20260101_000000_pin001", "pinned one", pinned=True)
    _mk(db, "20260101_000001_free01", "unpinned one", ended=False)

    got = {c["id"] for c in db.list_prune_candidates(
        source="telegram", include_open=True, archived=None)}
    assert got == {"20260101_000001_free01"}, "unpinned open row only"

def test_prune_still_never_deletes_an_open_session(db):
    """The safety property this whole separation exists to protect."""
    _mk(db, "20260101_000000_open01", "live one", ended=False)
    _mk(db, "20260101_000001_done01", "ended one", ended=True)

    deleted = db.prune_sessions(older_than_days=None, source="telegram")
    assert deleted == 1
    assert db.get_session("20260101_000000_open01") is not None, "open row must survive"
    assert db.get_session("20260101_000001_done01") is None

def test_prune_still_spares_pinned_when_open_is_included(db):
    """include_open must not leak into the delete path through a shared filter dict."""
    _mk(db, "20260101_000000_pin001", "pinned one", pinned=True)
    _mk(db, "20260101_000001_done01", "ended one", ended=True)

    deleted = db.prune_sessions(older_than_days=None, source="telegram")
    assert deleted == 1
    assert db.get_session("20260101_000000_pin001") is not None

def test_an_unfiltered_include_open_selection_is_valid_sql(db):
    """`--older-than` is optional, so every clause can be absent. A bare `WHERE `
    is a syntax error, and the empty selection must degrade to "every session"
    rather than to a broken statement."""
    _mk(db, "20260101_000000_open01", "live one", ended=False)
    _mk(db, "20260101_000001_done01", "ended one", ended=True)

    got = {c["id"] for c in db.list_prune_candidates(
        include_open=True, include_pinned=True, archived=None)}
    assert got == {"20260101_000000_open01", "20260101_000001_done01"}

def test_count_open_cannot_be_combined_with_include_open(db):
    """A selection that already includes open sessions skips none, so asking how
    many it skipped is a contradiction, not a number."""
    with pytest.raises(ValueError):
        db.count_open_prune_matches(source="telegram", include_open=True)
