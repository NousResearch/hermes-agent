"""Regression tests: pinned sessions survive bulk prune/archive (round-3 SES-01).

Pinning is a durable "keep" flag. Before this fix `_prune_filter_where` had no
pinned exclusion, so `hermes sessions prune`/`archive` with a filter silently
destroyed pinned conversations. These drive the real SessionDB (temp file DB).
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


def _mk(db, sid, title, pinned=False):
    db.create_session(sid, source="cli")
    db.set_session_title(sid, title)
    # mark ended + old so it is prune-eligible
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, started_at=?, message_count=1 WHERE id=?",
            (time.time() - 100 * 86400, time.time() - 100 * 86400, sid),
        )
        db._conn.commit()
    if pinned:
        db.set_session_pinned(sid, True)


def test_prune_spares_pinned_by_default(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)
    _mk(db, "20260101_000001_bbbbbb", "delete me", pinned=False)

    # Filter matches both by title substring "me"; default must spare pinned.
    cands = db.list_prune_candidates(title_like="me")
    ids = {c["id"] for c in cands}
    assert "20260101_000001_bbbbbb" in ids
    assert "20260101_000000_aaaaaa" not in ids  # pinned excluded

    deleted = db.prune_sessions(older_than_days=None, title_like="me")
    assert deleted == 1
    assert db.get_session("20260101_000000_aaaaaa") is not None  # pinned survived
    assert db.get_session("20260101_000001_bbbbbb") is None


def test_prune_include_pinned_opts_in(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)

    # With the opt-in, the pinned row IS eligible.
    cands = db.list_prune_candidates(title_like="me", include_pinned=True)
    assert {c["id"] for c in cands} == {"20260101_000000_aaaaaa"}

    deleted = db.prune_sessions(older_than_days=None, title_like="me", include_pinned=True)
    assert deleted == 1
    assert db.get_session("20260101_000000_aaaaaa") is None


def test_count_prune_matches_reflects_pinned_flag(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)
    _mk(db, "20260101_000001_bbbbbb", "delete me", pinned=False)

    assert db.count_prune_matches(title_like="me", include_pinned=False) == 1
    assert db.count_prune_matches(title_like="me", include_pinned=True) == 2


def test_archive_spares_pinned(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)
    _mk(db, "20260101_000001_bbbbbb", "arch me", pinned=False)

    archived = db.archive_sessions(older_than_days=None, title_like="me")
    assert archived == 1  # only the unpinned one
