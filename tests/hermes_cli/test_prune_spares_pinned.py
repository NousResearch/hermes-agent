"""Regression tests: pinned sessions survive bulk prune/archive (round-3 SES-01).

Pinning is a durable "keep" flag. Before this fix `_prune_filter_where` had no
pinned exclusion, so `hermes sessions prune`/`archive` with a filter silently
destroyed pinned conversations. These drive the real store (temp file DB).
"""

import time

import pytest

from gateway.session_ingress_media import native_inputs_root
from persistence.housekeeping.maintenance import PruneCriteria
from persistence_assembly import open_persistence, profile_scope
from session_retention import archive_sessions
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


@pytest.fixture()
def db(tmp_path):
    d = open_persistence(profile_scope(home=tmp_path))
    yield d
    try:
        d.close()
    except Exception:
        pass


def _mk(db, sid, title, pinned=False):
    seed_session(db, sid, "cli", title=title)
    # mark ended + old so it is prune-eligible
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET ended_at=?, started_at=?, message_count=1 WHERE id=?",
        (time.time() - 100 * 86400, time.time() - 100 * 86400, sid),
    ))
    if pinned:
        db.sessions.set_pinned(sid, True)


def test_prune_spares_pinned_by_default(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)
    _mk(db, "20260101_000001_bbbbbb", "delete me", pinned=False)

    # Filter matches both by title substring "me"; default must spare pinned.
    cands = db.maintenance.list_prune_candidates(criteria=PruneCriteria(title_like="me"))
    ids = {c.session_id for c in cands}
    assert "20260101_000001_bbbbbb" in ids
    assert "20260101_000000_aaaaaa" not in ids  # pinned excluded

    deleted = db.maintenance.prune_sessions(older_than_days=None, criteria=PruneCriteria(title_like="me"),
                                            media_root=str(native_inputs_root()))
    assert deleted.count == 1
    assert session_row(db, "20260101_000000_aaaaaa") is not None  # pinned survived
    assert session_row(db, "20260101_000001_bbbbbb") is None


def test_prune_include_pinned_opts_in(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)

    # With the opt-in, the pinned row IS eligible.
    cands = db.maintenance.list_prune_candidates(criteria=PruneCriteria(title_like="me", include_pinned=True))
    assert {c.session_id for c in cands} == {"20260101_000000_aaaaaa"}

    deleted = db.maintenance.prune_sessions(older_than_days=None,
                                            criteria=PruneCriteria(title_like="me", include_pinned=True),
                                            media_root=str(native_inputs_root()))
    assert deleted.count == 1
    assert session_row(db, "20260101_000000_aaaaaa") is None


def test_count_prune_matches_reflects_pinned_flag(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)
    _mk(db, "20260101_000001_bbbbbb", "delete me", pinned=False)

    assert db.maintenance.count_prune_matches(criteria=PruneCriteria(title_like="me", include_pinned=False)) == 1
    assert db.maintenance.count_prune_matches(criteria=PruneCriteria(title_like="me", include_pinned=True)) == 2


def test_archive_spares_pinned(db):
    _mk(db, "20260101_000000_aaaaaa", "keep me", pinned=True)
    _mk(db, "20260101_000001_bbbbbb", "arch me", pinned=False)

    archived = archive_sessions(db, db.scope.profile_dir, None, criteria=PruneCriteria(title_like="me"))
    assert archived == 1  # only the unpinned one
