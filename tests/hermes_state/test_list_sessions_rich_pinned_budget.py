"""Regression tests for the pinned-budget fix in ``list_sessions_rich``.

History / bug:
- The sidebar's "load more" (再加载 N 个) button gates on a truncation flag
  computed by the desktop backend: ``recents_truncated[name] = unpinned_count
  >= recents_cap``, where ``unpinned_count`` is counted from the rows returned
  by ``list_sessions_rich(limit=cap, include_pinned=True)``.
- With ``include_pinned=True`` the LIMIT/OFFSET page was *not* excluding
  pinned rows, so active pins consumed page capacity: a profile with 43
  unpinned + 7 recently-active pinned rows in the first 50 returned
  ``unpinned_count = 43 < 50`` and falsely cleared the "load more" flag even
  though hundreds of unpinned sessions existed.
- Fix: when ``include_pinned=True`` the page query applies ``s.pinned = 0``
  (LIMIT capacity belongs to unpinned rows) and the pinned back-fill remains
  the single source of pinned rows.
"""

from __future__ import annotations

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _seed(db: SessionDB, unpinned: int, pinned: int):
    """Seed ``unpinned`` normal sessions and ``pinned`` active pin sessions.

    Pinned sessions get the freshest ``started_at`` so, in the pre-fix
    implementation, they would occupy the first rows of the LIMIT window.
    """
    base = time.time()
    for i in range(unpinned):
        db.create_session(f"u{i:03d}", source="cli")
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = ?",
            (base - unpinned + i, f"u{i:03d}"),
        )
    for i in range(pinned):
        db.create_session(f"p{i:03d}", source="cli")
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = ?",
            (base + 100 + i, f"p{i:03d}"),  # fresher than every unpinned row
        )
        assert db.set_session_pinned(f"p{i:03d}", True) is True
    db._conn.commit()


def test_include_pinned_page_capacity_not_diluted_by_active_pins(db):
    """55 unpinned + 10 freshly-active pins: the page must return 50 unpinned
    rows (LIMIT budget intact) plus all 10 pinned back-fills — not 40+10."""
    _seed(db, unpinned=55, pinned=10)

    rows = db.list_sessions_rich(
        limit=50,
        offset=0,
        order_by_last_active=True,
        include_pinned=True,
    )
    unpinned = [s for s in rows if not s.get("pinned")]
    pins = [s for s in rows if s.get("pinned")]

    assert len(unpinned) == 50, (
        "active pins must not consume the LIMIT page budget; "
        f"got {len(unpinned)} unpinned rows"
    )
    assert len(pins) == 10, "all pinned rows must be back-filled"
    assert len(rows) == 60

    # The truncation signal the sidebar depends on.
    assert len(unpinned) >= 50


def test_include_pinned_short_list_no_fake_full_page(db):
    """30 unpinned + 10 pins: page holds 30 unpinned (no fake full page), all
    pins back-filled. This is the ''no more to load'' case."""
    _seed(db, unpinned=30, pinned=10)

    rows = db.list_sessions_rich(
        limit=50,
        offset=0,
        order_by_last_active=True,
        include_pinned=True,
    )
    unpinned = [s for s in rows if not s.get("pinned")]
    pins = [s for s in rows if s.get("pinned")]

    assert len(unpinned) == 30
    assert len(pins) == 10
    assert len(rows) == 40
    assert len(unpinned) < 50


def test_without_include_pinned_still_excludes_children_only(db):
    """include_pinned=False must keep the historical behaviour: no pinned
    back-fill, but pinned rows may still surface if they fall inside the
    LIMIT window (the page query is unfiltered on pinned when no back-fill
    is requested — back-compat with callers that page over everything)."""
    _seed(db, unpinned=55, pinned=10)

    rows = db.list_sessions_rich(
        limit=50,
        offset=0,
        order_by_last_active=True,
        include_pinned=False,
    )
    # With pinned rows freshest, the unfiltered page may include pins.
    assert len(rows) == 50
    # No back-fill happened: pins in the page are only the ones that fit.
    assert all(s.get("pinned") for s in rows[:10])
