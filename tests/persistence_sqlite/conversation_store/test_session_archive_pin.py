"""Archive and pin in the conversation store on a backend store's StoreHolder: archived sessions leave the default listing, pinning un-hides an ordinary session but not the canonical Bot Chat, pins survive the page window, and the idle auto-archive sweep (``session_retention.run_auto_archive`` over the maintenance repository) spares pinned sessions."""

import time

import pytest

from persistence.sqlite.conversation._store.core import CANONICAL_BOT_CHAT_TITLE
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    get_session,
    list_sessions_rich,
    session_count,
    set_session_archived,
    set_session_hidden,
    set_session_pinned,
)
from persistence.sqlite.conversation._store.titles import set_session_title
from session_retention import run_auto_archive


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _archive_stale(store, idle_days, *, exclude_pinned=True):
    """One unthrottled idle auto-archive pass; the number of sessions it archived."""
    return run_auto_archive(store, idle_days=idle_days, min_interval_hours=0, exclude_pinned=exclude_pinned)["archived"]


class TestSessionArchive:
    """Soft-archiving hides a session from default listings without deleting it."""

    def _seed(self, db, sid, *, archived=False):
        create_session(db, session_id=sid, source="cli")
        append_message(db, session_id=sid, role="user", content=f"hello from {sid}")
        if archived:
            set_session_archived(db, sid, True)

    def test_set_session_archived_roundtrip(self, db):
        self._seed(db, "s1")
        assert set_session_archived(db, "s1", True)[0] is True
        assert get_session(db, "s1")["archived"] == 1
        assert set_session_archived(db, "s1", False)[0] is True
        assert get_session(db, "s1")["archived"] == 0

    def test_archived_excluded_by_default(self, db):
        self._seed(db, "live")
        self._seed(db, "hidden", archived=True)

        ids = [s["id"] for s in list_sessions_rich(db)]
        assert ids == ["live"]
        assert session_count(db) == 1


class TestSessionPinAndStaleArchive:
    """Pin as a durable keep flag + last-activity-based stale auto-archive."""

    def _pinned(self, db, sid):
        row = db._conn.execute(
            "SELECT pinned FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        return row["pinned"] if row is not None else None

    def _make_idle(self, db, sid, *, days_idle, source="cli"):
        """A session whose latest activity was ``days_idle`` days ago."""
        create_session(db, session_id=sid, source=source)
        append_message(db, session_id=sid, role="user", content=f"msg {sid}")
        old = time.time() - days_idle * 86400
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (old, sid))
        db._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE session_id = ?", (old, sid)
        )
        db._conn.commit()

    # ── pin flag ──────────────────────────────────────────────────────────
    def test_set_session_pinned_roundtrip(self, db):
        create_session(db, session_id="s1", source="cli")
        assert set_session_pinned(db, "s1", True) is True
        assert self._pinned(db, "s1") == 1
        assert set_session_pinned(db, "s1", False) is True
        assert self._pinned(db, "s1") == 0

    def test_pinning_a_hidden_session_makes_it_listable(self, db):
        """A bot-tile session is born hidden (#106171). Pinning it must clear ``hidden``, or the
        session is pinned-but-invisible: absent from both the default listing and the back-fill."""
        create_session(db, session_id="s1", source="cli")
        append_message(db, session_id="s1", role="user", content="hi")
        set_session_hidden(db, "s1", True)

        set_session_pinned(db, "s1", True)

        assert get_session(db, "s1")["hidden"] == 0
        listed_ids = [s["id"] for s in list_sessions_rich(db, min_message_count=1)]
        assert "s1" in listed_ids

    def test_pinning_the_canonical_bot_chat_leaves_it_hidden(self, db):
        """The canonical Bot Chat (hidden + exact registry title) is desktop-owned and must stay
        hidden even when pinned, or it leaks into the Sessions sidebar and loses its rename guard
        (review on #106180). Unlike an ordinary hidden session, pinning must not clear ``hidden``."""
        create_session(db, session_id="bot1", source="desktop")
        set_session_title(db, "bot1", CANONICAL_BOT_CHAT_TITLE)
        set_session_hidden(db, "bot1", True)

        set_session_pinned(db, "bot1", True)

        assert get_session(db, "bot1")["hidden"] == 1

    # ── pinned back-fill past the page window ─────────────────────────────
    def test_pinned_session_survives_the_limit_window(self, db):
        """A pin outlives recency: paging must not evict a pinned row.

        Without ``include_pinned`` the desktop's Pinned section renders empty
        for any conversation that has aged off the sidebar page.
        """
        for i in range(6):
            self._make_idle(db, f"s{i}", days_idle=6 - i)
        set_session_pinned(db, "s0", True)  # the oldest — off a 3-row page

        def ids(**kw):
            return [
                s["id"]
                for s in list_sessions_rich(db,
                    limit=3, min_message_count=1, order_by_last_active=True, **kw
                )
            ]

        page = ids()
        assert "s0" not in page, "precondition: the pin is off the page"

        with_pins = ids(include_pinned=True)
        assert "s0" in with_pins
        # The page itself is untouched; the pin is additive.
        assert with_pins[:3] == page
        assert len(with_pins) == len(page) + 1

    # ── stale archive ─────────────────────────────────────────────────────

    def test_pinned_sessions_are_spared(self, db, store):
        self._make_idle(db, "keep", days_idle=10)
        set_session_pinned(db, "keep", True)

        assert _archive_stale(store, 3) == 0
        assert get_session(db, "keep")["archived"] == 0
        # Opting out of the pin guard sweeps it.
        assert _archive_stale(store, 3, exclude_pinned=False) == 1
        assert get_session(db, "keep")["archived"] == 1
