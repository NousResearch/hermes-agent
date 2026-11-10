"""Sweeping never-active keyed gateway rows with the SQLite routing repository (#82770): ``never_active_keyed_sessions`` and ``prune_never_active``.

The live-DB guard stops *new* fixture escapes, but it cannot touch rows that
are already in a developer's ``state.db``, and bulk prune/archive cannot
either: their shared selector is pinned to ``ended_at IS NOT NULL`` so a live
session is never picked, which permanently excludes every never-closed row.

These tests pin the narrow selector that reaches them, and (more importantly)
pin the rows it must refuse to touch.
"""

import json
import os
import time

import pytest

from persistence.sqlite.conversation._store.sessions import get_session
from tests.contracts._scopes import MEDIA_ROOT

DAY = 86400.0


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture()
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _prune(store):
    """``(sessions deleted, routing entries deleted, sessions skipped)`` of one never-active prune, followed by the
    routing sweep over its deleted ids as its callers run it."""
    outcome = store.routing.prune_never_active(older_than_days=30, media_root=MEDIA_ROOT)
    store.routing.sweep_routing_entries_for_sessions(outcome.deleted_ids)
    return outcome.sessions_deleted, outcome.routing_entries_deleted, outcome.sessions_skipped


def _insert(db, session_id, *, age_days, **overrides):
    """Insert a keyed gateway row directly, defaulting to the junk shape."""
    row = {
        "id": session_id,
        "source": "telegram",
        "user_id": "user-1",
        "session_key": f"agent:main:telegram:dm:{session_id}",
        "chat_id": "chat-1",
        "chat_type": "dm",
        "started_at": time.time() - age_days * DAY,
        "message_count": 0,
        "tool_call_count": 0,
        "api_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "archived": 0,
        "pinned": 0,
    }
    row.update(overrides)
    cols = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    db._conn.execute(
        f"INSERT INTO sessions ({cols}) VALUES ({placeholders})", list(row.values())
    )
    db._conn.commit()
    return session_id


class TestSelector:
    def test_selects_old_never_active_keyed_row(self, db, store):
        _insert(db, "junk-old", age_days=45)
        found = store.routing.never_active_keyed_sessions(older_than_days=30)
        assert [r.session_id for r in found] == ["junk-old"]

    def test_ignores_row_inside_the_age_floor(self, db, store):
        _insert(db, "junk-young", age_days=3)
        assert store.routing.never_active_keyed_sessions(older_than_days=30) == ()

    def test_ignores_unkeyed_row(self, db, store):
        _insert(db, "cli-row", age_days=45, session_key=None, source="cli")
        assert store.routing.never_active_keyed_sessions(older_than_days=30) == ()

    def test_ignores_ended_row(self, db, store):
        """Ended rows belong to ordinary prune: this selector must not
        double-claim them."""
        _insert(db, "ended", age_days=45, ended_at=time.time() - 40 * DAY)
        assert store.routing.never_active_keyed_sessions(older_than_days=30) == ()

    @pytest.mark.parametrize(
        "field, value",
        [
            ("message_count", 1),
            ("tool_call_count", 1),
            ("api_call_count", 1),
            ("input_tokens", 12),
            ("output_tokens", 12),
            ("title", "kept by the user"),
            ("last_activity_at", 1785354069.0),
            ("pinned", 1),
            ("archived", 1),
        ],
    )
    def test_any_sign_of_use_or_intent_protects_the_row(self, db, store, field, value):
        _insert(db, "used", age_days=45, **{field: value})
        assert store.routing.never_active_keyed_sessions(older_than_days=30) == ()

    def test_row_with_messages_is_protected_even_if_counter_says_zero(self, db, store):
        """``message_count`` is a denormalised counter: trust the messages."""
        _insert(db, "stale-counter", age_days=45)
        db._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            ("stale-counter", "user", "hello", time.time() - 44 * DAY),
        )
        db._conn.commit()
        assert store.routing.never_active_keyed_sessions(older_than_days=30) == ()


class TestPrune:
    def test_deletes_candidates_and_leaves_everything_else(self, db, store):
        _insert(db, "junk-a", age_days=45)
        _insert(db, "junk-b", age_days=60)
        _insert(db, "keeper-young", age_days=1)
        _insert(db, "keeper-used", age_days=45, message_count=3)

        deleted, _, _ = _prune(store)

        assert deleted == 2
        surviving = {
            r[0] for r in db._conn.execute("SELECT id FROM sessions").fetchall()
        }
        assert surviving == {"keeper-young", "keeper-used"}

    def test_drops_routing_entries_pointing_at_deleted_rows(self, db, store):
        """A routing entry that outlived its target would leave the gateway
        resuming a session id that no longer exists."""
        _insert(db, "junk", age_days=45)
        _insert(db, "keeper", age_days=1)
        store.routing.upsert_routing_entry(
            "agent:main:telegram:dm:junk",
            json.dumps({"session_id": "junk"}),
            routing_scope="/tmp/pytest-of-dev/test0",
        )
        store.routing.upsert_routing_entry(
            "agent:main:telegram:dm:keeper",
            json.dumps({"session_id": "keeper"}),
            routing_scope="/home/dev/project",
        )

        result = _prune(store)

        assert result == (1, 1, 0)
        remaining = db._conn.execute(
            "SELECT session_key FROM gateway_routing"
        ).fetchall()
        assert [r[0] for r in remaining] == ["agent:main:telegram:dm:keeper"]

    def test_no_candidates_is_a_no_op(self, db, store):
        _insert(db, "keeper", age_days=1)
        assert _prune(store) == (0, 0, 0)
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


class TestPruneSkipsLiveTurns:
    """#123583 on the never-active selector: the predicate reads committed state, so a
    keyed row whose first turn lease is already held (messages not yet flushed) looks
    empty. The prune must skip it (and keep its routing entry) instead of deleting it
    mid-turn."""

    def test_guarded_row_survives_and_keeps_its_routing_entry(self, db, store, monkeypatch):
        _insert(db, "live-junk", age_days=45)
        _insert(db, "dead-junk", age_days=45)
        _insert(db, "gone-junk", age_days=45)
        for sid, scope in (("live-junk", "/tmp/pytest-live"), ("gone-junk", "/tmp/pytest-gone")):
            store.routing.upsert_routing_entry(
                f"agent:main:telegram:dm:{sid}", json.dumps({"session_id": sid}), routing_scope=scope)
        holder = f"pid={os.getpid()}:turn=1"
        assert store.coordination.try_acquire_turn_lease(
            session_id="live-junk", holder=holder, ttl_seconds=300.0) is True

        # gone-junk vanishes between listing and the sweep (a concurrent delete): it is not
        # counted as deleted or skipped, but its routing entry must still go; a stale entry
        # would have the gateway resume a nonexistent id.
        real_list = store.routing.never_active_keyed_sessions

        def list_then_vanish(**kw):
            rows = real_list(**kw)
            db._conn.execute("DELETE FROM sessions WHERE id = 'gone-junk'")
            db._conn.commit()
            return rows

        monkeypatch.setattr(store.routing, "never_active_keyed_sessions", list_then_vanish)
        # dead-junk deleted; the guarded row is skipped (counted), not fatal
        assert _prune(store) == (1, 1, 1)
        monkeypatch.undo()
        assert get_session(db, "live-junk") is not None
        remaining = {
            r[0]
            for r in db._conn.execute("SELECT session_key FROM gateway_routing").fetchall()
        }
        assert remaining == {"agent:main:telegram:dm:live-junk"}

        store.coordination.release_turn_lease(session_id="live-junk", holder=holder)
        assert _prune(store) == (1, 1, 0)
        assert get_session(db, "live-junk") is None
