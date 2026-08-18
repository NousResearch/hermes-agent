"""Sweeping never-active keyed gateway rows (#82770).

The live-DB guard stops *new* fixture escapes, but it cannot touch rows that
are already in a developer's ``state.db`` — and bulk prune/archive cannot
either: their shared selector is pinned to ``ended_at IS NOT NULL`` so a live
session is never picked, which permanently excludes every never-closed row.

These tests pin the narrow selector that reaches them, and — more importantly
— pin the rows it must refuse to touch.
"""

import json
import time

import pytest

from hermes_state import SessionDB

DAY = 86400.0


@pytest.fixture()
def db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


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
    def test_selects_old_never_active_keyed_row(self, db):
        _insert(db, "junk-old", age_days=45)
        found = db.list_never_active_keyed_sessions(older_than_days=30)
        assert [r["id"] for r in found] == ["junk-old"]

    def test_ignores_row_inside_the_age_floor(self, db):
        _insert(db, "junk-young", age_days=3)
        assert db.list_never_active_keyed_sessions(older_than_days=30) == []

    def test_ignores_unkeyed_row(self, db):
        _insert(db, "cli-row", age_days=45, session_key=None, source="cli")
        assert db.list_never_active_keyed_sessions(older_than_days=30) == []

    def test_ignores_ended_row(self, db):
        """Ended rows belong to ordinary prune — this selector must not
        double-claim them."""
        _insert(db, "ended", age_days=45, ended_at=time.time() - 40 * DAY)
        assert db.list_never_active_keyed_sessions(older_than_days=30) == []

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
    def test_any_sign_of_use_or_intent_protects_the_row(self, db, field, value):
        _insert(db, "used", age_days=45, **{field: value})
        assert db.list_never_active_keyed_sessions(older_than_days=30) == []

    def test_row_with_messages_is_protected_even_if_counter_says_zero(self, db):
        """``message_count`` is a denormalised counter — trust the messages."""
        _insert(db, "stale-counter", age_days=45)
        db._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            ("stale-counter", "user", "hello", time.time() - 44 * DAY),
        )
        db._conn.commit()
        assert db.list_never_active_keyed_sessions(older_than_days=30) == []


class TestPrune:
    def test_deletes_candidates_and_leaves_everything_else(self, db):
        _insert(db, "junk-a", age_days=45)
        _insert(db, "junk-b", age_days=60)
        _insert(db, "keeper-young", age_days=1)
        _insert(db, "keeper-used", age_days=45, message_count=3)

        deleted, _ = db.prune_never_active_keyed_sessions(older_than_days=30)

        assert deleted == 2
        surviving = {
            r[0] for r in db._conn.execute("SELECT id FROM sessions").fetchall()
        }
        assert surviving == {"keeper-young", "keeper-used"}

    def test_drops_routing_entries_pointing_at_deleted_rows(self, db):
        """A routing entry that outlived its target would leave the gateway
        resuming a session id that no longer exists."""
        _insert(db, "junk", age_days=45)
        _insert(db, "keeper", age_days=1)
        db.save_gateway_routing_entry(
            "agent:main:telegram:dm:junk",
            json.dumps({"session_id": "junk"}),
            scope="/tmp/pytest-of-dev/test0",
        )
        db.save_gateway_routing_entry(
            "agent:main:telegram:dm:keeper",
            json.dumps({"session_id": "keeper"}),
            scope="/home/dev/project",
        )

        deleted, routing_deleted = db.prune_never_active_keyed_sessions(
            older_than_days=30
        )

        assert (deleted, routing_deleted) == (1, 1)
        remaining = db._conn.execute(
            "SELECT session_key FROM gateway_routing"
        ).fetchall()
        assert [r[0] for r in remaining] == ["agent:main:telegram:dm:keeper"]

    def test_no_candidates_is_a_no_op(self, db):
        _insert(db, "keeper", age_days=1)
        assert db.prune_never_active_keyed_sessions(older_than_days=30) == (0, 0)
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
