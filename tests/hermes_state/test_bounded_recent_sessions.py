"""Regression coverage for latency-bounded recent-session browsing."""

import sqlite3
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _set_activity(db, session_id, when):
    db._conn.execute(
        "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
        (when, session_id),
    )
    db._conn.commit()


def test_bounded_recent_uses_effective_activity_index(db):
    indexes = {
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert "idx_sessions_effective_activity" in indexes


def test_writable_startup_reconciles_legacy_activity_column_before_index(tmp_path):
    """A pre-last_activity_at store must heal through the real startup path."""
    path = tmp_path / "legacy-state.db"
    original = SessionDB(path)
    original.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
        conn.execute("ALTER TABLE sessions DROP COLUMN last_activity_at")
        conn.commit()
    finally:
        conn.close()

    healed = SessionDB(path)
    try:
        columns = {
            row[1] for row in healed._conn.execute("PRAGMA table_info(sessions)")
        }
        indexes = {
            row[0]
            for row in healed._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert "last_activity_at" in columns
        assert "idx_sessions_effective_activity" in indexes
        assert healed.list_recent_sessions_bounded(limit=1) == []
    finally:
        healed.close()


def test_bounded_recent_orders_by_durable_activity_and_shapes_preview(db):
    now = time.time()
    db.create_session("older", source="cli")
    db.append_message("older", role="user", content="older preview")
    db.create_session("newer", source="cli")
    db.append_message("newer", role="user", content="newer preview")
    _set_activity(db, "older", now - 20)
    _set_activity(db, "newer", now - 10)

    rows = db.list_recent_sessions_bounded(limit=2)

    assert [row["id"] for row in rows] == ["newer", "older"]
    assert rows[0]["preview"] == "newer preview"


def test_bounded_recent_maps_recent_compression_tip_to_logical_root(db):
    now = time.time()
    db.create_session("root", source="cli")
    db.append_message("root", role="user", content="root preview")
    db.end_session("root", "compression")
    db.create_session("tip", source="cli", parent_session_id="root")
    db.append_message("tip", role="user", content="tip preview")
    _set_activity(db, "root", now - 1000)
    _set_activity(db, "tip", now)

    rows = db.list_recent_sessions_bounded(limit=1)

    assert rows[0]["id"] == "tip"
    assert rows[0]["_lineage_root_id"] == "root"
    assert rows[0]["preview"] == "tip preview"


def test_bounded_recent_keeps_reset_child_user_visible(db):
    now = time.time()
    db.create_session("before-reset", source="cli", session_key="cli:one")
    db.end_session("before-reset", "session_reset")
    db.create_session(
        "after-reset",
        source="cli",
        parent_session_id="before-reset",
        session_key="cli:one",
    )
    db.append_message("after-reset", role="user", content="fresh conversation")
    _set_activity(db, "after-reset", now)

    rows = db.list_recent_sessions_bounded(limit=5)

    assert "after-reset" in [row["id"] for row in rows]


def test_bounded_recent_keeps_branch_separate_from_compression_parent(db):
    now = time.time()
    db.create_session("branch-parent", source="cli")
    db.end_session("branch-parent", "compression")
    db.create_session(
        "branch-child",
        source="cli",
        parent_session_id="branch-parent",
        model_config={"_branched_from": "branch-parent"},
    )
    db.append_message("branch-child", role="user", content="branch preview")
    _set_activity(db, "branch-child", now)

    rows = db.list_recent_sessions_bounded(limit=5)

    branch = next(row for row in rows if row["id"] == "branch-child")
    assert branch.get("_lineage_root_id") is None


def test_bounded_recent_excludes_delegated_children_and_sources(db):
    now = time.time()
    db.create_session("visible", source="cli")
    db.append_message("visible", role="user", content="visible")
    _set_activity(db, "visible", now - 1)
    db.create_session(
        "delegated",
        source="cli",
        model_config={"_delegate_from": "parent"},
    )
    db.append_message("delegated", role="user", content="hidden delegate")
    _set_activity(db, "delegated", now)
    db.create_session("hidden-source", source="cron")
    db.append_message("hidden-source", role="user", content="hidden source")
    _set_activity(db, "hidden-source", now + 1)

    rows = db.list_recent_sessions_bounded(
        limit=5,
        exclude_sources=["cron"],
    )

    assert [row["id"] for row in rows] == ["visible"]


def test_bounded_recent_omits_deep_lineage_when_traversal_cap_is_reached(db):
    now = time.time()
    parent = None
    for i in range(40):
        sid = f"deep-{i}"
        db.create_session(sid, source="cli", parent_session_id=parent)
        if parent is not None:
            db.end_session(parent, "compression")
        _set_activity(db, sid, now + i)
        parent = sid
    db.create_session("visible-deep-peer", source="cli")
    _set_activity(db, "visible-deep-peer", now + 100)

    rows = db.list_recent_sessions_bounded(
        limit=5,
        candidate_limit=8,
        lineage_limit=8,
    )

    assert [row["id"] for row in rows] == ["visible-deep-peer"]


def test_bounded_recent_omits_branching_lineage_at_total_row_cap(db):
    now = time.time()
    db.create_session("fanout-root", source="cli")
    db.end_session("fanout-root", "compression")
    for i in range(40):
        sid = f"fanout-{i}"
        db.create_session(sid, source="cli", parent_session_id="fanout-root")
        _set_activity(db, sid, now + i)
    db.create_session("visible-fanout-peer", source="cli")
    _set_activity(db, "visible-fanout-peer", now + 100)

    rows = db.list_recent_sessions_bounded(
        limit=5,
        candidate_limit=8,
        lineage_limit=8,
    )

    assert [row["id"] for row in rows] == ["visible-fanout-peer"]


def test_bounded_recent_cycle_is_deduplicated_and_omitted(db):
    now = time.time()
    db.create_session("cycle-a", source="cli")
    db.create_session("cycle-b", source="cli", parent_session_id="cycle-a")
    db.end_session("cycle-a", "compression")
    db.end_session("cycle-b", "compression")
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("cycle-b", "cycle-a"),
    )
    _set_activity(db, "cycle-a", now)
    _set_activity(db, "cycle-b", now + 1)
    db.create_session("visible-cycle-peer", source="cli")
    _set_activity(db, "visible-cycle-peer", now + 2)

    rows = db.list_recent_sessions_bounded(
        limit=5,
        candidate_limit=8,
        lineage_limit=8,
    )

    assert [row["id"] for row in rows] == ["visible-cycle-peer"]


def test_bounded_recent_deadline_interrupts_sqlite(db):
    for i in range(300):
        sid = f"session-{i}"
        db.create_session(sid, source="cli")
        db.append_message(sid, role="user", content=f"message {i}")

    with pytest.raises(TimeoutError, match="recent-session browse exceeded"):
        db.list_recent_sessions_bounded(
            limit=20,
            candidate_limit=300,
            timeout_seconds=0.0,
        )

    # The progress handler is removed in finally: the same connection remains
    # usable after cancellation instead of poisoning subsequent gateway reads.
    assert db.get_session("session-0")["id"] == "session-0"