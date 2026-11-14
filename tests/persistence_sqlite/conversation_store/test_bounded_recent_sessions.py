"""Regression coverage for latency-bounded recent-session browsing."""

import sqlite3
import time

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session, list_recent_sessions_bounded


@pytest.fixture
def db(tmp_path):
    return StoreHolder(tmp_path / "state.db")


def _set_activity(db, session_id, when):
    db._conn.execute(
        "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
        (when, session_id),
    )
    db._conn.commit()




def test_writable_startup_reconciles_legacy_activity_column_before_index(tmp_path):
    """A pre-last_activity_at store must heal through the real startup path."""
    path = tmp_path / "legacy-state.db"
    original = StoreHolder(path)
    original.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
        conn.execute("ALTER TABLE sessions DROP COLUMN last_activity_at")
        conn.commit()
    finally:
        conn.close()

    healed = StoreHolder(path)
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
        assert list_recent_sessions_bounded(healed, limit=1) == []
    finally:
        healed.close()


def test_bounded_recent_orders_by_durable_activity_and_shapes_preview(db):
    now = time.time()
    create_session(db, "older", source="cli")
    append_message(db, "older", role="user", content="older preview")
    create_session(db, "newer", source="cli")
    append_message(db, "newer", role="user", content="newer preview")
    _set_activity(db, "older", now - 20)
    _set_activity(db, "newer", now - 10)

    rows = list_recent_sessions_bounded(db, limit=2)

    assert [row["id"] for row in rows] == ["newer", "older"]
    assert rows[0]["preview"] == "newer preview"


def test_bounded_recent_maps_recent_compression_tip_to_logical_root(db):
    now = time.time()
    create_session(db, "root", source="cli")
    append_message(db, "root", role="user", content="root preview")
    end_session(db, "root", "compression")
    create_session(db, "tip", source="cli", parent_session_id="root")
    append_message(db, "tip", role="user", content="tip preview")
    _set_activity(db, "root", now - 1000)
    _set_activity(db, "tip", now)

    rows = list_recent_sessions_bounded(db, limit=1)

    assert rows[0]["id"] == "tip"
    assert rows[0]["_lineage_root_id"] == "root"
    assert rows[0]["preview"] == "tip preview"


def test_bounded_recent_keeps_reset_child_user_visible(db):
    now = time.time()
    create_session(db, "before-reset", source="cli", session_key="cli:one")
    end_session(db, "before-reset", "session_reset")
    create_session(
        db,
        "after-reset",
        source="cli",
        parent_session_id="before-reset",
        session_key="cli:one",
    )
    append_message(db, "after-reset", role="user", content="fresh conversation")
    _set_activity(db, "after-reset", now)

    rows = list_recent_sessions_bounded(db, limit=5)

    assert "after-reset" in [row["id"] for row in rows]


def test_bounded_recent_keeps_branch_separate_from_compression_parent(db):
    now = time.time()
    create_session(db, "branch-parent", source="cli")
    end_session(db, "branch-parent", "compression")
    create_session(
        db,
        "branch-child",
        source="cli",
        parent_session_id="branch-parent",
        model_config={"_branched_from": "branch-parent"},
    )
    append_message(db, "branch-child", role="user", content="branch preview")
    _set_activity(db, "branch-child", now)

    rows = list_recent_sessions_bounded(db, limit=5)

    branch = next(row for row in rows if row["id"] == "branch-child")
    assert branch.get("_lineage_root_id") is None


def test_bounded_recent_excludes_delegated_children_and_sources(db):
    now = time.time()
    create_session(db, "visible", source="cli")
    append_message(db, "visible", role="user", content="visible")
    _set_activity(db, "visible", now - 1)
    create_session(
        db,
        "delegated",
        source="cli",
        model_config={"_delegate_from": "parent"},
    )
    append_message(db, "delegated", role="user", content="hidden delegate")
    _set_activity(db, "delegated", now)
    create_session(db, "hidden-source", source="cron")
    append_message(db, "hidden-source", role="user", content="hidden source")
    _set_activity(db, "hidden-source", now + 1)

    rows = list_recent_sessions_bounded(
        db,
        limit=5,
        exclude_sources=["cron"],
    )

    assert [row["id"] for row in rows] == ["visible"]


def test_bounded_recent_omits_deep_lineage_when_traversal_cap_is_reached(db):
    now = time.time()
    parent = None
    for i in range(40):
        sid = f"deep-{i}"
        create_session(db, sid, source="cli", parent_session_id=parent)
        if parent is not None:
            end_session(db, parent, "compression")
        _set_activity(db, sid, now + i)
        parent = sid
    create_session(db, "visible-deep-peer", source="cli")
    _set_activity(db, "visible-deep-peer", now + 100)

    rows = list_recent_sessions_bounded(
        db,
        limit=5,
        candidate_limit=8,
        lineage_limit=8,
    )

    assert [row["id"] for row in rows] == ["visible-deep-peer"]


def test_bounded_recent_omits_branching_lineage_at_total_row_cap(db):
    now = time.time()
    create_session(db, "fanout-root", source="cli")
    end_session(db, "fanout-root", "compression")
    for i in range(40):
        sid = f"fanout-{i}"
        create_session(db, sid, source="cli", parent_session_id="fanout-root")
        _set_activity(db, sid, now + i)
    create_session(db, "visible-fanout-peer", source="cli")
    _set_activity(db, "visible-fanout-peer", now + 100)

    rows = list_recent_sessions_bounded(
        db,
        limit=5,
        candidate_limit=8,
        lineage_limit=8,
    )

    assert [row["id"] for row in rows] == ["visible-fanout-peer"]


def test_bounded_recent_cycle_is_deduplicated_and_omitted(db):
    now = time.time()
    create_session(db, "cycle-a", source="cli")
    create_session(db, "cycle-b", source="cli", parent_session_id="cycle-a")
    end_session(db, "cycle-a", "compression")
    end_session(db, "cycle-b", "compression")
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("cycle-b", "cycle-a"),
    )
    _set_activity(db, "cycle-a", now)
    _set_activity(db, "cycle-b", now + 1)
    create_session(db, "visible-cycle-peer", source="cli")
    _set_activity(db, "visible-cycle-peer", now + 2)

    rows = list_recent_sessions_bounded(
        db,
        limit=5,
        candidate_limit=8,
        lineage_limit=8,
    )

    assert [row["id"] for row in rows] == ["visible-cycle-peer"]


def test_bounded_recent_deadline_interrupts_sqlite(db):
    for i in range(40):
        sid = f"session-{i}"
        create_session(db, sid, source="cli")
        append_message(db, sid, role="user", content=f"message {i}")

    with pytest.raises(TimeoutError, match="recent-session browse exceeded"):
        list_recent_sessions_bounded(
            db,
            limit=20,
            candidate_limit=300,
            timeout_seconds=0.0,
        )

    # The progress handler is removed in finally: the same connection remains
    # usable after cancellation instead of poisoning subsequent gateway reads.
    assert get_session(db, "session-0")["id"] == "session-0"
