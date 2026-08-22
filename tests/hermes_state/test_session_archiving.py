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


def _compression_pair(db: SessionDB):
    base = time.time() - 100
    db.create_session("root", source="cli")
    db.create_session("tip", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
        (base + 20,),
    )
    db._conn.commit()


def test_archiving_compression_tip_archives_projected_root(db):
    _compression_pair(db)

    assert db.set_session_archived("tip", True) is True

    assert db.get_session("root")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True, archived_only=True)] == ["tip"]


def test_unarchiving_compression_tip_unarchives_projected_root(db):
    _compression_pair(db)
    db.set_session_archived("tip", True)

    assert db.set_session_archived("tip", False) is True

    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["tip"]


# --- reopened-link regression -------------------------------------------------
#
# A mistaken TUI reaper close (``end_reason = 'ws_orphan_reap'``) followed by
# stale-route recovery (``reopen_session``) clears the parent's mutable
# ``end_reason``. Before the ``$._compression_from`` marker existed, every
# lineage walk keyed on ``end_reason = 'compression'`` silently lost the link:
# archiving only covered the segment below the break and the unarchived,
# re-active root resurrected the conversation in the sidebar.


def _published_pair(db: SessionDB):
    """Build root->tip through the real rotation path, then reap + reopen."""
    import json as _json

    db.create_session("root", source="cli")
    db.publish_compression_child(
        parent_session_id="root",
        child_session_id="tip",
        source="cli",
        messages=[{"role": "user", "content": "hello"}],
        require_compression_lease=False,
    )
    # The rotation must have stamped the durable lineage marker.
    row = db._conn.execute(
        "SELECT model_config FROM sessions WHERE id = 'tip'"
    ).fetchone()
    assert _json.loads(row["model_config"])["_compression_from"] == "root"

    # Simulate the production reap/reopen cycle that erases the link.
    db._conn.execute(
        "UPDATE sessions SET end_reason = 'ws_orphan_reap' WHERE id = 'root'"
    )
    db._conn.commit()
    db.reopen_session("root")
    assert db.get_session("root")["end_reason"] is None


def test_publish_compression_child_stamps_durable_marker(db):
    import json as _json

    db.create_session("root", source="cli")
    db.publish_compression_child(
        parent_session_id="root",
        child_session_id="tip",
        source="cli",
        messages=[{"role": "user", "content": "hello"}],
        require_compression_lease=False,
    )
    row = db._conn.execute(
        "SELECT model_config FROM sessions WHERE id = 'tip'"
    ).fetchone()
    assert _json.loads(row["model_config"])["_compression_from"] == "root"


def test_archiving_after_reopened_link_still_covers_whole_lineage(db):
    _published_pair(db)

    assert db.set_session_archived("tip", True) is True

    assert db.get_session("root")["archived"] == 1
    assert db.get_session("tip")["archived"] == 1
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == []


def test_unarchiving_after_reopened_link_restores_whole_lineage(db):
    _published_pair(db)
    db.set_session_archived("tip", True)

    assert db.set_session_archived("tip", False) is True

    assert db.get_session("root")["archived"] == 0
    assert db.get_session("tip")["archived"] == 0
    # The reopened root is live again (ended_at cleared by reopen_session),
    # so the sidebar surfaces the root row itself.
    assert [s["id"] for s in db.list_sessions_rich(order_by_last_active=True)] == ["root"]


def test_pinning_after_reopened_link_covers_whole_lineage(db):
    _published_pair(db)

    assert db.set_session_pinned("tip", True) is True

    assert db.get_session("root")["pinned"] == 1
    assert db.get_session("tip")["pinned"] == 1


def test_creation_time_marker_covers_children_of_already_ended_parent(db):
    """Children created under an ended compression parent keep the link even
    if the parent is reopened before any rotation backfill runs."""
    import json as _json

    base = time.time() - 100
    db.create_session("root", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, "
        "end_reason = 'compression' WHERE id = 'root'",
        (base, base + 10),
    )
    db.create_session("late", source="cli", parent_session_id="root")

    row = db._conn.execute(
        "SELECT model_config FROM sessions WHERE id = 'late'"
    ).fetchone()
    assert _json.loads(row["model_config"])["_compression_from"] == "root"

    db.reopen_session("root")
    assert db.set_session_archived("late", True) is True
    assert db.get_session("root")["archived"] == 1
