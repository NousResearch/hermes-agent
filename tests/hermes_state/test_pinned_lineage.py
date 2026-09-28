"""A pin covers the whole conversation, including segments compression publishes after it."""
import time
from contextlib import closing

from hermes_state import SessionDB


def _pinned_then_rotated(db):
    """Chat ``keep`` is pinned, then rotates to ``keep-2``; both idle for 120 days."""
    old = time.time() - 120 * 86400
    db.create_session("keep", source="cli")
    db.append_message("keep", "user", "early detail", timestamp=old)
    assert db.set_session_pinned("keep", True)
    db.publish_compression_child(parent_session_id="keep", child_session_id="keep-2", source="cli",
                                 messages=[{"role": "user", "content": "[summary]", "timestamp": old + 60}],
                                 require_compression_lease=False)
    db.end_session("keep-2", "done")
    db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id IN ('keep', 'keep-2')",
                     (old, old + 120))
    db._conn.commit()


def test_a_pin_follows_the_chat_through_compression(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        _pinned_then_rotated(db)

        assert db.get_session("keep-2")["pinned"] == db.get_session("keep")["pinned"] == 1


def test_bulk_cleanup_spares_a_chat_pinned_before_it_rotated(tmp_path):
    """Stores written before the pin followed rotation hold a pinned segment with an unpinned tip."""
    with closing(SessionDB(tmp_path / "state.db")) as db:
        _pinned_then_rotated(db)
        db._conn.execute("UPDATE sessions SET pinned = 0 WHERE id = 'keep-2'")
        db._conn.commit()

        assert db.list_prune_candidates(older_than_days=90, whole_lineages=True) == []
        assert db.prune_sessions(older_than_days=90) == 0
        assert db.archive_stale_sessions(3) == db.archive_sessions(older_than_days=90) == 0
        assert db.get_compression_lineage("keep") == ["keep", "keep-2"]
        assert not db.get_session("keep")["archived"]
        assert db.prune_sessions(older_than_days=90, include_pinned=True) == 2
