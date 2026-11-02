"""A pin covers the whole conversation, including segments compression publishes after it."""
import time
from contextlib import closing

from hermes_cli.sessions_cmd import _note_pinned_skipped
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


def test_a_pin_follows_the_chat_through_compression_and_bulk_cleanup_spares_it(tmp_path, capsys):
    """The published segment joins the pin. Stores written before it did hold a pinned segment
    with an unpinned tip, and bulk cleanup spares those too."""
    with closing(SessionDB(tmp_path / "state.db")) as db:
        _pinned_then_rotated(db)
        assert db.get_session("keep-2")["pinned"] == db.get_session("keep")["pinned"] == 1

        db._conn.execute("UPDATE sessions SET pinned = 0 WHERE id = 'keep-2'")
        db._conn.commit()

        assert db.list_prune_candidates(older_than_days=90, whole_lineages=True) == []
        _note_pinned_skipped(db, {"older_than_days": 90}, "prune")
        assert "Note: 1 pinned session also match" in capsys.readouterr().out
        assert db.prune_sessions(older_than_days=90) == 0
        assert db.archive_stale_sessions(3) == db.archive_sessions(older_than_days=90) == 0
        assert db.get_compression_lineage("keep") == ["keep", "keep-2"]
        assert not db.get_session("keep")["archived"]
        assert db.prune_sessions(older_than_days=90, include_pinned=True) == 2


def test_orphan_sweep_spares_the_open_tip_of_a_chat_pinned_before_it_rotated(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        _pinned_then_rotated(db)
        db._conn.execute("UPDATE sessions SET pinned = 0, ended_at = NULL, end_reason = NULL,"
                         " last_activity_at = started_at WHERE id = 'keep-2'")
        db._conn.commit()
        sweep = dict(max_idle_seconds=90 * 86400, sources=("cli",), respect_gateway_heartbeats=False)

        assert db.sweep_orphaned_sessions(exclude_pinned=True, **sweep) == []
        assert db.get_session("keep-2")["ended_at"] is None
        assert db.sweep_orphaned_sessions(exclude_pinned=False, **sweep) == ["keep-2"]
