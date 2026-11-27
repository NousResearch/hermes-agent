"""On a SQLite backend store a pin covers the whole conversation, including segments compression publishes after it."""
import time

import pytest

from hermes_cli.sessions_cmd import _note_pinned_skipped
from persistence.housekeeping.maintenance import PruneCriteria
from persistence.sqlite.conversation._store.compression import get_compression_lineage, publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, end_session, get_session, set_session_pinned,
)
from persistence_assembly import open_persistence_reader
from session_retention import archive_sessions, run_auto_archive
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _pinned_then_rotated(db):
    """Chat ``keep`` is pinned, then rotates to ``keep-2``; both idle for 120 days."""
    old = time.time() - 120 * 86400
    create_session(db, "keep", source="cli")
    append_message(db, "keep", "user", "early detail", timestamp=old)
    assert set_session_pinned(db, "keep", True)
    publish_compression_child(db, parent_session_id="keep", child_session_id="keep-2", source="cli",
                              messages=[{"role": "user", "content": "[summary]", "timestamp": old + 60}],
                              require_compression_lease=False)
    end_session(db, "keep-2", "done")
    db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id IN ('keep', 'keep-2')",
                     (old, old + 120))
    db._conn.commit()


def test_a_pin_follows_the_chat_through_compression_and_bulk_cleanup_spares_it(db, store, capsys):
    """The published segment joins the pin. Stores written before it did hold a pinned segment
    with an unpinned tip, and bulk cleanup spares those too."""
    _pinned_then_rotated(db)
    assert get_session(db, "keep-2")["pinned"] == get_session(db, "keep")["pinned"] == 1

    db._conn.execute("UPDATE sessions SET pinned = 0 WHERE id = 'keep-2'")
    db._conn.commit()

    assert store.maintenance.list_prune_candidates(older_than_days=90, whole_lineages=True) == ()
    with open_persistence_reader(db.db_path) as reader:
        _note_pinned_skipped(reader, {"older_than_days": 90}, "prune")
    assert "Note: 1 pinned session also match" in capsys.readouterr().out
    assert store.maintenance.prune_sessions(older_than_days=90, media_root=MEDIA_ROOT).count == 0
    assert run_auto_archive(store, idle_days=3, min_interval_hours=0)["archived"] == archive_sessions(
        store, db.db_path.parent, older_than_days=90) == 0
    assert get_compression_lineage(db, "keep") == ["keep", "keep-2"]
    assert not get_session(db, "keep")["archived"]
    assert store.maintenance.prune_sessions(
        older_than_days=90, criteria=PruneCriteria(include_pinned=True), media_root=MEDIA_ROOT).count == 2


def test_orphan_sweep_spares_the_open_tip_of_a_chat_pinned_before_it_rotated(db, store):
    _pinned_then_rotated(db)
    db._conn.execute("UPDATE sessions SET pinned = 0, ended_at = NULL, end_reason = NULL,"
                     " last_activity_at = started_at WHERE id = 'keep-2'")
    db._conn.commit()
    sweep = dict(max_idle_seconds=90 * 86400, sources=("cli",), respect_gateway_heartbeats=False)

    assert store.maintenance.sweep_orphaned_sessions(exclude_pinned=True, **sweep) == ()
    assert get_session(db, "keep-2")["ended_at"] is None
    assert store.maintenance.sweep_orphaned_sessions(exclude_pinned=False, **sweep) == ("keep-2",)
