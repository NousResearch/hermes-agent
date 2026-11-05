"""The SQLite maintenance repository's prune ages a session by its last activity, and a rotated conversation as one."""
import time

import pytest

from hermes_cli.sessions_cmd import _note_pinned_skipped
from persistence.sqlite.conversation._store.compression import get_compression_lineage, publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, end_session, get_session, touch_session_activity,
)
from persistence_assembly import open_persistence_reader
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def test_prune_sessions_respects_touch_session_activity(db, store):
    now = time.time()
    old_time = now - 100 * 86400  # 100 days ago
    recent_time = now - 2 * 86400  # 2 days ago

    # Session with older messages and started_at, but recent heartbeat / activity
    create_session(db, "active_by_heartbeat", source="cli")
    append_message(db, "active_by_heartbeat", "user", "hello", timestamp=old_time)
    end_session(db, "active_by_heartbeat", "done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
        (old_time, old_time, "active_by_heartbeat"),
    )
    db._conn.commit()
    touch_session_activity(db, "active_by_heartbeat", ts=recent_time)

    # Truly old session
    create_session(db, "truly_old", source="cli")
    append_message(db, "truly_old", "user", "old message", timestamp=old_time)
    end_session(db, "truly_old", "done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
        (old_time, old_time, "truly_old"),
    )
    db._conn.commit()

    # Check dry-run candidates
    candidates = store.maintenance.list_prune_candidates(older_than_days=10)
    candidate_ids = [c.session_id for c in candidates]
    assert "active_by_heartbeat" not in candidate_ids
    assert "truly_old" in candidate_ids

    candidate_map = {c.session_id: c for c in store.maintenance.list_prune_candidates(older_than_days=None)}
    assert candidate_map["active_by_heartbeat"].last_active == pytest.approx(recent_time, abs=1.0)

    # Pruning sessions older than 10 days should spare active_by_heartbeat
    pruned = store.maintenance.prune_sessions(older_than_days=10, media_root=MEDIA_ROOT).count
    assert pruned == 1
    assert get_session(db, "truly_old") is None
    assert get_session(db, "active_by_heartbeat") is not None


def test_prune_keeps_the_compressed_segments_of_a_chat_still_in_use(db, store, capsys):
    """A rotated conversation ages as one: its old compressed-away segments stay while a later segment
    is still in use, and go together once the whole chat is idle. The CLI/dashboard preview lists what prune deletes."""
    old = time.time() - 120 * 86400

    def rotate(parent, child, at):
        publish_compression_child(db, parent_session_id=parent, child_session_id=child, source="cli",
                                  messages=[{"role": "user", "content": "[summary]", "timestamp": at}],
                                  require_compression_lease=False)
        db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (at, parent))
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (at, child))
        db._conn.commit()

    for chat in ("live", "idle"):
        create_session(db, chat, source="cli")
        append_message(db, chat, "user", f"{chat} early detail", timestamp=old)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (old, chat))
        db._conn.commit()
        rotate(chat, f"{chat}-2", old + 60)
        rotate(f"{chat}-2", f"{chat}-3", old + 120)
    append_message(db, "live-3", "user", "still talking today")
    end_session(db, "idle-3", "done")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'idle-3'", (old + 180,))
    db._conn.commit()

    idle = {"idle", "idle-2", "idle-3"}
    # The CLI pinned-skip note counts pinned rows, not the unpinned ancestors a pinned tip spares.
    db._conn.execute("UPDATE sessions SET pinned = 1 WHERE id = 'idle-3'")
    db._conn.commit()
    with open_persistence_reader(db.db_path) as reader:
        _note_pinned_skipped(reader, {"older_than_days": 90, "lineage_tips_only": False}, "prune")
    assert "Note: 1 pinned session also match" in capsys.readouterr().out
    db._conn.execute("UPDATE sessions SET pinned = 0 WHERE id = 'idle-3'")
    db._conn.commit()
    assert {c.session_id for c in store.maintenance.list_prune_candidates(older_than_days=90, whole_lineages=True)} == idle
    assert store.maintenance.prune_sessions(older_than_days=90, media_root=MEDIA_ROOT).count == len(idle)
    assert get_compression_lineage(db, "live-3") == ["live", "live-2", "live-3"]
    assert not any(get_session(db, sid) for sid in idle)
