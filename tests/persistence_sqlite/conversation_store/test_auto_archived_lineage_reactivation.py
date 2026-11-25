"""A chat the idle sweep archived must reappear once it is live again (#117713).

The sidebar admits a compression lineage by its ROOT's ``archived`` flag. When
``sessions.auto_archive`` hid an idle lineage and the chat later resumed and
compressed, the fresh tip was inserted with ``archived=0`` under the archived
root, so the live chat stayed invisible. A DELIBERATE archive (user/API/CLI)
must keep hiding the chat through the same re-activation.
"""
import time

import pytest

from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    end_session,
    get_session,
    list_sessions_rich,
    reopen_session,
    set_session_archived,
)


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    return store.holder()


def _sidebar_ids(db):
    """Same flags as the desktop sidebar slice (tips projected from roots)."""
    return {
        row["id"]
        for row in list_sessions_rich(
            db, limit=50, offset=0, min_message_count=1, include_archived=False,
            archived_only=False, order_by_last_active=True, compact_rows=True, include_pinned=True,
        )
    }


def _compress(store, db, parent, child):
    holder = f"holder-{child}"
    assert store.coordination.try_acquire_compression_lock(session_id=parent, holder=holder, ttl_seconds=60)
    publish_compression_child(
        db, parent_session_id=parent, child_session_id=child, source="telegram", system_prompt="p",
        messages=[{"role": "user", "content": f"summary for {child}"}], compression_lock_holder=holder)


def _flags(db, *ids):
    return {sid: (get_session(db, sid)["archived"], get_session(db, sid)["auto_archived"]) for sid in ids}


def _lineage(store, db):
    """root -> mid (compression) -> tip, all with messages."""
    create_session(db, "root", "telegram")
    append_message(db, "root", "user", "hello")
    _compress(store, db, "root", "mid")
    _compress(store, db, "mid", "tip")
    append_message(db, "tip", "user", "latest")


def _sweep(store):
    """The idle sweep: archive every idle lineage tip, stamping the sweep's provenance."""
    time.sleep(0.02)  # health: allow HX013 -- the zero-day idle cutoff compares the rows' last activity with the wall clock
    ids = store.maintenance.list_idle_archive_candidates(0)
    for session_id in ids:
        store.maintenance.auto_archive_lineage(session_id)
    return len(ids)


def test_compression_after_auto_archive_unhides_the_live_chat(store, db):
    _lineage(store, db)
    assert _sweep(store) == 1
    assert _flags(db, "root", "mid", "tip") == {s: (1, 1) for s in ("root", "mid", "tip")}
    assert not _sidebar_ids(db)

    # A new inbound message resumes the chat and compression publishes a fresh tip.
    _compress(store, db, "tip", "tip2")

    assert _flags(db, "root", "mid", "tip", "tip2") == {
        s: (0, 0) for s in ("root", "mid", "tip", "tip2")}
    assert _sidebar_ids(db) == {"tip2"}


def test_compression_keeps_a_manually_archived_chat_hidden(store, db):
    _lineage(store, db)
    assert set_session_archived(db, "tip", True)[0]

    _compress(store, db, "tip", "tip2")

    # The new tip inherits the deliberate archive instead of creating a mixed lineage.
    assert _flags(db, "root", "mid", "tip", "tip2") == {
        s: (1, 0) for s in ("root", "mid", "tip", "tip2")}
    assert not _sidebar_ids(db)
    # ...so a later idle sweep has no archived=0 tip to relabel as sweep-owned.
    assert _sweep(store) == 0
    _compress(store, db, "tip2", "tip3")
    assert not _sidebar_ids(db)


def test_manual_archive_overrides_sweep_provenance(store, db):
    _lineage(store, db)
    assert _sweep(store) == 1
    assert set_session_archived(db, "root", True)[0]  # user archives it deliberately afterwards
    assert _flags(db, "root", "mid", "tip") == {s: (1, 0) for s in ("root", "mid", "tip")}

    _compress(store, db, "tip", "tip2")
    assert not _sidebar_ids(db)


def test_sweep_does_not_relabel_a_manually_archived_ancestor(store, db):
    """Legacy mixed lineage (pre-fix): manually archived root, unarchived continuation tip."""
    _lineage(store, db)
    db._write_sql("UPDATE sessions SET archived = 1 WHERE id IN ('root', 'mid')", ())
    assert _sweep(store) == 1  # the tip is still archived=0, so it is a sweep candidate
    assert _flags(db, "root", "mid", "tip") == {"root": (1, 0), "mid": (1, 0), "tip": (1, 1)}

    _compress(store, db, "tip", "tip2")
    assert not _sidebar_ids(db)
    assert get_session(db, "root")["archived"] == 1


def test_resume_reopen_unhides_auto_archived_but_not_manual(store, db):
    create_session(db, "auto", "cli")
    append_message(db, "auto", "user", "a")
    end_session(db, "auto", "cli_close")
    create_session(db, "manual", "cli")
    append_message(db, "manual", "user", "m")
    end_session(db, "manual", "cli_close")
    assert set_session_archived(db, "manual", True)[0]
    assert _sweep(store) == 1
    assert _flags(db, "auto", "manual") == {"auto": (1, 1), "manual": (1, 0)}

    reopen_session(db, "auto")
    reopen_session(db, "manual")

    assert _flags(db, "auto", "manual") == {"auto": (0, 0), "manual": (1, 0)}
    assert _sidebar_ids(db) == {"auto"}


def test_unarchive_clears_sweep_provenance(store, db):
    _lineage(store, db)
    assert _sweep(store) == 1
    assert set_session_archived(db, "tip", False)[0]
    assert _flags(db, "root", "mid", "tip") == {s: (0, 0) for s in ("root", "mid", "tip")}
