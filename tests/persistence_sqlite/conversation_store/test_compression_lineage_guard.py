"""Regression tests for stale writes after a compression session split."""

from __future__ import annotations

import time

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store import compression
from persistence.sqlite.conversation._store.compression import (
    find_live_compression_child,
    get_compression_lineage,
    publish_compression_child,
    reopen_orphaned_compression_session,
)
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session


@pytest.fixture()
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture()
def db(store):
    return store.holder()


def _compression_parent(db: StoreHolder, session_id: str = "parent") -> None:
    create_session(db, session_id, source="webui")
    append_message(db, session_id, "user", "before split")
    end_session(db, session_id, "compression")


def test_find_live_compression_child_returns_unique_direct_child(db: StoreHolder) -> None:
    _compression_parent(db)
    create_session(db, "child", source="webui", parent_session_id="parent")

    child = find_live_compression_child(db, "parent")

    assert child is not None
    assert child["id"] == "child"
    assert child["parent_session_id"] == "parent"
    assert child["ended_at"] is None


def test_find_live_compression_child_fails_closed_when_ambiguous(db: StoreHolder) -> None:
    _compression_parent(db)
    create_session(db, "child-a", source="webui", parent_session_id="parent")
    create_session(db, "child-b", source="webui", parent_session_id="parent")

    assert find_live_compression_child(db, "parent") is None


def test_reopen_orphaned_compression_session_reopens_parent_without_child(
    db: StoreHolder,
) -> None:
    _compression_parent(db, "orphan")

    assert reopen_orphaned_compression_session(db, "orphan") is True
    assert get_session(db, "orphan")["ended_at"] is None
    assert get_session(db, "orphan")["end_reason"] is None

    append_message(db, "orphan", "user", "recovered turn")
    assert [m["content"] for m in get_messages(db, "orphan")] == [
        "before split",
        "recovered turn",
    ]


def test_reopen_orphaned_compression_session_fails_closed_with_child(
    db: StoreHolder,
) -> None:
    _compression_parent(db, "parent-with-child")
    create_session(db, "child", source="webui", parent_session_id="parent-with-child")

    assert reopen_orphaned_compression_session(db, "parent-with-child") is False
    parent = get_session(db, "parent-with-child")
    assert parent["end_reason"] == "compression"
    assert parent["ended_at"] is not None


def test_reopen_orphaned_compression_session_ignores_non_continuation_children(
    db: StoreHolder,
) -> None:
    _compression_parent(db, "parent-with-non-continuation-children")
    create_session(
        db,
        "branch",
        source="webui",
        parent_session_id="parent-with-non-continuation-children",
        model_config={"_branched_from": "parent-with-non-continuation-children"},
    )
    create_session(
        db,
        "delegate",
        source="tool",
        parent_session_id="parent-with-non-continuation-children",
        model_config={"_delegate_from": "parent-with-non-continuation-children"},
    )

    assert reopen_orphaned_compression_session(
        db,
        "parent-with-non-continuation-children"
    ) is True


def test_reopen_fails_closed_when_continuation_inherits_foreign_markers(
    db: StoreHolder,
) -> None:
    """A REAL continuation can carry ``_delegate_from``/``_branched_from``
    pointing at some OTHER session: ``publish_compression_child`` callers
    pass the rotated agent's ``_session_init_model_config`` verbatim, so a
    delegate subagent's continuation inherits ``_delegate_from=<the
    delegate's own parent>``. Marker-presence matching misclassified it as
    a delegate child: reopen returned True with a live continuation
    present, forking the lineage. Markers only disqualify a child when
    they point at the queried parent."""
    _compression_parent(db, "delegate-session")
    create_session(
        db,
        "delegate-continuation",
        source="subagent",
        parent_session_id="delegate-session",
        model_config={"_delegate_from": "some-original-parent"},
    )

    assert reopen_orphaned_compression_session(db, "delegate-session") is False
    parent = get_session(db, "delegate-session")
    assert parent["end_reason"] == "compression"


def test_find_live_child_returns_continuation_with_foreign_markers(
    db: StoreHolder,
) -> None:
    """Adoption-side twin of the reopen test above: the continuation that
    inherited a foreign ``_delegate_from`` must still be adoptable."""
    _compression_parent(db, "delegate-session-2")
    create_session(
        db,
        "inherited-continuation",
        source="subagent",
        parent_session_id="delegate-session-2",
        model_config={"_delegate_from": "some-original-parent"},
    )

    child = find_live_compression_child(db, "delegate-session-2")
    assert child is not None
    assert child["id"] == "inherited-continuation"


def test_compression_lineage_includes_continuation_with_foreign_markers(
    db: StoreHolder,
) -> None:
    """Lineage walk uses the same parent-bound marker rule as orphan recovery."""
    _compression_parent(db, "delegate-session-3")
    create_session(
        db,
        "inherited-tip",
        source="subagent",
        parent_session_id="delegate-session-3",
        model_config={"_delegate_from": "some-original-parent"},
    )

    assert get_compression_lineage(db, "inherited-tip") == [
        "delegate-session-3",
        "inherited-tip",
    ]
    assert get_compression_lineage(db, "delegate-session-3") == [
        "delegate-session-3",
        "inherited-tip",
    ]


def test_reopen_orphaned_compression_session_fails_closed_with_active_lease(
    store, db: StoreHolder,
) -> None:
    _compression_parent(db, "leased-parent")
    assert store.coordination.try_acquire_compression_lock(session_id="leased-parent", holder="compressor")

    assert reopen_orphaned_compression_session(db, "leased-parent") is False
    assert get_session(db, "leased-parent")["end_reason"] == "compression"


def test_reopen_orphaned_compression_session_reclaims_expired_lease(
    store, db: StoreHolder,
) -> None:
    _compression_parent(db, "expired-lease-parent")
    now = time.time()
    db._conn.execute(
        "INSERT INTO compression_locks "
        "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        ("expired-lease-parent", "old-compressor", now - 60, now - 30),
    )
    db._conn.commit()

    assert reopen_orphaned_compression_session(db, "expired-lease-parent") is True
    assert store.coordination.refresh_compression_lock(session_id="expired-lease-parent", holder="old-compressor") is False
    assert store.coordination.get_compression_lock_holder(session_id="expired-lease-parent") is None


def test_reopen_orphaned_compression_session_loses_to_expired_lease_refresh(
    store, db: StoreHolder,
) -> None:
    _compression_parent(db, "refreshed-lease-parent")
    now = time.time()
    db._conn.execute(
        "INSERT INTO compression_locks "
        "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        ("refreshed-lease-parent", "live-compressor", now - 60, now - 30),
    )
    db._conn.commit()

    assert store.coordination.refresh_compression_lock(session_id="refreshed-lease-parent", holder="live-compressor") is True
    assert reopen_orphaned_compression_session(db, "refreshed-lease-parent") is False
    assert get_session(db, "refreshed-lease-parent")["end_reason"] == "compression"


def test_find_live_compression_child_ignores_non_continuation_children(
    db: StoreHolder,
) -> None:
    _compression_parent(db)
    create_session(db, "canonical", source="webui", parent_session_id="parent")
    create_session(
        db,
        "branch",
        source="webui",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    create_session(
        db,
        "delegate",
        source="webui",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    create_session(db, "tool-child", source="tool", parent_session_id="parent")

    child = find_live_compression_child(db, "parent")

    assert child is not None
    assert child["id"] == "canonical"


def test_reset_fork_is_not_a_compression_continuation_for_recovery(db: StoreHolder) -> None:
    """A reset fork (``model_config._reset_from`` bound to the parent) is its own conversation:
    stale-agent recovery must neither recover to it nor let it block the real continuation or a
    reopen of a parent that has no continuation (#114271)."""
    _compression_parent(db)
    create_session(db, "canonical", source="webui", parent_session_id="parent")
    create_session(
        db,
        "reset", source="webui", parent_session_id="parent", model_config={"_reset_from": "parent"},
    )

    child = find_live_compression_child(db, "parent")
    assert child is not None and child["id"] == "canonical"

    end_session(db, "canonical", "startup_orphan_reap")
    assert find_live_compression_child(db, "parent") is None

    _compression_parent(db, "orphan")
    create_session(
        db,
        "orphan-reset", source="webui", parent_session_id="orphan", model_config={"_reset_from": "orphan"},
    )
    assert reopen_orphaned_compression_session(db, "orphan") is True


def test_publish_compression_child_is_atomic_on_handoff_failure(
    store, db: StoreHolder, monkeypatch
) -> None:
    create_session(db, "atomic-parent", source="webui")
    append_message(db, "atomic-parent", "user", "original")
    assert store.coordination.try_acquire_compression_lock(session_id="atomic-parent", holder="winner", ttl_seconds=60)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("handoff insert failed")

    monkeypatch.setattr(compression, "_insert_message_rows", _boom)
    with pytest.raises(RuntimeError, match="handoff insert failed"):
        publish_compression_child(
            db,
            parent_session_id="atomic-parent",
            child_session_id="atomic-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

    parent = get_session(db, "atomic-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert get_session(db, "atomic-child") is None


def test_publish_compression_child_exposes_complete_child(store, db: StoreHolder) -> None:
    create_session(db, "atomic-parent", source="webui")
    append_message(db, "atomic-parent", "user", "original")
    assert store.coordination.try_acquire_compression_lock(session_id="atomic-parent", holder="winner", ttl_seconds=60)

    publish_compression_child(
        db,
        parent_session_id="atomic-parent",
        child_session_id="atomic-child",
        source="webui",
        system_prompt="compressed system",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    assert get_session(db, "atomic-parent")["end_reason"] == "compression"
    child = find_live_compression_child(db, "atomic-parent")
    assert child is not None
    assert child["id"] == "atomic-child"
    assert child["system_prompt"] == "compressed system"
    assert [m["content"] for m in get_messages(db, "atomic-child")] == ["summary"]


def test_publish_compression_child_rejects_lost_or_expired_lease(store, db: StoreHolder) -> None:
    create_session(db, "lease-parent", source="webui")
    append_message(db, "lease-parent", "user", "new durable turn")
    assert store.coordination.try_acquire_compression_lock(session_id="lease-parent", holder="new-winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="lease lost"):
        publish_compression_child(
            db,
            parent_session_id="lease-parent",
            child_session_id="stale-child",
            source="webui",
            messages=[{"role": "user", "content": "stale summary"}],
            compression_lock_holder="old-loser",
        )

    parent = get_session(db, "lease-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert get_session(db, "stale-child") is None
    assert [m["content"] for m in get_messages(db, "lease-parent")] == [
        "new durable turn"
    ]


def test_compression_lease_blocks_non_owner_but_allows_owner_flush(
    store, db: StoreHolder,
) -> None:
    """Contract flipped by the watermark commit (#75316): a live lease no
    longer fences ordinary appends: both the owner's flush and a concurrent
    turn land immediately, and the commit-side watermark decides what
    survives compaction (see test_compression_watermark_commit.py)."""
    create_session(db, "leased", source="webui")
    assert store.coordination.try_acquire_compression_lock(session_id="leased", holder="winner", ttl_seconds=60)

    append_message(db, "leased", "user", "late concurrent turn")
    append_message(
        db,
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in get_messages(db, "leased")] == [
        "late concurrent turn",
        "winner flush",
    ]
