"""Regression tests for stale writes after a compression session split."""

from __future__ import annotations

from contextlib import contextmanager
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _compression_parent(db: SessionDB, session_id: str = "parent") -> None:
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "before split")
    db.end_session(session_id, "compression")


def _set_raw_model_config(db: SessionDB, session_id: str, value) -> None:
    with db._lock:
        assert db._conn is not None
        db._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (value, session_id),
        )
        db._conn.commit()


def test_find_live_compression_child_returns_unique_direct_child(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child", source="webui", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "child"
    assert child["parent_session_id"] == "parent"
    assert child["ended_at"] is None


def test_find_live_compression_child_follows_multi_hop_chain(db: SessionDB) -> None:
    _compression_parent(db)
    parent = "parent"
    for child in ("child-1", "child-2"):
        db.create_session(child, source="webui", parent_session_id=parent)
        db.end_session(child, "compression")
        parent = child
    db.create_session("live-tip", source="webui", parent_session_id=parent)

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "live-tip"
    assert child["parent_session_id"] == "child-2"


def test_find_live_compression_child_rejects_fork_at_any_hop(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("rotated", source="webui", parent_session_id="parent")
    db.end_session("rotated", "compression")
    db.create_session("tip-a", source="webui", parent_session_id="rotated")
    db.create_session("tip-b", source="webui", parent_session_id="rotated")

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_rejects_cycle(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child", source="webui", parent_session_id="parent")
    db.end_session("child", "compression")

    def _close_cycle(conn) -> None:
        conn.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
            ("child", "parent"),
        )

    db._execute_write(_close_cycle)

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_has_no_shallow_depth_cap(db: SessionDB) -> None:
    _compression_parent(db)
    parent = "parent"
    for index in range(128):
        child = f"compressed-{index}"
        db.create_session(child, source="webui", parent_session_id=parent)
        db.end_session(child, "compression")
        parent = child
    db.create_session("deep-live-tip", source="webui", parent_session_id=parent)

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "deep-live-tip"


def test_find_live_compression_child_counts_closed_canonical_sibling(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("closed-sibling", source="webui", parent_session_id="parent")
    db.end_session("closed-sibling", "agent_close")
    db.create_session("live-child", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_rejects_live_intermediate(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("live-intermediate", source="webui", parent_session_id="parent")
    db.create_session(
        "live-grandchild",
        source="webui",
        parent_session_id="live-intermediate",
    )

    assert db.find_live_compression_child("parent") is None


@pytest.mark.parametrize(
    ("ended_at", "end_reason"),
    [
        (None, "compression"),
        (123.0, None),
        (123.0, "agent_close"),
    ],
)
def test_find_live_compression_child_rejects_incoherent_child_lifecycle(
    db: SessionDB,
    ended_at,
    end_reason,
) -> None:
    _compression_parent(db)
    db.create_session("incoherent", source="webui", parent_session_id="parent")
    with db._lock:
        assert db._conn is not None
        db._conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
            (ended_at, end_reason, "incoherent"),
        )
        db._conn.commit()

    assert db.find_live_compression_child("parent") is None


@pytest.mark.parametrize("raw_config", ["{not-json", "[]", "null"])
def test_find_live_compression_child_rejects_malformed_metadata(
    db: SessionDB,
    raw_config: str,
) -> None:
    _compression_parent(db)
    db.create_session("malformed", source="webui", parent_session_id="parent")
    _set_raw_model_config(db, "malformed", raw_config)

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_rejects_nontext_metadata(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("blob-config", source="webui", parent_session_id="parent")
    _set_raw_model_config(db, "blob-config", b"{}")

    assert db.find_live_compression_child("parent") is None


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from"])
@pytest.mark.parametrize("value", [None, "", 7, "wrong-parent"])
def test_find_live_compression_child_rejects_invalid_decoy_marker(
    db: SessionDB,
    marker: str,
    value,
) -> None:
    _compression_parent(db)
    db.create_session("canonical", source="webui", parent_session_id="parent")
    db.create_session(
        "invalid-decoy",
        source="webui",
        parent_session_id="parent",
        model_config={marker: value},
    )

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_rejects_duplicate_decoy_markers(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("canonical", source="webui", parent_session_id="parent")
    db.create_session(
        "invalid-decoy",
        source="webui",
        parent_session_id="parent",
        model_config={
            "_branched_from": "parent",
            "_delegate_from": "parent",
        },
    )

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_ignores_only_valid_leaf_decoys(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("live-tip", source="webui", parent_session_id="parent")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="live-tip",
        model_config={"_branched_from": "live-tip"},
    )
    db.create_session(
        "delegate",
        source="webui",
        parent_session_id="live-tip",
        model_config={"_delegate_from": "live-tip"},
    )
    db.create_session("tool-child", source="tool", parent_session_id="live-tip")
    _set_raw_model_config(db, "tool-child", "{not-json")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "live-tip"


def test_find_live_compression_child_holds_one_read_snapshot(
    db: SessionDB,
    monkeypatch,
) -> None:
    _compression_parent(db)
    db.create_session("live-tip", source="webui", parent_session_id="parent")
    assert db._conn is not None
    with db._lock:
        journal_mode = db._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    assert str(journal_mode).lower() == "wal"
    writer = SessionDB(db_path=db.db_path)
    inserted = False

    def _insert_descendant_after_snapshot(sql: str) -> None:
        nonlocal inserted
        compact_sql = " ".join(sql.split())
        if not inserted and "parent_session_id = 'parent'" in compact_sql:
            inserted = True
            writer.create_session(
                "late-descendant",
                source="webui",
                parent_session_id="live-tip",
            )

    original_read_ctx = db._read_ctx

    @contextmanager
    def _traced_read_ctx():
        with original_read_ctx() as conn:
            assert conn is not None
            conn.set_trace_callback(_insert_descendant_after_snapshot)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    monkeypatch.setattr(db, "_read_ctx", _traced_read_ctx)
    try:
        child = db.find_live_compression_child("parent")
    finally:
        writer.close()

    assert inserted is True
    assert child is not None
    assert child["id"] == "live-tip"
    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_fails_closed_when_ambiguous(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child-a", source="webui", parent_session_id="parent")
    db.create_session("child-b", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None


def test_reopen_orphaned_compression_session_reopens_parent_without_child(
    db: SessionDB,
) -> None:
    _compression_parent(db, "orphan")

    assert db.reopen_orphaned_compression_session("orphan") is True
    assert db.get_session("orphan")["ended_at"] is None
    assert db.get_session("orphan")["end_reason"] is None

    db.append_message("orphan", "user", "recovered turn")
    assert [m["content"] for m in db.get_messages("orphan")] == [
        "before split",
        "recovered turn",
    ]


def test_reopen_orphaned_compression_session_fails_closed_with_child(
    db: SessionDB,
) -> None:
    _compression_parent(db, "parent-with-child")
    db.create_session("child", source="webui", parent_session_id="parent-with-child")

    assert db.reopen_orphaned_compression_session("parent-with-child") is False
    parent = db.get_session("parent-with-child")
    assert parent["end_reason"] == "compression"
    assert parent["ended_at"] is not None


def test_reopen_orphaned_compression_session_ignores_non_continuation_children(
    db: SessionDB,
) -> None:
    _compression_parent(db, "parent-with-non-continuation-children")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="parent-with-non-continuation-children",
        model_config={"_branched_from": "parent-with-non-continuation-children"},
    )
    db.create_session(
        "delegate",
        source="tool",
        parent_session_id="parent-with-non-continuation-children",
        model_config={"_delegate_from": "parent-with-non-continuation-children"},
    )

    assert db.reopen_orphaned_compression_session(
        "parent-with-non-continuation-children"
    ) is True


def test_reopen_fails_closed_when_continuation_inherits_foreign_markers(
    db: SessionDB,
) -> None:
    """A REAL continuation can carry ``_delegate_from``/``_branched_from``
    pointing at some OTHER session: ``publish_compression_child`` callers
    pass the rotated agent's ``_session_init_model_config`` verbatim, so a
    delegate subagent's continuation inherits ``_delegate_from=<the
    delegate's own parent>``. Marker-presence matching misclassified it as
    a delegate child — reopen returned True with a live continuation
    present, forking the lineage. Markers only disqualify a child when
    they point at the queried parent."""
    _compression_parent(db, "delegate-session")
    db.create_session(
        "delegate-continuation",
        source="subagent",
        parent_session_id="delegate-session",
        model_config={"_delegate_from": "some-original-parent"},
    )

    assert db.reopen_orphaned_compression_session("delegate-session") is False
    parent = db.get_session("delegate-session")
    assert parent["end_reason"] == "compression"


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from"])
def test_find_live_child_rejects_wrong_parent_marker(
    db: SessionDB,
    marker: str,
) -> None:
    """Conservative stale-writer adoption must not reinterpret a marker that
    points elsewhere; only a marker bound to the direct parent is a decoy."""
    _compression_parent(db, "delegate-session-2")
    db.create_session(
        "wrong-parent-marker",
        source="subagent",
        parent_session_id="delegate-session-2",
        model_config={marker: "some-original-parent"},
    )

    assert db.find_live_compression_child("delegate-session-2") is None


def test_compression_lineage_includes_continuation_with_foreign_markers(
    db: SessionDB,
) -> None:
    """Lineage walk uses the same parent-bound marker rule as orphan recovery."""
    _compression_parent(db, "delegate-session-3")
    db.create_session(
        "inherited-tip",
        source="subagent",
        parent_session_id="delegate-session-3",
        model_config={"_delegate_from": "some-original-parent"},
    )

    assert db.get_compression_lineage("inherited-tip") == [
        "delegate-session-3",
        "inherited-tip",
    ]
    assert db.get_compression_lineage("delegate-session-3") == [
        "delegate-session-3",
        "inherited-tip",
    ]


def test_reopen_orphaned_compression_session_fails_closed_with_active_lease(
    db: SessionDB,
) -> None:
    _compression_parent(db, "leased-parent")
    assert db.try_acquire_compression_lock("leased-parent", "compressor")

    assert db.reopen_orphaned_compression_session("leased-parent") is False
    assert db.get_session("leased-parent")["end_reason"] == "compression"


def test_reopen_orphaned_compression_session_reclaims_expired_lease(
    db: SessionDB,
) -> None:
    _compression_parent(db, "expired-lease-parent")
    now = time.time()
    db._conn.execute(
        "INSERT INTO compression_locks "
        "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        ("expired-lease-parent", "old-compressor", now - 60, now - 30),
    )
    db._conn.commit()

    assert db.reopen_orphaned_compression_session("expired-lease-parent") is True
    assert db.refresh_compression_lock(
        "expired-lease-parent", "old-compressor"
    ) is False
    assert db.get_compression_lock_holder("expired-lease-parent") is None


def test_reopen_orphaned_compression_session_loses_to_expired_lease_refresh(
    db: SessionDB,
) -> None:
    _compression_parent(db, "refreshed-lease-parent")
    now = time.time()
    db._conn.execute(
        "INSERT INTO compression_locks "
        "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        ("refreshed-lease-parent", "live-compressor", now - 60, now - 30),
    )
    db._conn.commit()

    assert db.refresh_compression_lock(
        "refreshed-lease-parent", "live-compressor"
    ) is True
    assert db.reopen_orphaned_compression_session("refreshed-lease-parent") is False
    assert db.get_session("refreshed-lease-parent")["end_reason"] == "compression"


def test_find_live_compression_child_ignores_non_continuation_children(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("canonical", source="webui", parent_session_id="parent")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    db.create_session(
        "delegate",
        source="webui",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    db.create_session("tool-child", source="tool", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "canonical"


def test_publish_compression_child_is_atomic_on_handoff_failure(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("handoff insert failed")

    monkeypatch.setattr(db, "_insert_message_rows", _boom)
    with pytest.raises(RuntimeError, match="handoff insert failed"):
        db.publish_compression_child(
            parent_session_id="atomic-parent",
            child_session_id="atomic-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

    parent = db.get_session("atomic-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("atomic-child") is None


def test_publish_compression_child_exposes_complete_child(db: SessionDB) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="atomic-parent",
        child_session_id="atomic-child",
        source="webui",
        system_prompt="compressed system",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    assert db.get_session("atomic-parent")["end_reason"] == "compression"
    child = db.find_live_compression_child("atomic-parent")
    assert child is not None
    assert child["id"] == "atomic-child"
    assert child["system_prompt"] == "compressed system"
    assert [m["content"] for m in db.get_messages("atomic-child")] == ["summary"]


def test_publish_compression_child_rejects_lost_or_expired_lease(db: SessionDB) -> None:
    db.create_session("lease-parent", source="webui")
    db.append_message("lease-parent", "user", "new durable turn")
    assert db.try_acquire_compression_lock("lease-parent", "new-winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="lease lost"):
        db.publish_compression_child(
            parent_session_id="lease-parent",
            child_session_id="stale-child",
            source="webui",
            messages=[{"role": "user", "content": "stale summary"}],
            compression_lock_holder="old-loser",
        )

    parent = db.get_session("lease-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("stale-child") is None
    assert [m["content"] for m in db.get_messages("lease-parent")] == [
        "new durable turn"
    ]


def test_compression_lease_blocks_non_owner_but_allows_owner_flush(
    db: SessionDB,
) -> None:
    """Contract flipped by the watermark commit (#75316): a live lease no
    longer fences ordinary appends — both the owner's flush and a concurrent
    turn land immediately, and the commit-side watermark decides what
    survives compaction (see test_compression_watermark_commit.py)."""
    db.create_session("leased", source="webui")
    assert db.try_acquire_compression_lock("leased", "winner", ttl_seconds=60)

    db.append_message("leased", "user", "late concurrent turn")
    db.append_message(
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in db.get_messages("leased")] == [
        "late concurrent turn",
        "winner flush",
    ]
