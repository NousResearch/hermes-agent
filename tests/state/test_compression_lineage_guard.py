"""Regression tests for stale writes after a compression session split."""

from __future__ import annotations

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
    db.create_session("child-1", source="webui", parent_session_id="parent")
    db.end_session("child-1", "compression")
    db.create_session("child-2", source="webui", parent_session_id="child-1")
    db.end_session("child-2", "compression")
    db.create_session("live-tip", source="webui", parent_session_id="child-2")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "live-tip"
    assert child["parent_session_id"] == "child-2"
    assert child["ended_at"] is None


def test_find_live_compression_child_fails_closed_at_ambiguous_hop(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("child-1", source="webui", parent_session_id="parent")
    db.end_session("child-1", "compression")
    db.create_session("tip-a", source="webui", parent_session_id="child-1")
    db.create_session("tip-b", source="webui", parent_session_id="child-1")

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_fails_closed_for_live_and_rotated_siblings(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("rotated", source="webui", parent_session_id="parent")
    db.end_session("rotated", "compression")
    db.create_session("rotated-tip", source="webui", parent_session_id="rotated")
    db.create_session("stale-live-sibling", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_ignores_non_continuations_at_each_hop(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("child-1", source="webui", parent_session_id="parent")
    db.end_session("child-1", "compression")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="child-1",
        model_config={"_branched_from": "child-1"},
    )
    db.create_session(
        "delegate",
        source="webui",
        parent_session_id="child-1",
        model_config={"_delegate_from": "child-1"},
    )
    db.create_session("tool-child", source="tool", parent_session_id="child-1")
    db.create_session("live-tip", source="webui", parent_session_id="child-1")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "live-tip"


def test_find_live_compression_child_rejects_noncompression_dead_end(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("closed-child", source="webui", parent_session_id="parent")
    db.end_session("closed-child", "agent_close")

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


def test_find_live_compression_child_accepts_maximum_supported_depth(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    parent = "parent"
    for index in range(99):
        child = f"compressed-{index}"
        db.create_session(child, source="webui", parent_session_id=parent)
        db.end_session(child, "compression")
        parent = child
    db.create_session("deep-live-tip", source="webui", parent_session_id=parent)

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "deep-live-tip"


def test_find_live_compression_child_has_no_shallow_depth_cap(db: SessionDB) -> None:
    _compression_parent(db)
    parent = "parent"
    for index in range(128):
        child = f"compressed-{index}"
        db.create_session(child, source="webui", parent_session_id=parent)
        db.end_session(child, "compression")
        parent = child
    db.create_session("deep-tip", source="webui", parent_session_id=parent)

    tip = db.find_live_compression_child("parent")
    assert tip is not None
    assert tip["id"] == "deep-tip"


def test_find_live_compression_child_rejects_closed_canonical_sibling(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("closed", source="webui", parent_session_id="parent")
    db.end_session("closed", "agent_close")
    db.create_session("live", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_rejects_live_intermediate(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("live", source="webui", parent_session_id="parent")
    db.create_session("descendant", source="webui", parent_session_id="live")

    assert db.find_live_compression_child("parent") is None


@pytest.mark.parametrize("raw_config", ["{not-json", "[]", "null"])
def test_find_live_compression_child_rejects_malformed_metadata(
    db: SessionDB, raw_config: str
) -> None:
    _compression_parent(db)
    db.create_session("malformed", source="webui", parent_session_id="parent")
    with db._lock:
        assert db._conn is not None
        db._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (raw_config, "malformed"),
        )
        db._conn.commit()

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_rejects_nontext_metadata(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("blob", source="webui", parent_session_id="parent")
    with db._lock:
        assert db._conn is not None
        db._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?", (b"{}", "blob")
        )
        db._conn.commit()

    assert db.find_live_compression_child("parent") is None


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from"])
def test_find_live_compression_child_rejects_invalid_decoy(
    db: SessionDB, marker: str
) -> None:
    _compression_parent(db)
    db.create_session("live", source="webui", parent_session_id="parent")
    db.create_session(
        "false-decoy",
        source="webui",
        parent_session_id="parent",
        model_config={marker: "wrong-parent"},
    )

    assert db.find_live_compression_child("parent") is None


def test_find_live_compression_child_fails_closed_when_ambiguous(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child-a", source="webui", parent_session_id="parent")
    db.create_session("child-b", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None




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
    db.create_session("leased", source="webui")
    assert db.try_acquire_compression_lock("leased", "winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="being compressed"):
        db.append_message("leased", "user", "late stale turn")

    db.append_message(
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in db.get_messages("leased")] == ["winner flush"]
