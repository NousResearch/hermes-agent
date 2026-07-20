"""Atomic SessionDB compression-rotation and lineage-quota contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hermes_state import SessionDB


def _db(tmp_path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _rotate(
    db: SessionDB,
    *,
    parent: str = "parent",
    child: str = "child",
    consume_auto_handoff: bool = False,
    max_auto_handoffs: int | None = None,
) -> int:
    return db.rotate_session_for_compression(
        parent_session_id=parent,
        child_session_id=child,
        source="cli",
        messages=[
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "ready"},
        ],
        model="test/model",
        model_config={"provider": "test"},
        system_prompt="system",
        consume_auto_handoff=consume_auto_handoff,
        max_auto_handoffs=max_auto_handoffs,
    )


def test_atomic_rotation_inserts_child_messages_and_closes_parent(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session("parent", source="cli", model_config={"parent": True})

        assert _rotate(db) == 0

        parent = db.get_session("parent")
        child = db.get_session("child")
        assert parent is not None and parent["ended_at"] is not None
        assert parent["end_reason"] == "compression"
        assert child is not None and child["ended_at"] is None
        assert child["parent_session_id"] == "parent"
        assert [row["content"] for row in db.get_messages("child")] == [
            "continue",
            "ready",
        ]
    finally:
        db.close()


def test_child_creation_failure_leaves_parent_active_and_no_partial_messages(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session("parent", source="cli")
        db._conn.execute(
            """CREATE TRIGGER fail_child_create
               BEFORE INSERT ON sessions
               WHEN NEW.id = 'child'
               BEGIN SELECT RAISE(ABORT, 'forced child creation failure'); END"""
        )

        with pytest.raises(Exception, match="forced child creation failure"):
            _rotate(db)

        parent = db.get_session("parent")
        assert parent is not None and parent["ended_at"] is None
        assert db.get_session("child") is None
        assert db.get_messages("child") == []
    finally:
        db.close()


def test_child_message_failure_rolls_back_child_and_parent_end(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session("parent", source="cli")
        db._conn.execute(
            """CREATE TRIGGER fail_child_message
               BEFORE INSERT ON messages
               WHEN NEW.session_id = 'child'
               BEGIN SELECT RAISE(ABORT, 'forced child message failure'); END"""
        )

        with pytest.raises(Exception, match="forced child message failure"):
            _rotate(db)

        parent = db.get_session("parent")
        assert parent is not None and parent["ended_at"] is None
        assert db.get_session("child") is None
        assert db.get_messages("child") == []
    finally:
        db.close()


def test_parent_end_failure_rolls_back_inserted_child_and_keeps_parent_active(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session("parent", source="cli")
        db._conn.execute(
            """CREATE TRIGGER fail_parent_end
               BEFORE UPDATE OF ended_at ON sessions
               WHEN OLD.id = 'parent' AND NEW.ended_at IS NOT NULL
               BEGIN SELECT RAISE(ABORT, 'forced parent end failure'); END"""
        )

        with pytest.raises(Exception, match="forced parent end failure"):
            _rotate(db)

        parent = db.get_session("parent")
        assert parent is not None and parent["ended_at"] is None
        assert db.get_session("child") is None
        assert db.get_messages("child") == []
    finally:
        db.close()


def test_atomic_rotation_rejects_ended_or_missing_parent(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session("parent", source="cli")
        db.end_session("parent", "user_exit")

        with pytest.raises(ValueError, match="active parent"):
            _rotate(db)
        assert db.get_session("child") is None

        with pytest.raises(ValueError, match="active parent"):
            _rotate(db, parent="missing", child="other-child")
        assert db.get_session("other-child") is None
    finally:
        db.close()


def test_quota_is_persisted_and_enforced_in_the_rotation_transaction(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session(
            "parent",
            source="cli",
            model_config={"_auto_handoff_count": 0},
        )

        assert _rotate(
            db,
            consume_auto_handoff=True,
            max_auto_handoffs=2,
        ) == 1
        assert db.get_auto_handoff_count("child") == 1

        assert _rotate(
            db,
            parent="child",
            child="grandchild",
            consume_auto_handoff=True,
            max_auto_handoffs=2,
        ) == 2
        assert db.get_auto_handoff_count("grandchild") == 2

        with pytest.raises(ValueError, match="quota exhausted"):
            _rotate(
                db,
                parent="grandchild",
                child="exhausted-child",
                consume_auto_handoff=True,
                max_auto_handoffs=2,
            )
        grandchild = db.get_session("grandchild")
        assert grandchild is not None and grandchild["ended_at"] is None
        assert db.get_session("exhausted-child") is None
    finally:
        db.close()


def test_generic_runtime_metadata_updates_cannot_reset_lineage_quota(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session(
            "session",
            source="acp",
            model_config={"_auto_handoff_count": 2, "cwd": "/old"},
        )

        db.update_session_meta(
            "session",
            json.dumps({"cwd": "/new", "provider": "openai"}),
        )

        row = db.get_session("session")
        config = json.loads(row["model_config"])
        assert config["cwd"] == "/new"
        assert config["provider"] == "openai"
        assert config["_auto_handoff_count"] == 2
        assert db.get_auto_handoff_count("session") == 2
    finally:
        db.close()


def test_generic_runtime_metadata_cannot_override_reserved_lineage_quota(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session(
            "session",
            source="gateway",
            model_config={"_auto_handoff_count": 2, "provider": "old"},
        )

        db.update_session_meta(
            "session",
            json.dumps(
                {
                    "_auto_handoff_count": 0,
                    "provider": "anthropic",
                    "gateway_runtime": {"chat_id": "123"},
                }
            ),
        )

        config = json.loads(db.get_session("session")["model_config"])
        assert config == {
            "_auto_handoff_count": 2,
            "provider": "anthropic",
            "gateway_runtime": {"chat_id": "123"},
        }
        assert db.get_auto_handoff_count("session") == 2
    finally:
        db.close()


def test_prompt_quota_consumption_is_atomic_and_does_not_exceed_maximum(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session(
            "parent",
            source="cli",
            model_config={"_auto_handoff_count": 0},
        )
        barrier = Barrier(2)

        def consume():
            barrier.wait()
            try:
                return db.consume_auto_handoff(
                    "parent",
                    max_auto_handoffs=1,
                )
            except ValueError as exc:
                return str(exc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: consume(), range(2)))

        assert results.count(1) == 1
        assert sum("quota exhausted" in str(result) for result in results) == 1
        assert db.get_auto_handoff_count("parent") == 1
    finally:
        db.close()
