"""Gateway peer writes and recovery never promote delegate children (#134522)."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("child_source", ["subagent", "delegate"])
def test_gateway_peer_write_preserves_delegate_provenance(tmp_path, child_source):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="discord", session_key="agent:main:discord:thread:one")
    db.create_session("child", source=child_source, parent_session_id="parent")

    db.record_gateway_session_peer(
        "child", source="discord", session_key="agent:main:discord:thread:one",
        chat_id="one", chat_type="thread",
    )

    row = db.get_session("child")
    assert row["source"] == row["created_source"] == child_source
    assert row["session_key"] is None


def test_gateway_peer_write_rejects_delegate_marker_even_if_birth_source_is_platform(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="discord")
    db.create_session(
        "child", source="discord", parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )

    db.record_gateway_session_peer(
        "child", source="discord", session_key="agent:main:discord:thread:one",
        chat_id="one", chat_type="thread",
    )

    row = db.get_session("child")
    assert row["created_source"] == "discord"
    assert row["session_key"] is None


@pytest.mark.parametrize("child_source", ["subagent", "delegate"])
def test_peer_recovery_excludes_an_already_promoted_child(tmp_path, child_source):
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "agent:main:discord:thread:one"
    db.create_session("parent", source="discord", session_key=key, chat_id="one", chat_type="thread")
    db.create_session("child", source=child_source, parent_session_id="parent")
    db._write_sql(
        "UPDATE sessions SET source = ?, session_key = ?, chat_id = ?, chat_type = ? WHERE id = ?",
        ("discord", key, "one", "thread", "child"),
    )

    recovered = db.find_latest_gateway_session_for_peer(
        source="discord", session_key=key, chat_id="one", chat_type="thread",
    )

    assert recovered is not None and recovered["id"] == "parent"
    by_origin = db.find_session_by_origin(platform="discord", chat_id="one")
    assert by_origin == "parent"


@pytest.mark.parametrize("child_source", ["subagent", "delegate"])
def test_legacy_child_peer_cleanup_restores_execution_identity_only(tmp_path, child_source):
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "agent:main:discord:thread:one"
    db.create_session("parent", source="discord", session_key=key, chat_id="one")
    db.create_session("child", source=child_source, parent_session_id="parent")
    db._write_sql(
        "UPDATE sessions SET source = ?, session_key = ?, user_id = ?, chat_id = ?, "
        "chat_type = ?, thread_id = ?, origin_json = ? WHERE id = ?",
        ("discord", key, "sender", "one", "thread", "one", "{}", "child"),
    )

    assert db.clear_poisoned_delegate_gateway_peer("child", key)

    child = db.get_session("child")
    assert child["source"] == child["created_source"] == child_source
    assert all(child[field] is None for field in
               ("session_key", "user_id", "chat_id", "chat_type", "thread_id", "origin_json"))
    assert child["ended_at"] is None
    assert db.get_session("parent")["session_key"] == key
    assert not db.clear_poisoned_delegate_gateway_peer("parent", key)
