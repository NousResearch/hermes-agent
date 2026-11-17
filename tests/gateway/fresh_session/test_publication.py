"""A fresh-topic handoff publishes its successor and existing binding atomically."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

# Bootstrap probes the checkout's git directory, which can live under the operator's home;
# finish that entry-point initialization before the fixture-scoped home-I/O guard.
import hermes_bootstrap

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key
from gateway.fresh_session.publication import publish
from hermes_state import SessionDB


@pytest.fixture
def topic(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = SessionSource(Platform.TELEGRAM, "-100123", chat_type="group", user_id="456",
                           thread_id="789", profile="research")
    key = build_session_key(source)
    config = {"provider": "custom", "base_url": "http://127.0.0.1:9/v1",
              "reasoning_effort": "high", "service_tier": "priority",
              "safe_mode": True, "disabled_toolsets": ["browser"]}
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("parent", source="telegram", model="test-model", model_config=config,
                          system_prompt="Existing policy: ask before publishing.", cwd=str(tmp_path),
                          profile_name=source.profile, transport_profile="receiving-bot",
                          user_id=source.user_id, session_key=key, chat_id=source.chat_id,
                          chat_type=source.chat_type, thread_id=source.thread_id,
                          origin_json=json.dumps(source.to_dict()), display_name="Synthetic topic")
        db.update_session_tool_names("parent", ["terminal", "session_search"])
        db.append_message("parent", "user", "Keep the entire predecessorword transcript.")
        db.append_message("parent", "assistant", "Inspecting the workspace.", tool_calls=[{
            "id": "call-1", "type": "function",
            "function": {"name": "terminal", "arguments": '{"command":"pwd"}'}}])
        db.append_message("parent", "tool", "Synthetic workspace output.",
                          tool_name="terminal", tool_call_id="call-1")
        db.append_message("parent", "assistant", "The final answer is ready.")
        assert db.bind_telegram_topic(chat_id=source.chat_id, thread_id=source.thread_id,
                                      user_id=source.user_id, session_key=key, session_id="parent",
                                      managed_mode="manual", profile_name=source.profile)
        assert db.try_acquire_session_turn_lease("parent", "turn-owner")
        yield SimpleNamespace(db=db, source=source, key=key, config=config, home=home,
                              parent=db.get_session("parent"), messages=db.get_messages("parent"))
    finally:
        db.close()


def _publish(topic, *, child_id="child", holder="turn-owner"):
    return publish(topic.db, parent_id="parent", child_id=child_id,
                   summary="Continue the synthetic task; verification remains pending.",
                   final_response="The final answer is ready.", source=topic.source,
                   session_key=topic.key, holder=holder)


def _binding(topic, db=None):
    return (db or topic.db).get_telegram_topic_binding(
        chat_id=topic.source.chat_id, thread_id=topic.source.thread_id,
        profile_name=topic.source.profile)


def _assert_parent_unchanged(topic):
    assert topic.db.get_session("parent") == topic.parent
    assert topic.db.get_messages("parent") == topic.messages
    assert topic.db.get_session("child") is None


def test_publication_preserves_policy_routing_and_searchable_parent(topic, monkeypatch):
    binding = _binding(topic)
    insert = topic.db._insert_message_rows
    observations = []

    def inspect_uncommitted(conn, session_id, messages):
        result = insert(conn, session_id, messages)
        reader = sqlite3.connect(topic.db.db_path)
        try:
            observations.append((
                reader.execute("SELECT ended_at FROM sessions WHERE id='parent'").fetchone()[0],
                reader.execute("SELECT id FROM sessions WHERE id='child'").fetchone(),
                reader.execute("SELECT session_id FROM telegram_dm_topic_bindings").fetchone()[0]))
        finally:
            reader.close()
        return result

    monkeypatch.setattr(topic.db, "_insert_message_rows", inspect_uncommitted)
    assert _publish(topic) is True
    assert observations == [(None, None, "parent")]
    child = topic.db.get_session("child")
    for field in ("source", "model", "system_prompt", "tool_names", "cwd", "profile_name",
                  "transport_profile", "session_key", "user_id", "chat_id", "chat_type", "thread_id",
                  "origin_json", "display_name"):
        assert child[field] == topic.parent[field], field
    assert json.loads(child["model_config"]) == {
        **topic.config, "_reset_from": "parent", "_fresh_session_handoff": True}
    assert child["parent_session_id"] == "parent"
    assert child["ended_at"] is None
    assert (child["message_count"], child["tool_call_count"]) == (1, 0)
    parent = topic.db.get_session("parent")
    assert parent["ended_at"] is not None
    assert parent["end_reason"] == "session_handoff"
    assert (parent["message_count"], parent["tool_call_count"]) == (4, 1)
    assert topic.db.get_messages("parent") == topic.messages
    assert any(hit["session_id"] == "parent" for hit in topic.db.search_messages("predecessorword"))
    messages = topic.db.get_messages_as_conversation("child", include_ancestors=True)
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    for text in ("parent", "not new user authorization", "session_search",
                 "Continue the synthetic task", "The final answer is ready."):
        assert text in messages[0]["content"]
    published = _binding(topic)
    assert published["session_id"] == "child"
    assert published["session_key"] == topic.key
    for field in ("profile_name", "chat_id", "thread_id", "user_id", "managed_mode", "linked_at"):
        assert published[field] == binding[field]


def test_duplicate_intent_publishes_only_one_successor(topic):
    assert _publish(topic) is True
    assert _publish(topic, child_id="duplicate") is False
    assert topic.db.get_session("duplicate") is None
    assert _binding(topic)["session_id"] == "child"
    assert len(topic.db.get_messages("child")) == 1


def test_insert_failure_rolls_back_child_parent_and_binding(topic, monkeypatch):
    binding = _binding(topic)
    insert = topic.db._insert_message_rows

    def fail_after_insert(conn, session_id, messages):
        insert(conn, session_id, messages)
        raise RuntimeError("synthetic insert failure")

    with monkeypatch.context() as patch:
        patch.setattr(topic.db, "_insert_message_rows", fail_after_insert)
        with pytest.raises(RuntimeError, match="synthetic insert failure"):
            _publish(topic)
    _assert_parent_unchanged(topic)
    assert _binding(topic) == binding
    assert topic.db.session_turn_lease_expires_at("parent", "turn-owner") is not None
    assert _publish(topic) is True


def test_active_compression_lease_refuses_publication(topic):
    assert topic.db.try_acquire_compression_lock("parent", "compressor")
    assert _publish(topic) is False
    _assert_parent_unchanged(topic)
    assert _binding(topic)["session_id"] == "parent"


@pytest.mark.parametrize("lease", ["missing", "wrong-holder", "expired"])
def test_publication_requires_the_current_turn_lease(topic, lease):
    holder = "turn-owner"
    if lease == "missing":
        topic.db.release_session_turn_lease("parent", holder)
    elif lease == "wrong-holder":
        holder = "another-turn"
    else:
        topic.db._write_sql("UPDATE session_turn_leases SET expires_at=0")
    assert _publish(topic, holder=holder) is False
    _assert_parent_unchanged(topic)
    assert _binding(topic)["session_id"] == "parent"


@pytest.mark.parametrize("replacement", ["new-session", None])
def test_concurrent_new_or_topic_removal_wins_over_handoff(topic, replacement):
    if replacement:
        topic.db.create_session(replacement, source="telegram")
        assert topic.db.bind_telegram_topic(
            chat_id=topic.source.chat_id, thread_id=topic.source.thread_id,
            user_id=topic.source.user_id, session_key=topic.key, session_id=replacement,
            profile_name=topic.source.profile, expected_session_id="parent")
    else:
        assert topic.db.delete_telegram_topic_binding(
            chat_id=topic.source.chat_id, thread_id=topic.source.thread_id,
            profile_name=topic.source.profile) == 1
    assert _publish(topic) is False
    _assert_parent_unchanged(topic)
    binding = _binding(topic)
    if replacement:
        assert binding["session_id"] == replacement
    else:
        assert binding is None


def test_committed_binding_survives_restart_before_json_store_updates(topic):
    routing_file = topic.home / "sessions.json"
    stale_routes = {topic.key: {"session_id": "parent"}}
    routing_file.write_text(json.dumps(stale_routes))
    assert _publish(topic) is True
    topic.db.close()
    reopened = SessionDB(db_path=topic.home / "state.db")
    try:
        assert json.loads(routing_file.read_text()) == stale_routes
        assert _binding(topic, reopened)["session_id"] == "child"
        assert reopened.get_session("parent")["end_reason"] == "session_handoff"
        assert reopened.get_session("child")["ended_at"] is None
        assert len(reopened.get_messages("child")) == 1
    finally:
        reopened.close()


def test_retired_predecessor_rejects_late_writer_and_in_place_rescue(topic):
    from hermes_state_errors import SessionTurnLeaseLostError
    assert _publish(topic)
    with pytest.raises(SessionTurnLeaseLostError, match="fresh-session handoff"):
        topic.db.append_message("parent", "assistant", "late writer")
    with pytest.raises(SessionTurnLeaseLostError, match="fresh-session handoff"):
        topic.db.archive_and_compact("parent", [{"role": "assistant", "content": "late rescue"}])
    assert topic.db.get_messages("parent") == topic.messages


@pytest.mark.parametrize("kind", ["goal", "heartbeat", "loop"])
def test_live_automation_is_not_orphaned_by_handoff(topic, kind):
    topic.db.set_meta(f"{kind}:parent", json.dumps({"status": "active"}))
    assert _publish(topic) is False
    _assert_parent_unchanged(topic)


def test_handoff_advances_peer_generation_once_and_drops_session_yolo(topic):
    topic.db.update_session_meta('parent', json.dumps({**topic.config, 'yolo_mode': True}))
    before = topic.db.latest_conversation_boundary(topic.key, 'telegram')
    assert _publish(topic)
    after = topic.db.latest_conversation_boundary(topic.key, 'telegram')
    assert after != before and after is not None
    assert json.loads(topic.db.get_session('child')['model_config']).get('yolo_mode') is not True
    assert not _publish(topic, child_id='duplicate')
    assert topic.db.latest_conversation_boundary(topic.key, 'telegram') == after
