"""A persisted unanswered user turn survives a new turn's clean-text override."""

from agent.agent_runtime_helpers import repair_message_sequence
from agent.session_persistence import SessionPersistenceMixin, durable_user_row_content


def _merged_turn():
    messages = [
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": "please deploy build 42 to staging"},
        {"role": "user", "content": "[03:00] can you also run the smoke tests"},
    ]
    assert repair_message_sequence(None, messages) == 1
    assert len(messages) == 2
    return messages


def test_live_override_keeps_unanswered_prefix():
    messages = _merged_turn()
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 1
    agent._persist_user_message_override = "can you also run the smoke tests"
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    agent._apply_persist_user_message_override(messages)
    assert messages[1]["content"] == (
        "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    )


def test_replay_row_override_keeps_unanswered_prefix():
    messages = _merged_turn()
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_override = "can you also run the smoke tests"
    content, api_content = durable_user_row_content(
        agent, messages[1], messages[1]["content"], None
    )
    assert (
        content
        == "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    )
    assert api_content == messages[1]["content"]


def test_non_merged_turn_override_still_replaces_the_entire_wire_text():
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "clean question"
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    messages = [{"role": "user", "content": "[03:00] clean question"}]
    agent._apply_persist_user_message_override(messages)
    assert messages[0]["content"] == "clean question"


def test_merged_turn_marker_is_not_sent_to_provider():
    from agent.transports.chat_completions import _sanitize_message

    msg = _merged_turn()[1]
    sent = _sanitize_message(msg, strip_extra_content=False)
    assert sent is not None
    assert "_merged_turn_prefix" not in sent


def test_replay_after_live_override_keeps_unanswered_prefix():
    messages = _merged_turn()
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 1
    agent._persist_user_message_override = "can you also run the smoke tests"
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    agent._apply_persist_user_message_override(messages)
    content, _ = durable_user_row_content(
        agent, messages[1], messages[1]["content"], None
    )
    assert (
        content
        == "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    )


def test_repair_after_replay_preserves_the_oldest_unanswered_turn():
    messages = [
        {"role": "user", "content": "first unanswered"},
        {"role": "user", "content": "second unanswered"},
        {"role": "user", "content": "[03:00] new question"},
    ]
    assert repair_message_sequence(None, messages) == 2
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "new question"
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    agent._apply_persist_user_message_override(messages)
    assert (
        messages[0]["content"]
        == "first unanswered\n\nsecond unanswered\n\nnew question"
    )


def test_replay_flush_row_keeps_prefix_and_exact_api_sidecar():
    from agent.session_persistence import _db_flush_row

    messages = _merged_turn()
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_override = "can you also run the smoke tests"
    row = _db_flush_row(agent, messages[1], True)
    assert (
        row["content"]
        == "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    )
    assert row["api_content"] == messages[1]["content"]
