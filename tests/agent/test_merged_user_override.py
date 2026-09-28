"""A replayed merged row keeps the unanswered request and exact wire sidecar."""

from agent.agent_runtime_helpers import repair_message_sequence
from agent.session_persistence import SessionPersistenceMixin, _db_flush_row


def test_replay_flush_row_keeps_prefix_and_exact_api_sidecar():
    from agent.model_metadata import estimate_messages_tokens_rough
    from gateway.message_timestamps import strip_leading_message_timestamps

    wire = "[2026-04-13T17:02:06+0200] can you also run the smoke tests"
    clean, _epoch = strip_leading_message_timestamps(wire)
    messages = [
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": "please deploy build 42 to staging"},
        {"role": "user", "content": wire},
    ]
    assert repair_message_sequence(None, messages) == 1
    merged_wire = messages[1]["content"]
    # The recorded prefix is bookkeeping: the estimate prices only the row's content.
    assert estimate_messages_tokens_rough([messages[1]]) == estimate_messages_tokens_rough(
        [{"role": "user", "content": merged_wire}]
    )
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 1
    agent._persist_user_message_override = clean
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    expected = "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    row = _db_flush_row(agent, messages[1], True)
    assert row["content"] == expected
    assert row["api_content"] == merged_wire
    agent._apply_persist_user_message_override(messages)
    assert messages[1]["content"] == expected


def test_empty_gateway_timestamp_turn_preserves_all_unanswered_rows_on_live_and_replay():
    from gateway.message_timestamps import strip_leading_message_timestamps

    clean, _epoch = strip_leading_message_timestamps("[2026-04-13T17:02:06+0200]")
    assert clean == ""
    for prior in (("request R",), ("request R", "request U")):
        messages = [{"role": "assistant", "content": "previous response"}]
        messages.extend({"role": "user", "content": text} for text in prior)
        messages.append({"role": "user", "content": clean})
        assert repair_message_sequence(None, messages) == len(prior)
        expected = "\n\n".join(prior)
        assert messages[1]["content"] == expected
        agent = object.__new__(SessionPersistenceMixin)
        agent._persist_user_message_idx = 1
        agent._persist_user_message_override = clean
        agent._persist_user_message_timestamp = None
        agent._persist_user_message_platform_id = None
        agent._apply_persist_user_message_override(messages)
        assert messages[1]["content"] == expected
        row = _db_flush_row(agent, messages[1], True)
        assert row["content"] == expected
        agent._apply_persist_user_message_override(messages)
        assert messages[1]["content"] == expected
        assert _db_flush_row(agent, messages[1], True)["content"] == expected
