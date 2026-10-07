"""A replayed merged row keeps the unanswered request and exact wire sidecar."""

from agent.agent_runtime_helpers import repair_message_sequence
from agent.session_persistence import SessionPersistenceMixin, _db_flush_row


def _agent_persisting(override):
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 1
    agent._persist_user_message_override = override
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    return agent


def _live_agent():
    """FIXTURE: an agent carrying a session store so repair_message_sequence runs the LIVE
    user fold. These are upstream in-base tests (``87cefde203``) whose bare-list fold
    expectation conflicts with #63298's wire-boundary contract (bare repair surfaces stay
    verbatim); re-targeted to the live fold where the merged-override machinery lives.
    Assertions byte-unchanged. Contract fork — maintainers may flip."""

    class _Agent:
        _session_db = True

    return _Agent()


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
    assert repair_message_sequence(_live_agent(), messages) == 1
    merged_wire = messages[1]["content"]
    # The recorded prefix is bookkeeping: the estimate prices only the row's content.
    assert estimate_messages_tokens_rough([messages[1]]) == estimate_messages_tokens_rough(
        [{"role": "user", "content": merged_wire}]
    )
    agent = _agent_persisting(clean)
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
        assert repair_message_sequence(_live_agent(), messages) == len(prior)
        expected = "\n\n".join(prior)
        assert messages[1]["content"] == expected
        agent = _agent_persisting(clean)
        agent._apply_persist_user_message_override(messages)
        assert messages[1]["content"] == expected
        row = _db_flush_row(agent, messages[1], True)
        assert row["content"] == expected
        agent._apply_persist_user_message_override(messages)
        assert messages[1]["content"] == expected
        assert _db_flush_row(agent, messages[1], True)["content"] == expected
        # A later non-empty override on the same row joins like the merge does.
        assert _db_flush_row(_agent_persisting("later"), messages[1], True)["content"] == expected + "\n\nlater"
