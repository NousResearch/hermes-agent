"""A replayed merged row keeps the unanswered request and exact wire sidecar."""

from agent.agent_runtime_helpers import repair_message_sequence
from agent.session_persistence import SessionPersistenceMixin, _db_flush_row


def test_replay_flush_row_keeps_prefix_and_exact_api_sidecar():
    messages = [
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": "please deploy build 42 to staging"},
        {"role": "user", "content": "[03:00] can you also run the smoke tests"},
    ]
    assert repair_message_sequence(None, messages) == 1
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_override = "can you also run the smoke tests"
    row = _db_flush_row(agent, messages[1], True)
    assert row["content"] == (
        "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    )
    assert row["api_content"] == messages[1]["content"]
