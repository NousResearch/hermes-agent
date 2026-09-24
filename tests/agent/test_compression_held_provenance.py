"""Compaction provenance for repaired turns (regression for #121734)."""

from agent.agent_runtime_helpers import _merge_consecutive_assistants, _merge_consecutive_users


def test_repaired_turn_keeps_every_durable_source() -> None:
    users, count = _merge_consecutive_users([
        {"role": "user", "content": "first", "_row_id": 10},
        {"role": "user", "content": "second", "_row_id": 12},
        {"role": "user", "content": "not persisted"},
    ])
    assert count == 2
    assert users[0]["content"] == "first\n\nsecond\n\nnot persisted"
    assert users[0]["_source_row_ids"] == (10, 12)

    assistants, count = _merge_consecutive_assistants([
        {"role": "assistant", "content": "one", "_row_id": 20},
        {"role": "assistant", "content": "two", "_row_id": 22},
    ])
    assert count == 1
    assert assistants[0]["_source_row_ids"] == (20, 22)
