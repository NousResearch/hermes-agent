"""Compaction grounding must not treat async-delegation completion notices as the user's request."""

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import _is_real_user_message
from agent.prompt_builder import steer_user_row

NOTICE = {
    "role": "user",
    "content": "[ASYNC DELEGATION BATCH COMPLETE 2 tasks]\nsubagent finished the audit",
    "display_kind": "async_delegation_complete",
}
MESSAGES = [
    {"role": "user", "content": "migrate the billing service to postgres"},
    {"role": "assistant", "content": "delegated"},
    NOTICE,
]


def test_async_delegation_notice_is_not_a_real_user_message():
    assert not _is_real_user_message(NOTICE)


def test_snapshot_quotes_the_earlier_human_request():
    snapshot = ContextCompressor._latest_user_task_snapshot(MESSAGES)
    assert snapshot is not None
    assert "migrate the billing service" in snapshot
    assert "ASYNC DELEGATION" not in snapshot


def test_steer_row_still_counts_as_human_input():
    steer = steer_user_row("focus on the error handling")
    assert _is_real_user_message(steer)
    snapshot = ContextCompressor._latest_user_task_snapshot([*MESSAGES, steer])
    assert snapshot is not None and "error handling" in snapshot
