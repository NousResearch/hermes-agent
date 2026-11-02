"""Display projection trusts durable provenance and retains user metadata."""

import copy

import pytest

from agent.compaction_display import project_compaction_message_for_display
from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
)
from agent.conversation_compression import _extract_steer_text_from_message
from agent.prompt_builder import steer_user_row
from hermes_state import SessionDB


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
@pytest.mark.parametrize(
    "content",
    [
        "[CONTEXT SUMMARY]: this is a literal example",
        f"{SUMMARY_PREFIX}\nquoted documentation\n{_SUMMARY_END_MARKER}",
        [{"type": "text", "text": "[CONTEXT SUMMARY]: literal tool output"}],
    ],
    ids=["prefix", "paired-markers", "content-blocks"],
)
def test_summary_lookalikes_remain_visible(role, content):
    message = {"role": role, "content": content}
    original = copy.deepcopy(message)
    projected = project_compaction_message_for_display(message)
    assert projected == original
    assert projected is not message
    assert message == original


@pytest.mark.parametrize("force_user_leading", [False, True])
def test_merged_steer_retains_identity_after_storage(tmp_path, force_user_leading):
    text = "focus on the failing request"
    message = steer_user_row(text)
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor._summary_has_user_turn = True
    compressor._merge_summary_into_tail_row(
        message, SUMMARY_PREFIX + "\nold context", "user", force_user_leading
    )
    db = SessionDB(tmp_path / "state.db")
    try:
        sid = db.create_session("display-projection", "cli")
        db.replace_messages(sid, [message])
        stored = db.get_messages(sid)[0]
        original = copy.deepcopy(stored)
        projected = project_compaction_message_for_display(stored)
        assert projected["display_kind"] == "steer"
        assert _extract_steer_text_from_message(projected) == text
        assert stored == original
        assert (
            project_compaction_message_for_display({
                "role": "user",
                "content": SUMMARY_PREFIX + "\nold context",
                "_compressed_summary": True,
            })
            is None
        )
    finally:
        db.close()
