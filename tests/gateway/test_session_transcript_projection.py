"""Regression coverage for compaction carriers in gateway transcript writes (#137540)."""

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from gateway.session_transcript import transcript_append_kwargs


STANDALONE_SUMMARY = (
    f"{SUMMARY_PREFIX}\n\n"
    f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
    f"{_SUMMARY_END_MARKER}"
)
MERGED_CARRIER = (
    f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
    "The migration is complete.\n\n"
    f"{_MERGED_SUMMARY_DELIMITER}\n\n"
    f"{STANDALONE_SUMMARY}"
)


def test_transcript_writer_hides_pure_compaction_handoff_but_preserves_api_replay():
    kwargs = transcript_append_kwargs(
        "session-1", {"role": "assistant", "content": STANDALONE_SUMMARY}
    )

    assert kwargs["content"] == ""
    assert kwargs["display_kind"] == "hidden"
    assert kwargs["api_content"] == STANDALONE_SUMMARY


def test_transcript_writer_keeps_only_real_tail_content_visible_from_merged_handoff():
    kwargs = transcript_append_kwargs(
        "session-1", {"role": "assistant", "content": MERGED_CARRIER}
    )

    assert kwargs["content"] == "The migration is complete."
    assert kwargs["display_kind"] is None
    assert kwargs["api_content"] == MERGED_CARRIER
