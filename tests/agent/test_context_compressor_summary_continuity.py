"""Regression tests for iterative context-summary continuity."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
)


def _compressor(protect_first_n: int = 1) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=protect_first_n,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _messages_with_handoff(summary_body: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def _messages_with_default_handoff(summary_body: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "original task before first compaction"},
        {"role": "assistant", "content": "original answer before first compaction"},
        {"role": "user", "content": "original follow-up before first compaction"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "user", "content": "new user turn after restart"},
        {"role": "assistant", "content": "new assistant work after restart"},
        {"role": "user", "content": "more new work after restart"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def test_existing_previous_summary_is_not_serialized_again_as_new_turn():
    """Same-process iterative compression should not feed the old handoff twice."""
    compressor = _compressor()
    old_summary = "OLD-SUMMARY-BODY unique continuity facts"
    compressor._previous_summary = old_summary

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_resume_rehydrates_previous_summary_from_handoff_message():
    """After restart/resume, the persisted handoff should regain summary identity."""
    compressor = _compressor()
    old_summary = "RESUMED-SUMMARY-BODY durable continuity facts"
    assert compressor._previous_summary is None

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert "TURNS TO SUMMARIZE:" not in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_handoff_in_protected_head_populates_previous_summary_before_update():
    """A resumed protected-head handoff should restore iterative-summary state."""
    compressor = _compressor()
    old_summary = "PROTECTED-HEAD-SUMMARY durable facts from before restart"
    seen_turns = []

    def fake_generate_summary(turns_to_summarize, focus_topic=None):
        seen_turns.extend(turns_to_summarize)
        return "new summary from resumed turns"

    with patch.object(compressor, "_generate_summary", side_effect=fake_generate_summary):
        compressor.compress(_messages_with_handoff(old_summary))

    assert compressor._previous_summary == old_summary
    assert seen_turns
    assert all(old_summary not in str(msg.get("content", "")) for msg in seen_turns)


def test_resume_handoff_in_protected_head_is_not_preserved_as_fossil():
    """After restart, a persisted handoff summary should decay head protection."""
    compressor = _compressor()
    old_summary = "RESTART-FOSSIL-SUMMARY durable facts from before restart"

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = compressor.compress(_messages_with_handoff(old_summary))

    assert compressor._previous_summary == "fresh summary"
    summary_messages = [
        msg for msg in result
        if ContextCompressor._has_compressed_summary_metadata(msg)
        or ContextCompressor._is_context_summary_content(msg.get("content"))
    ]
    assert len(summary_messages) == 1
    assert all(
        old_summary not in str(msg.get("content", ""))
        for msg in result
    )


def test_resume_handoff_after_default_protected_head_decays_initial_turns():
    """Default protect_first_n=3 should not fossilize old protected head turns."""
    compressor = _compressor(protect_first_n=3)
    old_summary = "DEFAULT-RESTART-SUMMARY durable facts from before restart"

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        result = compressor.compress(_messages_with_default_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert prompt.count(old_summary) == 1
    assert "original task before first compaction" in prompt
    assert "original answer before first compaction" in prompt
    assert "original follow-up before first compaction" in prompt
    assert f"[ASSISTANT]: {SUMMARY_PREFIX}" not in prompt
    assert compressor._previous_summary == "fresh summary"
    assert all(
        "original task before first compaction" not in str(msg.get("content", ""))
        for msg in result
    )
    assert all(
        "original answer before first compaction" not in str(msg.get("content", ""))
        for msg in result
    )
    assert all(
        old_summary not in str(msg.get("content", ""))
        for msg in result
    )


def test_restart_stacked_handoffs_fold_stray_head_and_collapse_to_single_summary():
    """Stacked restart summaries should keep stray head turns as new input."""
    compressor = _compressor(protect_first_n=3)
    old_summary = "OLD-STACKED-SUMMARY earlier compacted facts"
    newer_summary = "NEW-STACKED-SUMMARY already incorporates prior compacted facts"

    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "FOSSIL-HEAD-TURN live detail before summary"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{newer_summary}"},
        {"role": "assistant", "content": "work after restart"},
        {"role": "user", "content": "more work after restart"},
        {"role": "assistant", "content": "tail answer"},
        {"role": "user", "content": "active tail request"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        result = compressor.compress(msgs)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert prompt.count(newer_summary) == 1
    assert "FOSSIL-HEAD-TURN live detail before summary" in prompt
    assert old_summary not in prompt
    assert f"[ASSISTANT]: {SUMMARY_PREFIX}" not in prompt
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt
    summary_messages = [
        msg for msg in result
        if ContextCompressor._is_context_summary_message(msg)
    ]
    assert len(summary_messages) == 1
    assert all(old_summary not in str(msg.get("content", "")) for msg in result)
    assert all(newer_summary not in str(msg.get("content", "")) for msg in result)
    assert all("FOSSIL-HEAD-TURN" not in str(msg.get("content", "")) for msg in result)


def test_metadata_summary_decay_also_rehydrates_previous_summary():
    """Metadata-only in-process summaries should decay and rehydrate together."""
    compressor = _compressor(protect_first_n=3)

    msgs = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": "metadata-only prior summary",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "user", "content": "new work"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "tail request"},
        {"role": "assistant", "content": "tail answer"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        compressor.compress(msgs)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "metadata-only prior summary" in prompt
    assert compressor._previous_summary == "fresh summary"
