"""Regression tests for iterative context-summary continuity."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
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


def _messages_with_merged_handoff(summary_body: str, prior_tail: str):
    merged = {
        "role": "user",
        "content": (
            f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{prior_tail}\n\n"
            f"{_MERGED_SUMMARY_DELIMITER}\n\n"
            f"{SUMMARY_PREFIX}\n{summary_body}\n\n{_SUMMARY_END_MARKER}"
        ),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    messages = _messages_with_handoff(summary_body)
    messages[1] = merged
    return messages


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


def test_recompression_drops_prior_protected_handoff_from_output():
    """Repeated compression must not preserve stale handoff bubbles forever."""
    compressor = _compressor()
    old_summary = "DUPLICATE-HANDOFF-BODY unique old facts"

    with patch.object(
        compressor,
        "_generate_summary",
        return_value=ContextCompressor._with_summary_prefix(
            "updated summary with old facts folded in"
        ),
    ):
        result = compressor.compress(_messages_with_handoff(old_summary))

    joined = "\n".join(str(message.get("content", "")) for message in result)
    assert old_summary not in joined
    assert joined.count(SUMMARY_PREFIX) == 1
    assert "updated summary with old facts folded in" in joined


def test_legacy_string_merged_handoff_preserves_real_tail_text():
    """Pre-delimiter string handoffs still unwrap content after the end marker."""
    message = {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\nold summary\n\n"
            f"{_SUMMARY_END_MARKER}\n\nreal tail message"
        ),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    result = ContextCompressor._strip_context_summary_handoff_message(message)

    assert result == {"role": "user", "content": "real tail message"}


def test_recompression_of_current_merged_handoff_preserves_prior_tail_once():
    """Current merged handoffs lose only stale summary data on recompression."""
    compressor = _compressor()
    old_summary = "CURRENT-MERGED-OLD-SUMMARY unique continuity facts"
    prior_tail = "PRESERVED-PRIOR-TAIL real user content"

    with patch.object(
        compressor,
        "_generate_summary",
        return_value=ContextCompressor._with_summary_prefix(
            "fresh replacement summary"
        ),
    ):
        result = compressor.compress(
            _messages_with_merged_handoff(old_summary, prior_tail)
        )

    joined = "\n".join(str(message.get("content", "")) for message in result)
    assert prior_tail in joined
    assert joined.count(prior_tail) == 1
    assert old_summary not in joined
    assert joined.count(SUMMARY_PREFIX) == 1
    assert "fresh replacement summary" in joined


def test_current_multimodal_merged_handoff_preserves_original_blocks():
    """Unwrapping current list content must retain text and image blocks."""
    prior_text = {"type": "text", "text": "real multimodal tail"}
    prior_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"},
            prior_text,
            prior_image,
            {
                "type": "text",
                "text": (
                    f"\n\n{_MERGED_SUMMARY_DELIMITER}\n\n"
                    f"{SUMMARY_PREFIX}\nstale summary\n\n{_SUMMARY_END_MARKER}"
                ),
            },
        ],
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    result = ContextCompressor._strip_context_summary_handoff_message(message)

    assert result == {
        "role": "user",
        "content": [prior_text, prior_image],
    }


def test_legacy_multimodal_merged_handoff_preserves_original_blocks():
    """Persisted pre-delimiter list handoffs must not lose their real tail."""
    prior_text = {"type": "text", "text": "legacy real tail"}
    prior_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,BBBB"},
    }
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"{SUMMARY_PREFIX}\nlegacy stale summary\n\n"
                    f"{_SUMMARY_END_MARKER}\n\n"
                ),
            },
            prior_text,
            prior_image,
        ],
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    result = ContextCompressor._strip_context_summary_handoff_message(message)

    assert result == {
        "role": "user",
        "content": [prior_text, prior_image],
    }
