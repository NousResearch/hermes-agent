"""Transport contract for redacted, bounded native compression summary calls."""

import re
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, pin_summary_route
from agent.auxiliary_client import AuxiliaryExplicitCancellation


def _response(text="## Historical Task Snapshot\nNone. This session contains no user-authored turns.\n\n## Goal\nSynthetic checkpoint."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.choices[0].finish_reason = "stop"
    return response


@pytest.mark.parametrize("before,after", [(1_003, 27_011), (12_007, 19_003), (24_517, 6_019)])
def test_full_redacted_body_reaches_one_prompt_at_irregular_positions(before, after):
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    body = "alpha:" + "a" * before + " important=47 corrected=49 error=invalid " + "b" * after + ":omega"
    with patch("agent.context_compressor.call_llm", return_value=_response()) as call:
        result = compressor._generate_summary([{"role": "tool", "tool_call_id": "t1", "content": body}])
    assert result is not None
    assert call.call_count == 1
    prompt = call.call_args.kwargs["messages"][0]["content"]
    assert body in prompt


def test_global_budget_transmits_every_source_character_in_order_without_sampling():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    records = [f"item={i:04d} " + chr(65 + (i % 26)) * 5_000 for i in range(48)]
    messages = [{"role": "tool", "tool_call_id": f"t{i}", "content": record} for i, record in enumerate(records)]
    expected = "\n\n".join(compressor._serialize_records_for_summary(messages, full_tool_results=True))
    seen = []

    def answer(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        assert len(prompt) <= compressor._SUMMARY_INPUT_MAX_CHARS
        match = re.search(r"(?:TURNS TO SUMMARIZE|NEW TURNS TO INCORPORATE):\n(.*?)(?:\n\nUse this exact structure:|\n\nUpdate the summary using this exact structure\.)", prompt, re.DOTALL)
        assert match
        seen.append(match.group(1))
        return _response()

    with patch("agent.context_compressor.call_llm", side_effect=answer) as call:
        assert compressor._generate_summary(messages) is not None
    assert 1 < call.call_count <= 8
    assert "".join(seen) == expected


def test_budget_failure_preserves_previous_summary_and_never_claims_success():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor._previous_summary = "earlier checkpoint"
    compressor._SUMMARY_INPUT_MAX_CHARS = 8000
    messages = [{"role": "tool", "tool_call_id": "t", "content": "x" * 30_000}]
    with patch("agent.context_compressor.call_llm", side_effect=[_response(), RuntimeError("provider unavailable")]) as call:
        assert compressor._generate_summary(messages) is None
    assert call.call_count == 2
    assert compressor._previous_summary == "earlier checkpoint"
    assert compressor._last_summary_error == "provider unavailable"


def test_chunk_ceiling_is_fail_closed_without_intermediate_commit():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor._previous_summary = "earlier checkpoint"
    compressor._SUMMARY_INPUT_MAX_CHARS = 8000
    compressor._MAX_COMPLETE_SUMMARY_CHUNKS = 2
    messages = [{"role": "tool", "tool_call_id": "t", "content": "x" * 30_000}]
    with patch("agent.context_compressor.call_llm", return_value=_response()) as call:
        assert compressor._generate_summary(messages) is None
    assert call.call_count == 2
    assert compressor._previous_summary == "earlier checkpoint"
    assert compressor._last_summary_truncated_failure is True


def test_one_use_pinned_route_never_switches_provider_between_chunks():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor._SUMMARY_INPUT_MAX_CHARS = 8000
    messages = [{"role": "tool", "tool_call_id": "t", "content": "x" * 20_000}]
    with pin_summary_route({"provider": "test", "model": "test/pinned"}):
        with patch("agent.context_compressor.call_llm", return_value=_response()) as call:
            assert compressor._generate_summary(messages)
    assert call.call_count >= 3
    assert all(k.kwargs.get("provider") == "test" and k.kwargs.get("model") == "test/pinned"
               for k in call.call_args_list)
    assert max(len(k.kwargs["messages"][0]["content"]) for k in call.call_args_list) <= 8_000


def test_cancellation_after_a_chunk_rolls_back_intermediate_summary():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor._SUMMARY_INPUT_MAX_CHARS = 8000
    compressor._previous_summary = "earlier checkpoint"
    messages = [{"role": "tool", "tool_call_id": "t", "content": "z" * 20_000}]
    with patch("agent.context_compressor.call_llm", side_effect=[_response(), AuxiliaryExplicitCancellation()]) as call:
        with pytest.raises(AuxiliaryExplicitCancellation):
            compressor._generate_summary(messages)
    assert call.call_count == 2
    assert compressor._previous_summary == "earlier checkpoint"


def test_strict_redaction_precedes_chunking_and_every_prompt_is_bounded(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor._SUMMARY_INPUT_MAX_CHARS = 8000
    synthetic_secret = "sk-proj-" + "a" * 40
    body = "first " + "x" * 9_000 + " " + synthetic_secret + " " + "y" * 9_000 + " last"
    seen = []

    def answer(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        assert synthetic_secret not in prompt
        assert len(prompt) <= 8_000
        seen.append(prompt)
        return _response()

    with patch("agent.context_compressor.call_llm", side_effect=answer):
        assert compressor._generate_summary([{"role": "tool", "tool_call_id": "t", "content": body}])
    assert len(seen) > 1
    assert any("[REDACTED]" in prompt for prompt in seen)


def test_auxiliary_retry_replays_only_failed_span_not_whole_history():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    compressor._SUMMARY_INPUT_MAX_CHARS = 8000
    compressor.summary_model = "test/auxiliary"
    messages = [{"role": "tool", "tool_call_id": "t", "content": "z" * 20_000}]
    prompts = []

    def answer(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        if len(prompts) == 2:
            raise RuntimeError("synthetic auxiliary failure")
        return _response()

    with patch("agent.context_compressor.call_llm", side_effect=answer) as call:
        assert compressor._generate_summary(messages) is not None
    assert call.call_count > 2
    assert prompts[1] == prompts[2]
    assert prompts[0] != prompts[1]
    assert all(len(prompt) <= 8_000 for prompt in prompts)
    assert compressor._previous_summary is not None


def test_major_compression_over_chunk_limit_keeps_original_messages(monkeypatch):
    monkeypatch.setattr(
        ContextCompressor, "_is_synthetic_compression_user_turn",
        classmethod(lambda cls, message: False),
    )
    compressor = ContextCompressor(
        model="test/model", config_context_length=100_000,
        protect_first_n=1, protect_last_n=2, quiet_mode=True,
    )
    compressor._SUMMARY_INPUT_MAX_CHARS = 8000
    compressor._MAX_COMPLETE_SUMMARY_CHUNKS = 1
    compressor.tail_token_budget = 50
    messages = [{"role": "system", "content": "Synthetic S0"}]
    for i in range(12):
        messages.extend([
            {"role": "user", "content": f"Read item {i}"},
            {"role": "assistant", "content": f"Item {i}: " + ("v" * 1_000)},
        ])
    with patch("agent.context_compressor.call_llm", return_value=_response()) as call:
        result = compressor.compress(messages, current_tokens=90_000)
    assert call.call_count == 1
    assert result == messages
    assert compressor._last_summary_truncated_failure is True
