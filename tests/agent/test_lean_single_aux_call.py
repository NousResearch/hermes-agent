"""Lean compaction uses one auxiliary call when its complete redacted source fits.

Contract: one call for a covered small input; for oversized input, at most eight
chronological calls during the same major compression, or an explicit failure.
The anchor index and recovery footer remain part of accepted summaries.
"""

from unittest.mock import patch, MagicMock

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _LEAN_ANCHOR_HEADING,
    _LEAN_RECOVERY_HEADING,
    _LEAN_SESSION_LOG_HEADING,
)


def _mk_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        threshold_percent=0.85,
        protect_first_n=2,
        protect_last_n=2,
        quiet_mode=True,
        tail_mode="lean",
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        c = ContextCompressor(**kwargs)
        _ = c.context_length
    c._session_id = "sess-lean-1call"
    return c


def _llm_response(text):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = text
    return resp


def _big_region(n_rounds=60, tool_chars=6_000):
    """Synthetic compacted region: user + assistant/tool rounds, ~360K chars."""
    turns = [
        {"role": "user", "content": "Please fix PR #12345 in agent/foo.py"},
    ]
    for i in range(n_rounds):
        turns.append({
            "role": "assistant",
            "content": f"Working on step {i}: editing agent/foo.py line {i}",
        })
        turns.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "tool_name": "terminal",
            "content": f"round {i} output: " + ("x" * tool_chars),
        })
    return turns


SUMMARY_BODY = (
    "## Historical Task Snapshot\nUser asked: 'Please fix PR #12345 in agent/foo.py'\n\n"
    "## Goal\nFix the bug.\n\n"
    "## Completed Actions\n1. EDIT agent/foo.py — done [tool: patch]\n\n"
    f"{_LEAN_SESSION_LOG_HEADING}\n- Edited agent/foo.py; PR #12345; ran pytest.\n"
)


class TestLeanSingleAuxiliaryCall:
    def test_exactly_one_call_llm_per_lean_attempt(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as main_call, patch(
            "agent.auxiliary_client.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as aux_call:
            summary = c._generate_summary(turns)
        assert summary is not None
        # A complete small input uses the one-call fast path.
        assert main_call.call_count + aux_call.call_count == 1

    def test_session_log_heading_lands_in_summary(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ):
            summary = c._generate_summary(turns)
        assert _LEAN_SESSION_LOG_HEADING in summary


    def test_anchor_index_and_recovery_footer_present(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ):
            summary = c._generate_summary(turns)
        assert _LEAN_ANCHOR_HEADING in summary
        assert _LEAN_RECOVERY_HEADING in summary
        assert "session_search" in summary

    def test_oversized_region_uses_bounded_complete_requests(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=120, tool_chars=6_000)
        source = c._serialize_for_summary(turns)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as mock_call:
            assert c._generate_summary(turns) is not None
        assert 1 < mock_call.call_count <= c._MAX_COMPLETE_SUMMARY_CHUNKS
        prompts = [item.kwargs["messages"][0]["content"] for item in mock_call.call_args_list]
        assert all(len(prompt) <= c._SUMMARY_INPUT_MAX_CHARS for prompt in prompts)
        assert source[:100] in prompts[0]
        assert source[-100:] in prompts[-1]
        assert not any("chars elided" in prompt for prompt in prompts)


class TestSampledSummaryInput:
    def test_small_input_passes_through(self):
        records = ["[USER]: abc" * 100] * 3
        assert ContextCompressor._sample_summary_records(records)[0] == "\n\n".join(records)

    def test_sampling_is_bounded_ordered_and_marked(self):
        # Distinct decade markers let us verify oldest-to-newest order and
        # uniform coverage across the whole region.
        records = [f"<seg{i:02d}>" + ("x" * 50_000) for i in range(10)]
        out, _coverage = ContextCompressor._sample_summary_records(records)
        assert len(out) <= ContextCompressor._SUMMARY_INPUT_MAX_CHARS
        assert "chars elided" in out
        seen = [i for i in range(10) if f"<seg{i:02d}>" in out]
        # Coverage reaches past the head AND includes the newest end.
        assert seen == sorted(seen)
        assert any(i >= 5 for i in seen)
        # The newest record anchors the last slice, so its token must be present outright.
        assert "<seg09>" in out

    def test_legacy_mode_also_preserves_full_source_with_bounded_calls(self):
        c = _mk_compressor(tail_mode="legacy")
        turns = _big_region(n_rounds=120, tool_chars=6_000)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response("## Historical Task Snapshot\nNone."),
        ) as mock_call:
            assert c._generate_summary(turns) is not None
        assert 1 < mock_call.call_count <= c._MAX_COMPLETE_SUMMARY_CHUNKS
        assert all(len(item.kwargs["messages"][0]["content"]) <= c._SUMMARY_INPUT_MAX_CHARS
                   for item in mock_call.call_args_list)
        assert not any("summary input truncated" in item.kwargs["messages"][0]["content"]
                       for item in mock_call.call_args_list)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
