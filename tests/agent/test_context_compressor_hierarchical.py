"""Tests for hierarchical context compression behavior."""

import logging
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


def _response(text: str):
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = text
    return mock


def _messages(count: int, chars_per_message: int = 1200):
    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"message {i} " + ("x" * chars_per_message)})
    return messages


class TestHierarchicalSummary:
    def test_large_summary_input_is_split_into_map_reduce_calls(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=200000):
            compressor = ContextCompressor(
                model="test/model",
                quiet_mode=False,
                hierarchical_chunk_tokens=900,
                hierarchical_min_serialized_chars=0,
                hierarchical_min_chunks=2,
            )

        messages = _messages(12, chars_per_message=1800)
        responses = [_response(f"segment summary {i}") for i in range(1, 20)]

        with patch("agent.context_compressor.call_llm", side_effect=responses) as mock_call:
            summary = compressor._generate_summary(messages)

        assert summary is not None
        assert summary.startswith(SUMMARY_PREFIX)
        assert "segment summary" in summary
        assert mock_call.call_count >= 3  # at least two map calls plus one reduce call

        map_prompts = [
            call.kwargs["messages"][0]["content"]
            for call in mock_call.call_args_list[:-1]
        ]
        reduce_prompt = mock_call.call_args_list[-1].kwargs["messages"][0]["content"]

        assert all("SEGMENT" in prompt for prompt in map_prompts)
        assert "SEGMENT SUMMARIES TO MERGE" in reduce_prompt
        assert "segment summary 1" in reduce_prompt
        assert "segment summary 2" in reduce_prompt

    def test_failed_large_segment_is_split_instead_of_whole_summary_fallback(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=200000):
            compressor = ContextCompressor(
                model="test/model",
                quiet_mode=True,
                hierarchical_chunk_tokens=600,
                hierarchical_min_serialized_chars=0,
                hierarchical_min_chunks=2,
            )

        messages = _messages(8, chars_per_message=1600)
        call_index = {"n": 0}

        def respond_or_fail(**_kwargs):
            call_index["n"] += 1
            if call_index["n"] == 1:
                raise TimeoutError("segment too large")
            if "SEGMENT SUMMARIES TO MERGE" in _kwargs["messages"][0]["content"]:
                return _response("final merged summary")
            return _response(f"split segment {call_index['n']}")

        with patch("agent.context_compressor.call_llm", side_effect=respond_or_fail) as mock_call:
            summary = compressor._generate_summary(messages)

        assert summary is not None
        assert summary.startswith(SUMMARY_PREFIX)
        assert "final merged summary" in summary
        assert mock_call.call_count >= 5
        assert compressor._last_summary_fallback_used is False

    def test_hierarchical_summary_logs_progress(self, caplog):
        with patch("agent.context_compressor.get_model_context_length", return_value=200000):
            compressor = ContextCompressor(
                model="test/model",
                quiet_mode=False,
                hierarchical_chunk_tokens=700,
                hierarchical_min_serialized_chars=0,
                hierarchical_min_chunks=2,
            )

        messages = _messages(8, chars_per_message=1600)
        responses = [_response(f"summary {i}") for i in range(1, 20)]

        caplog.set_level(logging.INFO, logger="agent.context_compressor")
        with patch("agent.context_compressor.call_llm", side_effect=responses):
            summary = compressor._generate_summary(messages)

        assert summary is not None
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Hierarchical compression: splitting" in log_text
        assert "Hierarchical compression: summarizing segment" in log_text
        assert "Hierarchical compression: merging" in log_text
