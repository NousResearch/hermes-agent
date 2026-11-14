"""Summary input spends the aggregate budget before clipping individual messages."""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor


def _compressor(mode):
    return ContextCompressor(
        model="test-model", quiet_mode=True, tail_mode=mode,
        config_context_length=200_000,
    )


def _capture_prompt(compressor, messages):
    response = MagicMock()
    response.choices[0].message.content = "## Historical Task Snapshot\nContinue the task."
    with patch("agent.context_compressor.call_llm", return_value=response) as call:
        assert compressor._generate_summary(messages) is not None
    assert call.call_count == 1
    return call.call_args.kwargs["messages"][0]["content"]


@pytest.mark.parametrize("mode", ["lean", "legacy"])
def test_fitting_window_keeps_long_message_and_tool_arguments(mode):
    compressor = _compressor(mode)
    fact = "The selected colour is violet."
    body = "a" * 5000 + fact + "b" * (5000 - len(fact))
    arguments = '{"text": "' + "c" * 1600 + "keep this tool detail" + "d" * 1300 + '"}'
    messages = [
        {"role": "user", "content": body},
        {"role": "assistant", "content": "Writing.", "tool_calls": [
            {"function": {"name": "write_file", "arguments": arguments}},
        ]},
    ]
    prompt = _capture_prompt(compressor, messages)
    assert body in prompt
    assert arguments in prompt
    # Micro-compaction still uses the default, individually clipped serializer.
    micro_input = compressor._serialize_for_summary(messages)
    assert fact not in micro_input
    assert "keep this tool detail" not in micro_input


@pytest.mark.parametrize("mode", ["lean", "legacy"])
@pytest.mark.parametrize("oversized", [False, True])
def test_existing_bounding_output_is_preserved_and_coverage_counts_clips(mode, oversized):
    compressor = _compressor(mode)
    body = "x" * (10_000 if oversized else 1000)
    messages = [{"role": "user", "content": f"record-{i}: {body}"} for i in range(40)]
    records = compressor._serialize_records_for_summary(messages)
    if mode == "lean":
        expected, coverage = compressor._sample_summary_records(records)
        sampled_chars = coverage["sampled_chars"]
    else:
        expected = compressor._bound_summary_input("\n\n".join(records))
        sampled_chars = len(expected) - 2 * (len(records) - 1)
    telemetry = compressor._begin_compression_telemetry(current_tokens=1_000_000)
    prompt = _capture_prompt(compressor, messages)
    transcript = prompt.split("TURNS TO SUMMARIZE:\n", 1)[1].split("\n\nUse this exact structure:", 1)[0]
    assert transcript == expected
    input_chars = sum(len(f"[USER]: {msg['content']}") for msg in messages)
    assert telemetry["summary_input_chars"] == input_chars
    assert telemetry["summary_input_sampled_chars"] == sampled_chars
    assert telemetry["summary_input_omitted_chars"] == input_chars - sampled_chars
