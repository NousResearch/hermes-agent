"""Pre-summary eligibility and lossless skip regressions for #23811 and #21470."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture
def compressor():
    c = ContextCompressor(
        model="test/model",
        config_context_length=200_000,
        protect_first_n=2,
        protect_last_n=3,
        quiet_mode=True,
    )
    c.last_prompt_tokens = c.threshold_tokens + 1
    return c


@pytest.fixture
def prunable_transcript():
    # The large recent request/reply fill the protected tail; the older tool result
    # is prunable, while the middle stays below the summary target plus overhead.
    return [
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "initial reply"},
        {"role": "user", "content": "inspect result"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "read-1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"sample.txt"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "read-1", "content": "x" * 6000},
        {"role": "assistant", "content": "Result inspected"},
        {"role": "user", "content": "new request"},
        {"role": "assistant", "content": "intermediate reply"},
        {"role": "user", "content": "latest request " + "z" * 24_000},
        {"role": "assistant", "content": "latest reply " + "y" * 24_000},
    ]


def test_ineligible_window_preserves_prunable_transcript(compressor, prunable_transcript):
    """Skipping declines both pruning and summarization, leaving tool output intact."""
    original = deepcopy(prunable_transcript)
    pruned, count = compressor._prune_old_tool_results(
        prunable_transcript, protect_tail_count=compressor.protect_last_n,
        protect_tail_tokens=compressor.tail_token_budget,
    )
    assert count > 0
    assert pruned[4]["content"] != original[4]["content"]

    with patch.object(compressor, "_generate_summary", return_value="Unused summary") as summarize:
        result = compressor.compress(prunable_transcript)

    summarize.assert_not_called()
    assert result == original
    assert prunable_transcript == original
    assert compressor.compression_count == 0


@pytest.mark.parametrize("force", [False, True], ids=["eligible-window", "manual-small-window"])
def test_eligible_or_manual_compression_generates_summary(compressor, prunable_transcript, force):
    """A large window proceeds; a manual request also proceeds for a small window."""
    if not force:
        # Plain conversation text survives tool pruning and makes the middle eligible.
        prunable_transcript[2]["content"] += " historical detail" * 2000

    with patch.object(compressor, "_generate_summary", return_value="Mock summary") as summarize:
        result = compressor.compress(prunable_transcript, force=force)

    summarize.assert_called_once()
    assert result != prunable_transcript
    assert compressor.compression_count == 1
