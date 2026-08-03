"""Progress semantics for prune-only compaction passes."""

from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.75,
            protect_first_n=1,
            protect_last_n=2,
            quiet_mode=True,
        )


def test_prune_only_no_summary_window_reports_progress():
    compressor = _compressor()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "content": "large old result"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "tail"},
    ]
    pruned = [dict(message) for message in messages]
    pruned[2]["content"] = "[tool result pruned]"

    with (
        patch.object(
            compressor,
            "_prune_old_tool_results",
            return_value=(pruned, 1),
        ),
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=1),
        patch.object(compressor, "_generate_summary") as summarize,
    ):
        result = compressor.compress(messages, current_tokens=90_000)

    assert result[2]["content"] == "[tool result pruned]"
    assert compressor.compression_made_progress() is True
    summarize.assert_not_called()


def test_identity_no_summary_window_reports_no_progress():
    compressor = _compressor()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "still working"},
        {"role": "user", "content": "status"},
        {"role": "assistant", "content": "tail"},
    ]

    with (
        patch.object(
            compressor,
            "_prune_old_tool_results",
            return_value=(messages, 0),
        ),
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=1),
        patch.object(compressor, "_generate_summary") as summarize,
    ):
        compressor.compress(messages, current_tokens=90_000)

    assert compressor.compression_made_progress() is False
    summarize.assert_not_called()
