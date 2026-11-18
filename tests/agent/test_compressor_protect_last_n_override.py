"""``protect_last_n_override`` replaces the configured tail window (#144423).

The ``/compress here N`` boundary already decides what stays verbatim; the
split-off head must not re-protect the compressor's own ``protect_last_n``
messages a few rows before that boundary, or a large tool result there survives
verbatim and the attempt is refused as ``would_grow``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=6,
            quiet_mode=True,
        )
    instance.tail_token_budget = 10
    return instance


def _messages(turns: int) -> list[dict]:
    rows: list[dict] = []
    for index in range(turns):
        rows.append({"role": "user", "content": f"question {index} " + "x" * 40})
        rows.append({"role": "assistant", "content": f"answer {index} " + "y" * 40})
    return rows


def test_default_window_keeps_the_configured_tail(compressor: ContextCompressor) -> None:
    messages = _messages(6)
    head_end = compressor._align_boundary_forward(messages, compressor._protect_head_size(messages))
    cut = compressor._find_tail_cut_by_tokens(messages, head_end)
    # protect_last_n=6 is the floor: at least six rows stay in the tail.
    assert len(messages) - cut >= 6


def test_override_zero_lets_the_boundary_own_the_tail(compressor: ContextCompressor) -> None:
    messages = _messages(6)
    head_end = compressor._align_boundary_forward(messages, compressor._protect_head_size(messages))
    cut = compressor._find_tail_cut_by_tokens(messages, head_end, protect_last_n=0)
    # No protected tail rows beyond the structural minimum (newest row, anchors):
    # the caller-chosen boundary decides, not the configured 6.
    assert len(messages) - cut < 6


def test_plan_compaction_window_threads_the_override(compressor: ContextCompressor) -> None:
    messages = _messages(6)
    _, _, _, default_end = compressor._plan_compaction_window(messages)
    _, _, _, override_end = compressor._plan_compaction_window(messages, protect_last_n=0)
    # A later cut compresses more of the middle: with the override the boundary owns
    # the tail, so the default 6-row protection no longer hides part of the head.
    assert override_end > default_end
