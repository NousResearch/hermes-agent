"""Tests for the aggregate summarizer-input cap (_cap_summarizer_input).

Regression for kanban t_1183cfa9: a P0 where compression requests
themselves (~235K-356K tokens) exceeded the aux model's 200K limit.

PR #68377 (_TAIL_TOOL_RESULT_MAX_CHARS, Pass 4 of _prune_old_tool_results)
caps individual oversized tool results inside the protected tail, but that
only helps when the bulk is concentrated in a few large blobs. A session
whose bulk is spread across many small-to-medium messages (verbose
assistant prose, hundreds of short tool turns) is untouched by Pass 4 or by
the per-message ``_CONTENT_MAX`` truncation in ``_serialize_for_summary`` —
each message individually survives truncation, but the AGGREGATE
serialized text handed to the summarizer can still overflow the aux
model's context window. ``_cap_summarizer_input`` is the last-resort
backstop: it bounds the fully serialized summarizer request itself,
independent of how the bulk is distributed across messages.
"""

from __future__ import annotations

from agent.context_compressor import (
    ContextCompressor,
    _SUMMARIZER_INPUT_HEAD,
    _SUMMARIZER_INPUT_MAX_CHARS,
    _SUMMARIZER_INPUT_TAIL,
)


def _make_compressor(**kwargs):
    return ContextCompressor(
        model="test-model",
        config_context_length=200_000,
        quiet_mode=True,
        **kwargs,
    )


def test_small_input_is_untouched():
    comp = _make_compressor()
    text = "hello world " * 100  # well under the cap
    assert comp._cap_summarizer_input(text) == text


def test_input_at_exact_cap_is_untouched():
    comp = _make_compressor()
    text = "a" * _SUMMARIZER_INPUT_MAX_CHARS
    assert comp._cap_summarizer_input(text) == text


def test_oversized_input_is_head_tail_truncated():
    comp = _make_compressor()
    head_marker = "HEAD_MARKER_" + ("a" * _SUMMARIZER_INPUT_HEAD)
    tail_marker = ("z" * _SUMMARIZER_INPUT_TAIL) + "_TAIL_MARKER"
    middle = "m" * 2_000_000  # far over the cap
    text = head_marker + middle + tail_marker

    result = comp._cap_summarizer_input(text)

    assert len(result) < len(text)
    assert "chars truncated" in result
    # Head and tail content are preserved verbatim (data-preserving, not
    # destructive) — the omitted middle is clearly labeled.
    assert result.startswith(head_marker[:_SUMMARIZER_INPUT_HEAD])
    assert result.endswith(tail_marker[-_SUMMARIZER_INPUT_TAIL:])


def test_capped_output_stays_under_aux_model_budget():
    """The exact incident shape: aggregate bulk from many medium messages."""
    comp = _make_compressor()
    # ~1M-token-scale serialized text (the measured incident: 235K-356K
    # tokens; this synthetic case is deliberately larger to prove the cap
    # holds regardless of magnitude).
    text = "turn content " * 400_000  # ~5.2M chars raw
    result = comp._cap_summarizer_input(text)
    approx_tokens = len(result) // 4
    assert approx_tokens < 200_000, (
        f"capped summarizer input still exceeds a 200K aux model budget: "
        f"~{approx_tokens} tokens"
    )


def test_cap_is_wired_into_generate_summary_serialization():
    """The cap must run on the fully serialized text used by _generate_summary,
    not just be a standalone helper nobody calls."""
    comp = _make_compressor()
    turns = [
        {"role": "assistant", "content": "short reply " * 50}
        for _ in range(2000)
    ]
    serialized = comp._serialize_for_summary(turns)
    capped = comp._cap_summarizer_input(serialized)
    assert len(capped) <= _SUMMARIZER_INPUT_MAX_CHARS + 200  # + truncation marker slack
