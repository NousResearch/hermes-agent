"""A configured trigger far below the window must leave compaction room under that trigger.

``compression.threshold_tokens: 12000`` on a 64K window: the lean 10K tail floor used to keep ~10K of verbatim
tail, so head + summary + tail stayed over the 12K trigger and every follow-up pass reclaimed nothing.
"""

from unittest.mock import patch

import agent.context_compressor as cc
from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens


def _make(ctx: int, threshold_tokens: int | None) -> ContextCompressor:
    with patch.object(cc, "get_model_context_length", return_value=ctx):
        comp = ContextCompressor(
            model="test/model", threshold_percent=0.50, protect_first_n=1, protect_last_n=2,
            quiet_mode=True, threshold_tokens_cap=threshold_tokens,
        )
        _ = comp.context_length  # resolve while the window mock is active
        return comp


def _chatty_transcript(turns: int) -> list[dict]:
    """Short user prompts, long assistant replies: the shape where a few rows fill the whole tail."""
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"HISTORY_{i} " + "preserve these facts " * 100})
        msgs.append({"role": "assistant", "content": f"ACK_{i} " + "historical detail " * 700})
    msgs.append({"role": "user", "content": "next " + "preserve these facts " * 100})
    return msgs


def test_retained_tail_leaves_half_the_configured_trigger_free():
    comp = _make(64_000, threshold_tokens=12_000)
    assert comp.threshold_tokens == 12_000
    msgs = _chatty_transcript(4)
    start, end = comp._compress_window(msgs)
    tail_tokens = sum(_estimate_msg_budget_tokens(m) for m in msgs[end:])
    assert end > start
    assert tail_tokens <= comp.threshold_tokens // 2 + _estimate_msg_budget_tokens(msgs[-2]), (start, end, tail_tokens)


def test_unconfigured_trigger_keeps_the_lean_floor():
    assert _make(131_072, threshold_tokens=None).tail_token_budget == cc.LEAN_TAIL_FLOOR_TOKENS
