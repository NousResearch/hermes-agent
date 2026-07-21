"""Tests for the Pass-4 tail tool-result cap in _prune_old_tool_results.

Regression for the 26-07-20 incident: a 246-message session whose bulk sat
in RECENT oversized tool results (60KB job lists, 52KB build logs, 50KB
delegation transcripts) compacted to the same message count / token size
because tail protection exempted those blobs wholesale, ending in
"Cannot compress further".
"""

from __future__ import annotations

from agent.context_compressor import (
    ContextCompressor,
    _TAIL_TOOL_RESULT_MAX_CHARS,
)


def _make_compressor(**kwargs):
    return ContextCompressor(
        model="test-model",
        config_context_length=200_000,
        quiet_mode=True,
        **kwargs,
    )


def _tool_pair(idx: int, content: str):
    call_id = f"call_{idx}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def test_oversized_tail_tool_result_is_truncated():
    comp = _make_compressor()
    big = "x" * (_TAIL_TOOL_RESULT_MAX_CHARS * 4)  # 64K chars, well over cap
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": "do the thing"}]
    # Oversized tool result that lands INSIDE the protected tail but outside
    # the last-3 exemption window.
    messages += _tool_pair(1, big)
    # Padding turns so the big result is not within the last 3 messages.
    messages += [
        {"role": "assistant", "content": "working on it"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    pruned_msgs, pruned_count = comp._prune_old_tool_results(
        messages, protect_tail_count=20, protect_tail_tokens=comp.tail_token_budget,
    )
    tool_msgs = [m for m in pruned_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert len(tool_msgs[0]["content"]) < len(big)
    assert "truncated during context compression" in tool_msgs[0]["content"]
    assert pruned_count >= 1


def test_recent_tool_result_in_exempt_window_is_kept():
    comp = _make_compressor()
    big = "y" * (_TAIL_TOOL_RESULT_MAX_CHARS * 2)
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": "go"}]
    messages += [{"role": "assistant", "content": "padding"}]
    # The big tool result is one of the LAST 3 messages — current work,
    # must survive untouched.
    messages += _tool_pair(1, big)
    pruned_msgs, _ = comp._prune_old_tool_results(
        messages, protect_tail_count=20, protect_tail_tokens=comp.tail_token_budget,
    )
    tool_msgs = [m for m in pruned_msgs if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == big


def test_small_tail_tool_results_untouched():
    comp = _make_compressor()
    small = "z" * 500
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": "go"}]
    messages += _tool_pair(1, small)
    messages += [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
    ]
    pruned_msgs, _ = comp._prune_old_tool_results(
        messages, protect_tail_count=20, protect_tail_tokens=comp.tail_token_budget,
    )
    tool_msgs = [m for m in pruned_msgs if m.get("role") == "tool"]
    assert tool_msgs[0]["content"] == small


def test_many_oversized_tail_results_all_capped():
    """The incident shape: many recent oversized tool results."""
    comp = _make_compressor()
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": "go"}]
    for i in range(6):
        # Distinct contents so dedup (Pass 1) doesn't collapse them.
        messages += _tool_pair(i, f"blob{i}-" + ("w" * 50_000))
    messages += [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
    ]
    pre = sum(len(str(m.get("content", ""))) for m in messages)
    pruned_msgs, pruned_count = comp._prune_old_tool_results(
        messages, protect_tail_count=20, protect_tail_tokens=comp.tail_token_budget,
    )
    post = sum(len(str(m.get("content", ""))) for m in pruned_msgs)
    assert post < pre * 0.5, f"expected >50% shrink, got {pre} -> {post}"
    for m in pruned_msgs:
        if m.get("role") == "tool":
            assert len(m["content"]) <= _TAIL_TOOL_RESULT_MAX_CHARS + 200
