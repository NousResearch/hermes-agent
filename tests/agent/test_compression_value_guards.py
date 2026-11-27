"""Compression value guards: ratio ceiling, minimum reclaim, truncation fallback."""
import pytest

from agent.conversation_compression import (
    MAX_SUMMARY_RATIO, MIN_RECLAIM_TOKENS,
    truncate_oldest_tool_results, _is_tool_result,
    _TRUNC_HEAD_LINES, _TRUNC_TAIL_LINES,
)
from agent.context_compressor import estimate_messages_tokens_rough


def _tool(n_lines, first="EXIT STATUS: 0", last="FINAL VERDICT: ok"):
    body = [first] + [f"noise line {i} " + ("x" * 60) for i in range(n_lines)] + [last]
    return {"role": "tool", "content": "\n".join(body)}


def _msgs(n_tools=4, lines=200):
    out = [{"role": "user", "content": "start"}]
    for _ in range(n_tools):
        out.append({"role": "assistant", "content": "calling"})
        out.append(_tool(lines))
    return out


def test_thresholds_are_the_measured_ones():
    assert MAX_SUMMARY_RATIO == 0.5
    assert MIN_RECLAIM_TOKENS == 2000


def test_truncation_actually_reclaims():
    m = _msgs()
    before = estimate_messages_tokens_rough(m)
    out = truncate_oldest_tool_results(m, before // 2)
    assert estimate_messages_tokens_rough(out) < before


def test_exit_status_and_verdict_survive():
    """A tool result truncated to nothing looks like a tool that returned nothing."""
    m = _msgs()
    out = truncate_oldest_tool_results(m, 1)
    truncated = [x for x in out if _is_tool_result(x) and "truncated" in x["content"]]
    assert truncated, "expected at least one truncated tool result"
    for t in truncated:
        assert "EXIT STATUS: 0" in t["content"], "first lines (exit status) must survive"
        assert "FINAL VERDICT: ok" in t["content"], "last lines (verdict) must survive"


def test_newest_tool_result_is_never_truncated():
    m = _msgs()
    out = truncate_oldest_tool_results(m, 1)
    last_tool = [x for x in out if _is_tool_result(x)][-1]
    assert "truncated" not in last_tool["content"]


def test_input_is_not_mutated():
    m = _msgs()
    original = m[2]["content"]
    truncate_oldest_tool_results(m, 1)
    assert m[2]["content"] == original


def test_short_tool_results_are_left_alone():
    m = [{"role": "tool", "content": "a\nb\nc"}, {"role": "tool", "content": "d\ne"}]
    out = truncate_oldest_tool_results(m, 1)
    assert out[0]["content"] == "a\nb\nc"


def test_single_tool_result_is_untouched():
    m = [{"role": "tool", "content": "\n".join(str(i) for i in range(500))}]
    assert truncate_oldest_tool_results(m, 1) == m


def test_non_tool_messages_are_untouched():
    m = _msgs()
    out = truncate_oldest_tool_results(m, 1)
    for a, b in zip(m, out):
        if not _is_tool_result(a):
            assert a == b


def test_ratio_ceiling_is_ten_times_looser_than_observed_good():
    """The measured healthy ratio was 0.05; the ceiling must not fire on it."""
    healthy_out, healthy_in = 1041, 19950
    assert healthy_out <= healthy_in * MAX_SUMMARY_RATIO
