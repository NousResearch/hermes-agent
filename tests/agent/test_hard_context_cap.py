"""Tests for the hard context cap backstop added to ContextCompressor.

Root-cause fix for the compression death-spiral: when the summarizer is
permanently unavailable (e.g. auxiliary compression model 404s on low
credits), apply_hard_context_cap() deterministically drops the oldest
non-protected turns instead of letting context grow unbounded.
"""
import pytest
from unittest.mock import patch
from agent.context_compressor import ContextCompressor


@pytest.fixture()
def cap_compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.5,
            protect_first_n=1,
            protect_last_n=2,
            quiet_mode=True,
            hard_context_cap_tokens=100,
        )
    return c


def _build(n):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u0"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": "a", "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "content": f"r{i}", "tool_call_id": f"c{i}"})
    msgs.append({"role": "user", "content": "tail1"})
    msgs.append({"role": "assistant", "content": "tail2"})
    return msgs


def _assert_no_orphans(messages):
    ids = set()
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if isinstance(tc, dict):
                ids.add(tc["id"])
    kept_results = set(m["tool_call_id"] for m in messages if m.get("role") == "tool")
    for i in ids:
        assert i in kept_results, f"orphaned tool_call {i}"


def test_disabled_noop(cap_compressor):
    c = ContextCompressor(model="x", threshold_percent=0.5, hard_context_cap_tokens=0, quiet_mode=True)
    m = _build(5)
    out, trim = c.apply_hard_context_cap(m, 10 ** 9)
    assert not trim and out is m


def test_under_cap_noop(cap_compressor):
    m = _build(5)
    out, trim = cap_compressor.apply_hard_context_cap(m, 50)
    assert not trim


def test_trims_oldest_and_respects_protected(cap_compressor):
    m = _build(5)
    out, trim = cap_compressor.apply_hard_context_cap(m, 1000)
    assert trim
    assert out[0]["content"] == "sys"            # system kept
    assert out[1]["content"] == "u0"             # first protected kept
    assert out[-1]["content"] == "tail2"         # last protected kept
    assert out[-2]["content"] == "tail1"         # last protected kept
    _assert_no_orphans(out)


def test_trim_never_orphans_with_small_tail(cap_compressor):
    c = ContextCompressor(model="x", threshold_percent=0.5, protect_first_n=0,
                          protect_last_n=1, hard_context_cap_tokens=50, quiet_mode=True)
    m = _build(3)
    out, trim = c.apply_hard_context_cap(m, 500)
    _assert_no_orphans(out)
