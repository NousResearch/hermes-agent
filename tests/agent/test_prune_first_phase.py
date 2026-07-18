"""Tests for the prune-first phase (issue #513).

The prune-first phase runs the cheap, LLM-free tool-output elision on an
ABSOLUTE token budget, decoupled from the window-relative summarization
threshold. On large-context models the summary threshold
(context_length * threshold_percent) can sit at hundreds of thousands of
tokens, so re-sent tool output never trips it and the in-compress() prune
stays dormant. These tests assert the behavior CONTRACT:

  1. Disarmed by default: prune_protect_tokens=None => never fires, and
     should_prune_tools()/prune_tools_only() are inert (historical behavior).
  2. Trigger is absolute and independent of should_compress().
  3. When it fires, only old tool RESULTS are elided; tool CALLS, user, and
     assistant text survive verbatim (conversation structure preserved).
  4. prune_minimum_tokens gates trivial reclamations.
"""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _make(context_length, **kwargs):
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=context_length,
    ):
        return ContextCompressor(model="test/model", quiet_mode=True, **kwargs)


def _big_tool_result(call_id, n_chars):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "x" * n_chars,
    }


def _assistant_call(call_id, name="terminal"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _long_session(n_pairs, tool_chars=8000):
    """Build a realistic coding session: user opener, then N (assistant
    tool-call -> tool-result) pairs, plus interleaved assistant text."""
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Please refactor the auth module"},
    ]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid))
        msgs.append(_big_tool_result(cid, tool_chars))
        msgs.append({"role": "assistant", "content": f"step {i} reasoning"})
    msgs.append({"role": "user", "content": "now run the tests"})
    return msgs


class TestDisarmedByDefault:
    def test_prune_protect_none_disables_trigger(self):
        c = _make(1_000_000)
        assert c.prune_protect_tokens is None
        # Even at an enormous token count, disarmed => never fires.
        assert c.should_prune_tools(5_000_000) is False

    def test_prune_only_is_inert_when_disarmed(self):
        c = _make(1_000_000)
        msgs = _long_session(40)
        out, pruned = c.prune_tools_only(msgs)
        assert pruned == 0
        assert out is msgs  # untouched, same object

    def test_default_matches_historical_constructor(self):
        # A compressor built the old way (no prune kwargs) must behave
        # identically: disarmed.
        c = _make(200_000)
        assert c.prune_protect_tokens is None
        assert c.should_prune_tools(199_000) is False


class TestAbsoluteTriggerIndependentOfCompress:
    def test_fires_below_summary_threshold_on_large_window(self):
        # 1M window => summary threshold ~800K. Arm prune at 40K protect.
        c = _make(1_000_000, prune_protect_tokens=40_000)
        # A token count that is FAR below the summary threshold but above the
        # prune trigger (protect + minimum).
        tokens = 40_000 + c.prune_minimum_tokens + 1
        # The window-relative summary trigger must NOT fire here...
        assert c.should_compress(tokens) is False
        # ...but the absolute prune trigger MUST.
        assert c.should_prune_tools(tokens) is True

    def test_does_not_fire_below_protect_plus_minimum(self):
        c = _make(1_000_000, prune_protect_tokens=40_000, prune_minimum_tokens=20_000)
        assert c.should_prune_tools(40_000 + 20_000 - 1) is False
        assert c.should_prune_tools(40_000 + 20_000) is True

    def test_custom_minimum_respected(self):
        c = _make(1_000_000, prune_protect_tokens=10_000, prune_minimum_tokens=5_000)
        assert c.prune_minimum_tokens == 5_000
        assert c.should_prune_tools(14_999) is False
        assert c.should_prune_tools(15_000) is True


class TestPrunePreservesConversationStructure:
    def test_only_tool_results_are_elided(self):
        c = _make(1_000_000, prune_protect_tokens=8_000)
        msgs = _long_session(40, tool_chars=8000)

        # capture the user/assistant text before pruning
        user_texts = [m["content"] for m in msgs if m["role"] == "user"]
        asst_texts = [
            m["content"] for m in msgs
            if m["role"] == "assistant" and not m.get("tool_calls")
        ]
        tool_call_ids = [
            tc["id"]
            for m in msgs if m["role"] == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]

        out, pruned = c.prune_tools_only(msgs)
        assert pruned > 0, "expected old tool results to be pruned"

        # user messages verbatim
        assert [m["content"] for m in out if m["role"] == "user"] == user_texts
        # assistant text verbatim
        assert [
            m["content"] for m in out
            if m["role"] == "assistant" and not m.get("tool_calls")
        ] == asst_texts
        # every tool CALL id still present (tool calls not dropped)
        out_call_ids = [
            tc["id"]
            for m in out if m["role"] == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]
        assert out_call_ids == tool_call_ids

    def test_recent_tool_results_survive_verbatim(self):
        # Generous protect budget keeps the tail's tool output intact.
        c = _make(1_000_000, prune_protect_tokens=8_000)
        msgs = _long_session(40, tool_chars=8000)
        out, pruned = c.prune_tools_only(msgs)

        # The last tool result (most recent, within protect window) must keep
        # its full original content.
        last_tool_in = next(
            m for m in reversed(msgs) if m["role"] == "tool"
        )
        last_tool_out = next(
            m for m in reversed(out) if m["role"] == "tool"
        )
        assert last_tool_out["content"] == last_tool_in["content"]

    def test_pruning_reduces_total_size(self):
        c = _make(1_000_000, prune_protect_tokens=8_000)
        msgs = _long_session(40, tool_chars=8000)

        def total_chars(ms):
            return sum(len(str(m.get("content") or "")) for m in ms)

        before = total_chars(msgs)
        out, pruned = c.prune_tools_only(msgs)
        after = total_chars(out)
        assert pruned > 0
        assert after < before, "pruning must reduce total content size"

    def test_message_count_unchanged(self):
        # Prune elides CONTENT, it does not drop rows — so tool_call/result
        # pairing stays valid for the provider API.
        c = _make(1_000_000, prune_protect_tokens=8_000)
        msgs = _long_session(40)
        out, pruned = c.prune_tools_only(msgs)
        assert len(out) == len(msgs)


class TestConfigCoercion:
    def test_zero_and_negative_protect_disarm(self):
        assert _make(1_000_000, prune_protect_tokens=0).prune_protect_tokens is None
        assert _make(1_000_000, prune_protect_tokens=-5).prune_protect_tokens is None

    def test_minimum_defaults_when_unset(self):
        c = _make(1_000_000, prune_protect_tokens=40_000)
        assert c.prune_minimum_tokens == ContextCompressor._DEFAULT_PRUNE_MINIMUM_TOKENS


class TestResultOnlyContract:
    """prune_tools_only must never mutate tool-call arguments."""

    def test_tool_call_arguments_preserved_verbatim(self):
        c = _make(1_000_000, prune_protect_tokens=4_000)
        # Build a session with large tool-call arguments that Pass 3
        # would truncate if result_only were not in effect.
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "write a large file"},
        ]
        big_args = '{"content": "' + "A" * 2000 + '"}'
        for i in range(20):
            cid = f"call_{i}"
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": cid,
                    "type": "function",
                    "function": {"name": "write_file", "arguments": big_args},
                }],
            })
            msgs.append(_big_tool_result(cid, 8000))
            msgs.append({"role": "assistant", "content": f"step {i}"})
        msgs.append({"role": "user", "content": "done"})

        out, pruned = c.prune_tools_only(msgs)
        # Tool-call arguments must be byte-for-byte identical.
        for m_in, m_out in zip(msgs, out):
            if m_in.get("role") == "assistant" and m_in.get("tool_calls"):
                for tc_in, tc_out in zip(m_in["tool_calls"], m_out["tool_calls"]):
                    assert tc_in["function"]["arguments"] == tc_out["function"]["arguments"], \
                        "prune_tools_only must not truncate tool-call arguments"


class TestMeasuredSavingsGate:
    """prune_tools_only must roll back if measured savings < prune_minimum_tokens."""

    def test_rollback_when_savings_insufficient(self):
        # Very high minimum so trivial pruning won't meet it.
        c = _make(1_000_000, prune_protect_tokens=4_000, prune_minimum_tokens=999_999)
        msgs = _long_session(10, tool_chars=500)  # small results, little to save
        out, pruned = c.prune_tools_only(msgs)
        assert pruned == 0, "should roll back when savings < minimum"
        assert out is msgs, "should return original object on rollback"

    def test_commits_when_savings_sufficient(self):
        # Low minimum with a big session — savings will exceed it.
        c = _make(1_000_000, prune_protect_tokens=4_000, prune_minimum_tokens=100)
        msgs = _long_session(40, tool_chars=8000)
        out, pruned = c.prune_tools_only(msgs)
        assert pruned > 0, "should commit when savings >= minimum"
