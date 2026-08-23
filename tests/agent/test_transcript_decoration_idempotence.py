"""Idempotence law for transcript decoration passes: f(f(x)) == f(x).

Pins bug #90971 (fixed in PR #91350): ``apply_anthropic_cache_control`` was
non-idempotent on pre-decorated input — the negative-slice / tool-budget
interplay stacked a fifth ``cache_control`` marker onto messages a prior
pass had already decorated, reproducing Anthropic's ``cache_control can only
be specified up to 4 times`` HTTP 400 mid-conversation.

Behavior contracts pinned here (complementing, not duplicating,
tests/agent/test_prompt_caching.py::TestApplyIdempotency which covers exact
layout convergence and caller non-mutation):

1. IDEMPOTENCE — for every realistic transcript shape x, the serialized form
   of f(f(x)) is byte-identical to f(x).
2. BUDGET — the total decoration count after any number of passes never
   exceeds Anthropic's 4-breakpoint budget.

Shapes include the #90971 regression input (already-decorated transcripts)
plus dict-content and list-content messages, tool blocks, and mixed forms.
"""

import copy
import json

import pytest

from agent.prompt_caching import (
    _count_cache_markers,
    apply_anthropic_cache_control,
)

MARKER = {"type": "ephemeral"}
BUDGET = 4  # Anthropic's hard cap on cache_control breakpoints


def _canon(messages):
    """Deterministic byte serialization for the byte-identity assertion."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()


def _tool_block(i, decorated=False):
    # Two text parts: single-part [{"type","text"}] lists are canonicalized
    # to plain strings by the strip pass (by-design flattening of the shape
    # decoration itself produces), so multi-part is the realistic list form
    # that must round-trip byte-exactly.
    part = {"type": "text", "text": f"tool result {i}"}
    if decorated:
        part["cache_control"] = dict(MARKER)
    return {
        "role": "tool",
        "tool_call_id": f"call_{i}",
        "content": [{"type": "text", "text": f"tool header {i}"}, part],
    }


def _clean_transcript():
    """system + alternating user/assistant, plain string (dict) content."""
    msgs = [{"role": "system", "content": "STATIC_PREFIX system rules and tools"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"user turn {i}"})
        msgs.append({"role": "assistant", "content": f"assistant turn {i}"})
    return msgs


def _list_content_transcript():
    """List-content parts on user turns + tool blocks interleaved."""
    msgs = [{"role": "system", "content": "STATIC_PREFIX system rules"}]
    for i in range(4):
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"question {i}"},
                {"type": "text", "text": "attachment context"},
            ],
        })
        msgs.append({
            "role": "assistant",
            "content": f"calling tool {i}",
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append(_tool_block(i))
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def _predecorated_transcript():
    """The #90971 regression shape: input a PRIOR pass already decorated.

    Markers live both top-level and on content parts (both shapes the
    decorator itself produces), including on a tool block near the tail
    where the negative-slice budget math went wrong.
    """
    msgs = [{"role": "system", "content": [
        {"type": "text", "text": "STATIC_PREFIX early", "cache_control": dict(MARKER)},
        {"type": "text", "text": "late suffix", "cache_control": dict(MARKER)},
    ]}]
    for i in range(5):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    # tail: pre-decorated user (top-level marker) + decorated tool block
    msgs.append({"role": "user", "content": "recent question",
                 "cache_control": dict(MARKER)})
    msgs.append({"role": "assistant", "content": "using tool",
                 "tool_calls": [{"id": "call_z", "type": "function",
                                 "function": {"name": "t", "arguments": "{}"}}]})
    msgs.append(_tool_block(99, decorated=True))
    msgs.append({"role": "assistant", "content": "final answer"})
    return msgs


def _mixed_partially_decorated():
    """Stale markers scattered mid-transcript (replayed history)."""
    msgs = _clean_transcript()
    msgs[3] = dict(msgs[3], cache_control=dict(MARKER))
    msgs[7] = {"role": "user", "content": [
        {"type": "text", "text": "mid turn", "cache_control": dict(MARKER)},
    ]}
    return msgs


SHAPES = [
    pytest.param(_clean_transcript, id="clean-dict-content"),
    pytest.param(_list_content_transcript, id="list-content-with-tools"),
    pytest.param(_predecorated_transcript, id="predecorated-90971-regression"),
    pytest.param(_mixed_partially_decorated, id="mixed-stale-markers"),
]

VARIANTS = [
    pytest.param({}, id="default"),
    pytest.param({"static_system_prefix": "STATIC_PREFIX"}, id="split-system"),
    pytest.param({"native_anthropic": True}, id="native-anthropic"),
    pytest.param({"cache_ttl": "1h",
                  "static_system_prefix": "STATIC_PREFIX"}, id="1h-ttl"),
]


class TestDecorationIdempotenceLaw:
    """f(f(x)) == f(x), byte-identical, for all transcript shapes (#90971)."""

    @pytest.mark.parametrize("kwargs", VARIANTS)
    @pytest.mark.parametrize("make", SHAPES)
    def test_double_apply_is_byte_identical(self, make, kwargs):
        once = apply_anthropic_cache_control(make(), **kwargs)
        twice = apply_anthropic_cache_control(
            copy.deepcopy(once), **kwargs)
        assert _canon(twice) == _canon(once), (
            "decoration pass is not idempotent: f(f(x)) != f(x)"
        )

    @pytest.mark.parametrize("kwargs", VARIANTS)
    @pytest.mark.parametrize("make", SHAPES)
    def test_budget_never_exceeded_across_passes(self, make, kwargs):
        """No pass — first, second, or fifth — may exceed Anthropic's
        4-breakpoint budget, even when the input was already decorated."""
        msgs = make()
        for round_no in range(1, 6):
            msgs = apply_anthropic_cache_control(msgs, **kwargs)
            count = _count_cache_markers(msgs, [])
            assert count <= BUDGET, (
                f"round {round_no}: {count} cache_control markers exceed the "
                f"{BUDGET}-breakpoint budget (the #90971 HTTP-400 class)"
            )

    @pytest.mark.parametrize("make", SHAPES)
    def test_fixed_point_reached_after_one_pass(self, make):
        """Convergence in one step: repeated application never drifts —
        rounds 1..4 all serialize identically (stronger than pairwise)."""
        forms = []
        msgs = make()
        for _ in range(4):
            msgs = apply_anthropic_cache_control(
                msgs, static_system_prefix="STATIC_PREFIX")
            forms.append(_canon(msgs))
            msgs = copy.deepcopy(json.loads(forms[-1]))
        assert len(set(forms)) == 1, "decoration drifted across rounds"

    def test_predecorated_input_does_not_inherit_extra_markers(self):
        """#90971 core symptom: pre-decorated input must not carry stale
        markers PLUS fresh ones. Every marker in the output is one the
        current pass placed — total exactly within budget, and no message
        carries more than one marker at both levels."""
        out = apply_anthropic_cache_control(
            _predecorated_transcript(), static_system_prefix="STATIC_PREFIX")
        assert _count_cache_markers(out, []) <= BUDGET
        for msg in out:
            top = 1 if "cache_control" in msg else 0
            content = msg.get("content")
            parts = (
                sum(1 for p in content
                    if isinstance(p, dict) and "cache_control" in p)
                if isinstance(content, list) else 0
            )
            # A single message may legitimately hold two SYSTEM breakpoints
            # (prefix + suffix in the split-system layout); non-system
            # messages must never stack top-level + part markers.
            if msg.get("role") != "system":
                assert top + parts <= 1, (
                    f"stacked markers on one message: {msg!r}"
                )
