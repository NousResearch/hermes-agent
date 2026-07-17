"""Regression guards for #59009 — reasoning delivered duplicated (2-4x) to
gateway consumers.

Two independent bugs produced duplicated reasoning:

1. **Post-completion re-fire on gateway platforms.** Gateway consumers
   register ``reasoning_callback`` while the *text*-stream callbacks
   (``stream_delta_callback`` / ``_stream_callback``) are None. The old
   guard in ``build_assistant_message`` only tested the text-stream
   callbacks, so reasoning that had already been streamed incrementally
   via ``_fire_reasoning_delta`` was re-fired again as one accumulated
   blob — every reasoning burst arrived twice.

   The fix latches ``_reasoning_streamed_this_response`` inside
   ``_fire_reasoning_delta`` and checks BOTH signals in
   ``build_assistant_message``: the latch (gateway reasoning-delta path)
   and active text-stream consumers (CLI <think>-tag extraction path,
   which displays reasoning without touching the latch — see
   tests/cli/test_reasoning_command.py TestReasoningDeltasFiredFlag).

2. **Provider delta misbehavior.** Some providers re-send the entire
   accumulated reasoning as a trailing chunk (cumulative echo) or
   re-deliver an overlapping window after an internal reconnect. Naive
   appending stored the reasoning doubled. ``normalize_reasoning_delta``
   corrects both — with a minimum-overlap gate so legitimate short
   repetitions ("the", token fragments) are never dropped.
"""

import unittest
from types import SimpleNamespace

from run_agent import AIAgent
from agent.chat_completion_helpers import (
    _MIN_REASONING_OVERLAP,
    normalize_reasoning_delta,
)


def _make_agent(**overrides):
    agent = AIAgent.__new__(AIAgent)
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.verbose_logging = False
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


class TestGatewayShapedCallbacks(unittest.TestCase):
    """Gateway platforms: reasoning_callback set, text-stream callbacks None."""

    def test_streamed_reasoning_not_refired_post_completion(self):
        """The core #59009 duplication: deltas streamed to the reasoning
        callback must NOT be re-fired as a full blob by
        build_assistant_message, even though text-stream callbacks are None
        (the exact callback shape the gateway configures)."""
        agent = _make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)

        # Simulate the streaming loop delivering reasoning deltas.
        agent._fire_reasoning_delta("Let me think ")
        agent._fire_reasoning_delta("about merging.")
        self.assertEqual(captured, ["Let me think ", "about merging."])

        # Post-completion: the accumulated reasoning arrives on the message.
        msg = SimpleNamespace(
            content="I'll merge that.",
            tool_calls=None,
            reasoning_content="Let me think about merging.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg, "stop")

        # No third (duplicated, accumulated) delivery.
        self.assertEqual(captured, ["Let me think ", "about merging."])

    def test_non_streamed_reasoning_fires_exactly_once(self):
        """When nothing streamed (non-streaming gateway turn), the
        post-completion path must still deliver reasoning — exactly once."""
        agent = _make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)

        msg = SimpleNamespace(
            content="Done.",
            tool_calls=None,
            reasoning_content="Reasoning that never streamed.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg, "stop")
        self.assertEqual(captured, ["Reasoning that never streamed."])

    def test_latch_is_read_and_cleared(self):
        """The latch must be scoped to a single response: consumed by one
        build_assistant_message call, so the NEXT response (no streaming)
        still gets its reasoning delivered."""
        agent = _make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)

        agent._fire_reasoning_delta("First response reasoning.")
        msg1 = SimpleNamespace(
            content="One.",
            tool_calls=None,
            reasoning_content="First response reasoning.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg1, "stop")

        # Second response: nothing streamed. Must fire.
        msg2 = SimpleNamespace(
            content="Two.",
            tool_calls=None,
            reasoning_content="Second response reasoning.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg2, "stop")

        self.assertEqual(
            captured,
            ["First response reasoning.", "Second response reasoning."],
        )

    def test_stale_latch_does_not_leak_without_callback_delivery(self):
        """_fire_reasoning_delta with no callback registered must NOT set
        the latch — nothing was delivered, so build_assistant_message
        (with a callback registered later, e.g. gateway wiring order)
        must still fire."""
        agent = _make_agent()
        agent._fire_reasoning_delta("dropped — no consumer")

        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)
        msg = SimpleNamespace(
            content="Done.",
            tool_calls=None,
            reasoning_content="Now-visible reasoning.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg, "stop")
        self.assertEqual(captured, ["Now-visible reasoning."])


class TestCliShapedCallbacks(unittest.TestCase):
    """CLI tag-extraction path: text-stream callbacks active, reasoning
    arrives inline in content — the latch never trips, but the existing
    text-stream suppression must be retained (regression contract also
    pinned in tests/cli/test_reasoning_command.py)."""

    def test_text_stream_active_suppresses_post_completion_fire(self):
        agent = _make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)
        agent.stream_delta_callback = lambda t: None  # streaming active

        # Reasoning came via content tag extraction (cli.py
        # _stream_reasoning_delta) — agent latch untouched.
        msg = SimpleNamespace(
            content="I'll merge that.",
            tool_calls=None,
            reasoning_content="Let me merge the PR.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg, "stop")
        self.assertEqual(captured, [])

    def test_internal_stream_callback_also_suppresses(self):
        agent = _make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)
        agent._stream_callback = lambda t: None
        msg = SimpleNamespace(
            content="Done.",
            tool_calls=None,
            reasoning_content="Reasoning.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg, "stop")
        self.assertEqual(captured, [])


class TestNormalizeReasoningDelta(unittest.TestCase):
    """Provider delta normalization: dedupe cumulative echoes and reconnect
    overlaps WITHOUT dropping legitimate repeated tokens."""

    def test_empty_delta(self):
        self.assertEqual(normalize_reasoning_delta("abc", ""), "")

    def test_first_delta_passthrough(self):
        self.assertEqual(normalize_reasoning_delta("", "Hello"), "Hello")

    def test_incremental_tokens_passthrough(self):
        """Normal providers: true incremental tokens append verbatim."""
        acc = "Let me think"
        self.assertEqual(normalize_reasoning_delta(acc, " about"), " about")

    def test_cumulative_snapshot_keeps_new_suffix(self):
        acc = "Let me think about this problem carefully"  # past the gate
        delta = acc + " and then merge."
        self.assertEqual(
            normalize_reasoning_delta(acc, delta), " and then merge."
        )

    def test_early_stream_snapshot_below_gate_appends_verbatim(self):
        """Contract: below _MIN_REASONING_OVERLAP accumulated chars the
        snapshot branch does NOT engage — a short prefix match is
        indistinguishable from legitimate repetition ("the" + "the quick"),
        and dropping real tokens is worse than briefly duplicating a short
        one. Cumulative providers self-heal via the overlap branch once
        past the gate."""
        acc = "the "
        delta = "the quick"
        self.assertEqual(normalize_reasoning_delta(acc, delta), delta)

    def test_exact_echo_dropped(self):
        """The doubled-halves bug: provider re-sends the full accumulated
        reasoning as a trailing chunk. Must be dropped entirely."""
        acc = "Full reasoning text that already streamed."
        self.assertEqual(normalize_reasoning_delta(acc, acc), "")

    def test_reconnect_overlap_trimmed(self):
        """Overlapping re-delivery: delta re-sends a long tail of already
        delivered text followed by new tokens — overlapped head trimmed."""
        overlap = "x" * (_MIN_REASONING_OVERLAP + 10)
        acc = "prefix text " + overlap
        delta = overlap + " NEW TOKENS"
        self.assertEqual(normalize_reasoning_delta(acc, delta), " NEW TOKENS")

    def test_short_repeated_token_not_dropped(self):
        """The reviewer-flagged over-broad case: a short delta that happens
        to be a substring of earlier text is LEGITIMATE repetition and must
        be appended, not discarded."""
        acc = "I think the answer is that the list"
        # " the" appears twice in acc already — still a valid new token.
        self.assertEqual(normalize_reasoning_delta(acc, " the"), " the")

    def test_short_suffix_overlap_not_trimmed(self):
        """Suffix/prefix overlaps below the gate are treated as legitimate
        text (e.g. 'so ' ending acc and starting delta), not re-delivery."""
        acc = "and so "
        delta = "so what happens next"
        self.assertEqual(normalize_reasoning_delta(acc, delta), delta)

    def test_repeated_phrase_verbatim_not_dropped(self):
        """A repeated phrase that is NOT a suffix/prefix overlap (appears
        mid-accumulated-text) must never be dropped, regardless of length."""
        phrase = "check the merge conflicts carefully before pushing"
        acc = f"First I will {phrase} and then run tests. Next"
        self.assertEqual(normalize_reasoning_delta(acc, phrase), phrase)

    def test_streaming_loop_dedupes_cumulative_echo_end_to_end(self):
        """Wire the normalizer the way the streaming loop uses it and feed
        the observed provider misbehavior: tokens then a full echo."""
        parts = []
        fired = []

        def _consume(chunk_text):
            text = normalize_reasoning_delta("".join(parts), chunk_text)
            if text:
                parts.append(text)
                fired.append(text)

        _consume("Let me ")
        _consume("think about ")
        _consume("merging.")
        _consume("Let me think about merging.")  # trailing cumulative echo

        self.assertEqual("".join(parts), "Let me think about merging.")
        self.assertEqual(fired, ["Let me ", "think about ", "merging."])


class TestExtractReasoningDetailsDedup(unittest.TestCase):
    """extract_reasoning must not double reasoning when a streamed response
    carries BOTH the accumulated ``reasoning`` string AND reasoning_details
    thinking blocks of the same content (dogfood-observed 2026-07-16:
    state.db reasoning columns byte-identical to blocks-joined × 2, because
    the old dedup was exact list membership and each individual block never
    equals the accumulated string)."""

    def _agent(self):
        return SimpleNamespace()

    def test_accumulated_plus_matching_blocks_not_doubled(self):
        from agent.agent_runtime_helpers import extract_reasoning

        blocks = [
            {"type": "thinking", "thinking": "First chunk of thinking."},
            {"type": "thinking", "thinking": "Second, different chunk."},
        ]
        accumulated = "First chunk of thinking.\n\nSecond, different chunk."
        msg = SimpleNamespace(
            reasoning=accumulated,
            reasoning_content=None,
            reasoning_details=blocks,
            content="final answer",
        )
        self.assertEqual(extract_reasoning(self._agent(), msg), accumulated)

    def test_distinct_blocks_all_survive(self):
        from agent.agent_runtime_helpers import extract_reasoning

        msg = SimpleNamespace(
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[
                {"type": "thinking", "thinking": "Alpha block."},
                {"type": "thinking", "thinking": "Beta block, different."},
            ],
            content="x",
        )
        self.assertEqual(
            extract_reasoning(self._agent(), msg),
            "Alpha block.\n\nBeta block, different.",
        )

    def test_genuinely_new_detail_content_kept(self):
        from agent.agent_runtime_helpers import extract_reasoning

        msg = SimpleNamespace(
            reasoning="Streamed text.",
            reasoning_content=None,
            reasoning_details=[
                {"type": "thinking", "thinking": "Streamed text."},
                {"type": "thinking", "thinking": "Novel unseen block."},
            ],
            content="x",
        )
        self.assertEqual(
            extract_reasoning(self._agent(), msg),
            "Streamed text.\n\nNovel unseen block.",
        )


if __name__ == "__main__":
    unittest.main()
