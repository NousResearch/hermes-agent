"""Regression: a leaked "(empty)" sentinel in a final answer must not end the turn.

Observed live (session 74508b0380e4, msg 126895): the model emitted
`(empty) again risk. Need progress + tool. Already can. Use execute.` with
phase=final_answer and finish_reason=stop, ending a 27-minute task mid-flight.
Content was non-empty, so the post-tool empty-response nudge never fired.

DESIGN NOTE — no mirrored oracle.
Round 1 of this file defined its own copies of `_has_content_after_think_block`
and of the guard expression; 7 of its 9 tests then passed with the production
fix reverted, because they tested the copies. Round 2 still inlined the
production body of `_has_content_after_think_block`.

This version imports every production symbol it exercises:
  * `_visible_text_for_sentinel_check` and `EMPTY_RESPONSE_SENTINEL`
    from `agent.conversation_loop`
  * `AIAgent._has_content_after_think_block` bound to a minimal object
so a change to any of them fails here rather than silently diverging.
"""
import ast
import unittest
from pathlib import Path

import run_agent
from agent.conversation_loop import (
    EMPTY_RESPONSE_SENTINEL,
    _visible_text_for_sentinel_check,
)

SRC = Path(__file__).resolve().parents[1] / "agent" / "conversation_loop.py"
SOURCE = SRC.read_text(encoding="utf-8")
OWN_SOURCE = Path(__file__).read_text(encoding="utf-8")

LEAK = "(empty) again risk. Need progress + tool. Already can. Use execute."


class _MinimalAgent:
    """Borrows the REAL predicate off AIAgent — no reimplementation."""

    _has_content_after_think_block = run_agent.AIAgent._has_content_after_think_block
    _strip_think_blocks = run_agent.AIAgent._strip_think_blocks


def _guard_source() -> str:
    """The production guard, located by AST (not by brittle text search).

    Raises if the fix is absent so a revert cannot pass silently.
    """
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_leaked_empty_sentinel"
                for t in node.targets
            )
        ):
            return ast.unparse(node.value)
    raise AssertionError(
        "`_leaked_empty_sentinel = ...` not found in agent/conversation_loop.py "
        "— the fix is missing or was renamed"
    )


GUARD_EXPR = _guard_source()


def leaked_empty_sentinel(agent, final_response):
    """Evaluate the PRODUCTION guard expression against PRODUCTION helpers."""
    return bool(
        eval(  # noqa: S307 - our own source under test, by design
            GUARD_EXPR,
            {
                "_visible_text_for_sentinel_check": _visible_text_for_sentinel_check,
                "EMPTY_RESPONSE_SENTINEL": EMPTY_RESPONSE_SENTINEL,
                "agent": agent,
                "final_response": final_response,
                "_visible_final": _visible_text_for_sentinel_check(final_response),
            },
        )
    )


def enters_recovery(agent, final_response):
    return (
        not agent._has_content_after_think_block(final_response)
        or leaked_empty_sentinel(agent, final_response)
    )


class TestLeakedEmptySentinel(unittest.TestCase):
    def setUp(self):
        self.agent = _MinimalAgent()

    # ---- the bug: these FAIL when the production fix is reverted ----

    def test_the_real_leak_recovers(self):
        self.assertTrue(leaked_empty_sentinel(self.agent, LEAK))
        self.assertTrue(enters_recovery(self.agent, LEAK))

    def test_leading_whitespace_does_not_evade(self):
        self.assertTrue(enters_recovery(self.agent, "\n  (empty) need tool"))

    # ---- BLOCKER from review round 1: crash on non-str content ----

    def test_multimodal_list_content_does_not_crash(self):
        self.assertTrue(
            leaked_empty_sentinel(
                self.agent,
                [{"type": "text", "text": "(empty) need tool"}],
            )
        )

    def test_multimodal_legitimate_answer_does_not_trip(self):
        self.assertFalse(
            leaked_empty_sentinel(
                self.agent, [{"type": "text", "text": "The chart shows a decline."}]
            )
        )

    # ---- FALSE POSITIVE from review round 2: reasoning must not be judged ----

    def test_thinking_block_starting_with_sentinel_does_not_trip(self):
        """A scratchpad note must never condemn a good visible answer.

        flatten_message_text() keeps reasoning parts and accepts the generic
        `content` key, so {"type":"thinking","content":"(empty) ..."} used to
        flatten AHEAD of the real answer and fire the guard.
        """
        for reasoning_key in ("thinking", "content", "text"):
            for part_type in ("thinking", "reasoning", "redacted_thinking"):
                with self.subTest(type=part_type, key=reasoning_key):
                    content = [
                        {"type": part_type, reasoning_key: "(empty) scratch note"},
                        {"type": "text", "text": "Done."},
                    ]
                    self.assertFalse(
                        leaked_empty_sentinel(self.agent, content),
                        f"{part_type}/{reasoning_key} leaked into the visible check",
                    )

    def test_visible_leak_still_caught_alongside_reasoning(self):
        content = [
            {"type": "thinking", "thinking": "all good"},
            {"type": "text", "text": "(empty) need tool"},
        ]
        self.assertTrue(leaked_empty_sentinel(self.agent, content))

    def test_helper_never_raises_on_hostile_shapes(self):
        for junk in (None, 0, object(), [None], [{"type": None}], {"no": "type"}):
            with self.subTest(junk=junk):
                self.assertIsInstance(_visible_text_for_sentinel_check(junk), str)

    # ---- false-positive guards: legitimate answers MUST still end the turn ----

    def test_short_legitimate_answer_still_ends_turn(self):
        for good in ("Done.", "PASS", "OK", "42", "Yes — verified."):
            with self.subTest(good=good):
                self.assertFalse(enters_recovery(self.agent, good))

    def test_answer_merely_mentioning_the_word_empty_ends_turn(self):
        for good in (
            "The empty response guard fired once.",
            "`(empty)` is the sentinel Hermes injects.",
            "Result: the list was empty.",
        ):
            with self.subTest(good=good):
                self.assertFalse(
                    enters_recovery(self.agent, good),
                    "sentinel must be a PREFIX match, not a substring match",
                )

    def test_markdown_answer_ends_turn(self):
        self.assertFalse(
            enters_recovery(self.agent, "## Triage complete\n\n- 21 tickets\n")
        )

    def test_thinking_only_still_recovers(self):
        self.assertTrue(enters_recovery(self.agent, "<think>weighing</think>"))


class TestNoMirroredOracle(unittest.TestCase):
    """Meta-tests: this file must not reimplement what it tests."""

    def test_predicate_is_borrowed_from_production(self):
        self.assertIs(
            _MinimalAgent._has_content_after_think_block,
            run_agent.AIAgent._has_content_after_think_block,
        )

    def test_no_local_copy_of_the_stripper(self):
        # Compare against code only: the assertions themselves contain these
        # needles, so scanning raw source would always self-trip.
        code_only = "\n".join(
            line for line in OWN_SOURCE.splitlines()
            if "assertNotIn" not in line and "assertIn" not in line
        )
        self.assertNotIn("def _strip_think_blocks(self", code_only)
        self.assertNotIn("_NON_VISIBLE_CONTENT_PART_TYPES = ", code_only)

    def test_guard_expression_is_extracted_not_retyped(self):
        self.assertIn("ast.unparse", OWN_SOURCE)
        # The unparsed guard binds the pre-computed visible text and the shared
        # constant — proof it came from production, not from this file.
        self.assertIn("_visible_final", GUARD_EXPR)
        self.assertIn("EMPTY_RESPONSE_SENTINEL", GUARD_EXPR)
        self.assertIn("_has_content_after_think_block", GUARD_EXPR)

    def test_sentinel_is_a_shared_constant(self):
        self.assertEqual(EMPTY_RESPONSE_SENTINEL, "(empty)")
        code_only = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
        self.assertIn("startswith(EMPTY_RESPONSE_SENTINEL)", code_only)

    def test_guard_does_not_use_a_bare_lstrip(self):
        """Comments are stripped: they legitimately quote the old broken code."""
        code_only = "\n".join(l.split("#", 1)[0] for l in SOURCE.splitlines())
        self.assertNotIn('(final_response or "").lstrip()', code_only)

    def test_partial_stream_recovery_cannot_redeliver_the_leak(self):
        self.assertIn(
            "if not _leaked_empty_sentinel and agent._has_content_after_think_block("
            "_partial_streamed):",
            SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
