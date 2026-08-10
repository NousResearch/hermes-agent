"""Regression: a leaked "(empty)" sentinel in a final answer must not end the turn.

Observed live (session 74508b0380e4, msg 126895): the model emitted
`(empty) again risk. Need progress + tool. Already can. Use execute.` with
phase=final_answer and finish_reason=stop, ending a 27-minute task mid-flight.
Content was non-empty, so the post-tool empty-response nudge never fired.

DESIGN NOTE — no mirrored oracle.
An earlier version of this file defined its own copies of
``_has_content_after_think_block`` and the guard expression. 7 of its 9 tests
then passed with the production fix reverted, because they tested the copies.
This version imports the REAL production helpers and extracts the REAL guard
expression from source, so reverting the fix fails the behavioural tests.
"""
import ast
import re
import unittest
from pathlib import Path

from agent.agent_runtime_helpers import strip_think_blocks
from agent.message_content import flatten_message_text

SRC = Path(__file__).resolve().parents[1] / "agent" / "conversation_loop.py"
SOURCE = SRC.read_text(encoding="utf-8")

LEAK = "(empty) again risk. Need progress + tool. Already can. Use execute."


class _MinimalAgent:
    """Only what the production helpers actually touch."""

    def _strip_think_blocks(self, content):
        return strip_think_blocks(self, content)

    def _has_content_after_think_block(self, content):
        # Production body, verbatim (run_agent.py:1649) — delegates to the
        # real strip_think_blocks, so tag handling cannot drift.
        if not content:
            return False
        return bool(self._strip_think_blocks(content).strip())


def _extract_guard_source():
    """Pull the real guard expression out of conversation_loop.py.

    Fails loudly if the fix is absent, so a revert cannot silently pass.
    """
    m = re.search(
        r"_leaked_empty_sentinel = \(\n(.*?)\n\s*\)\n", SOURCE, re.S
    )
    if not m:
        raise AssertionError(
            "guard `_leaked_empty_sentinel = (...)` not found in "
            "agent/conversation_loop.py — the fix is missing or was reworded"
        )
    return m.group(1)


GUARD_SRC = _extract_guard_source()


def leaked_empty_sentinel(agent, final_response):
    """Evaluate the PRODUCTION guard expression against real helpers."""
    expr = GUARD_SRC.replace("_flatten_final", "flatten_message_text")
    # Collapse to a single expression and evaluate with production callables.
    return bool(
        eval(  # noqa: S307 - evaluating our own source under test, by design
            " ".join(expr.split()),
            {
                "flatten_message_text": flatten_message_text,
                "agent": agent,
                "final_response": final_response,
            },
        )
    )


def enters_recovery(agent, final_response):
    """The full `if` condition guarding the empty-recovery block."""
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
        self.assertTrue(
            enters_recovery(self.agent, LEAK),
            "leaked shorthand must not end the turn",
        )

    def test_leading_whitespace_does_not_evade(self):
        self.assertTrue(enters_recovery(self.agent, "\n  (empty) need tool"))

    def test_multimodal_list_content_does_not_crash(self):
        """A non-empty list is truthy: a bare .lstrip() raised AttributeError.

        Anthropic-via-OpenRouter really returns this shape, which is why
        strip_think_blocks() coerces its input.
        """
        multimodal = [
            {"type": "text", "text": "(empty) need tool"},
            {"type": "thinking", "thinking": "scratch"},
        ]
        self.assertTrue(leaked_empty_sentinel(self.agent, multimodal))

    def test_multimodal_legitimate_answer_does_not_crash_or_trip(self):
        multimodal = [{"type": "text", "text": "The chart shows a decline."}]
        self.assertFalse(leaked_empty_sentinel(self.agent, multimodal))

    # ---- false-positive guards: legitimate answers MUST still end the turn ----

    def test_short_legitimate_answer_still_ends_turn(self):
        for good in ("Done.", "PASS", "OK", "42", "Yes — verified."):
            with self.subTest(good=good):
                self.assertFalse(leaked_empty_sentinel(self.agent, good))
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
    """Meta-test: this file must not reimplement the logic it tests."""

    def test_guard_is_extracted_from_production_not_retyped(self):
        self.assertIn("_leaked_empty_sentinel = (", SOURCE)
        self.assertIn("re.search", Path(__file__).read_text(encoding="utf-8"))

    def test_helpers_are_imported_not_copied(self):
        own = Path(__file__).read_text(encoding="utf-8")
        self.assertIn("from agent.agent_runtime_helpers import strip_think_blocks", own)
        # A local `def _strip_think_blocks` copy would be the mirror antipattern.
        self.assertNotIn("re.sub(r\"<(think", own)

    def test_guard_uses_a_type_tolerant_flattener(self):
        """Pin the BLOCKER fix: a bare .lstrip() on content crashes on lists.

        Comments are stripped first: the explanatory comment above the guard
        legitimately quotes the old broken expression, and a naive substring
        assertion would match its own documentation (the
        'comment-grepping tests self-trip' pitfall).
        """
        code_only = "\n".join(
            line.split("#", 1)[0] for line in SOURCE.splitlines()
        )
        self.assertNotIn('(final_response or "").lstrip()', code_only)
        self.assertIn("_flatten_final(final_response)", code_only)

    def test_partial_stream_recovery_cannot_redeliver_the_leak(self):
        tree = ast.parse(SOURCE)
        self.assertTrue(
            any(
                isinstance(n, ast.Name) and n.id == "_leaked_empty_sentinel"
                for n in ast.walk(tree)
            ),
            "guard variable vanished from production",
        )
        self.assertIn(
            "if not _leaked_empty_sentinel and agent._has_content_after_think_block("
            "_partial_streamed):",
            SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
