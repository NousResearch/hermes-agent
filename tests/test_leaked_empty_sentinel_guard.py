"""Regression: a leaked "(empty)" sentinel in a final answer must not end the turn.

Observed live (session 74508b0380e4, msg 126895): the model emitted
`(empty) again risk. Need progress + tool. Already can. Use execute.` with
phase=final_answer and finish_reason=stop, ending a 27-minute task mid-flight.
Content was non-empty, so the post-tool empty-response nudge never fired.

These tests pin the classifier decision, not the whole conversation loop:
the guard condition is a pure expression over `final_response`.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "agent" / "conversation_loop.py"


def _has_content_after_think_block(content: str) -> bool:
    """Mirror of AIAgent._has_content_after_think_block for the stripped tags."""
    if not content:
        return False
    cleaned = re.sub(
        r"<(think|thinking|reasoning)>.*?</\1>", "", content, flags=re.S | re.I
    )
    return bool(cleaned.strip())


def leaked_empty_sentinel(final_response: str) -> bool:
    """The condition added to conversation_loop.py, kept in sync by test below."""
    return (final_response or "").lstrip().startswith("(empty)") and (
        _has_content_after_think_block(final_response)
    )


def enters_recovery(final_response: str) -> bool:
    """True when the turn routes into the recovery/nudge block instead of ending."""
    return (
        not _has_content_after_think_block(final_response)
        or leaked_empty_sentinel(final_response)
    )


class TestLeakedEmptySentinel(unittest.TestCase):
    def test_the_real_leak_recovers(self):
        leak = "(empty) again risk. Need progress + tool. Already can. Use execute."
        self.assertTrue(leaked_empty_sentinel(leak))
        self.assertTrue(enters_recovery(leak), "leaked shorthand must not end the turn")

    def test_bare_sentinel_still_recovers(self):
        self.assertTrue(enters_recovery("(empty)"))

    def test_leading_whitespace_does_not_evade(self):
        self.assertTrue(enters_recovery("\n  (empty) need tool"))

    # ---- false-positive guards: legitimate answers MUST still end the turn ----

    def test_short_legitimate_answer_still_ends_turn(self):
        for good in ("Done.", "PASS", "OK", "42", "Yes — verified."):
            with self.subTest(good=good):
                self.assertFalse(leaked_empty_sentinel(good))
                self.assertFalse(enters_recovery(good), f"{good!r} must end the turn")

    def test_answer_merely_mentioning_the_word_empty_ends_turn(self):
        for good in (
            "The empty response guard fired once.",
            "`(empty)` is the sentinel Hermes injects.",
            "Result: the list was empty.",
        ):
            with self.subTest(good=good):
                self.assertFalse(
                    enters_recovery(good),
                    "sentinel must be a PREFIX match, not a substring match",
                )

    def test_markdown_answer_ends_turn(self):
        self.assertFalse(enters_recovery("## Triage complete\n\n- 21 tickets\n"))

    def test_thinking_only_still_recovers(self):
        self.assertTrue(enters_recovery("<think>weighing options</think>"))


class TestGuardStaysWiredInSource(unittest.TestCase):
    """The mirror above is only meaningful if the real file still uses it."""

    def test_source_defines_and_uses_the_guard(self):
        src = SRC.read_text(encoding="utf-8")
        self.assertIn("_leaked_empty_sentinel = (", src)
        self.assertIn('.lstrip().startswith("(empty)")', src)
        self.assertIn("or _leaked_empty_sentinel", src)

    def test_partial_stream_recovery_cannot_redeliver_the_leak(self):
        """The streamed text holds the leak verbatim — it must be fenced out."""
        src = SRC.read_text(encoding="utf-8")
        self.assertIn(
            "if not _leaked_empty_sentinel and agent._has_content_after_think_block(_partial_streamed):",
            src,
        )


if __name__ == "__main__":
    unittest.main()
