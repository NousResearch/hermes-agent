"""Unit tests for repetition-dominated model output detection."""

from __future__ import annotations

import random

import pytest

from agent.repetition_guard import MIN_FRAGMENT_LENGTH, is_repetition_dominated, is_runaway_repetition

# The exact sentence from the #86581 incident (echoed hundreds of times by
# the model before the provider cut it off at finish_reason=length).
_INCIDENT_ECHO = "好，你幫我更改成 Google Gemini 4 31B。"

# Ordinary reasoning: distinct lines, written before a loop starts.
_DISTINCT_LINES = "".join(
    f"Check {i}: service {i} answers on port {8000 + i}, its config parses and the redirect for host-{i} is in place.\n"
    for i in range(16)
)


class TestRepetitionGuard:
    def test_incident_shape_flags_repetition(self):
        # Narration + the echoed sentence on its own line, repeated (line path).
        text = ("We need to verify the model setting.\n" + _INCIDENT_ECHO + "\n") * 800
        assert is_repetition_dominated(text) is True

    def test_repeated_sentence_without_line_breaks_flags(self):
        # Repetition loop with no line breaks — exercises the window path.
        text = _INCIDENT_ECHO * 2000
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is True

    def test_multiline_paragraph_run_uses_true_period_coverage(self):
        rng = random.Random(11)
        paragraph = "\n".join(
            "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(151))
            for _ in range(5)
        ) + "\n"

        # The incident unit was approximately 764 characters. Detection must
        # remain scale-free as the number of exact repeats grows.
        for repeat_count in (100, 1_000, 10_000):
            assert is_repetition_dominated(paragraph * repeat_count) is True

    @pytest.mark.parametrize("shape", ["unique_prefix_suffix", "counter_loop"])
    def test_dominant_run_with_unique_prefix_and_suffix_flags(self, shape):
        if shape == "counter_loop":
            # A changing counter breaks exact periodicity; main's window scan (#86581) must
            # still flag it.
            text = "".join(
                f"Step {i}: I will now carefully re-check the configuration file for the error again.\n"
                for i in range(200)
            )
            assert is_repetition_dominated(text) is True
            return
        paragraph = (
            "A deliberately long repeated paragraph has enough distinct text "
            "to make its period exceed the guard's minimum anchor length.\n"
            "It also spans multiple lines, matching the real incident shape.\n"
        )
        repeated = paragraph * 20
        text = ("unique introduction " * 20) + repeated + (" unique ending" * 20)

        assert len(repeated) > len(text) * 0.5
        assert is_repetition_dominated(text) is True

    def test_long_legitimate_text_not_flagged(self):
        # Long, unique prose — no 60-char window ever repeats.
        text = " ".join(
            f"Sentence number {i} describes a distinct topic with unique words "
            f"such as quasar-{i} and nebula-{i} to keep every window distinct."
            for i in range(1200)
        )
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is False

    def test_short_fragment_never_flagged(self):
        # Below MIN_FRAGMENT_LENGTH the guard fails open — short truncations
        # are legitimately continued even if they look repetitive.
        assert is_repetition_dominated("A. " * 50) is False
        assert is_repetition_dominated("hello ") is False

    def test_repeat_not_dominant_not_flagged(self):
        # A repeated sentence scattered through a long unique text: repeated
        # windows exist but cover far less than half of the fragment.
        filler = " ".join(f"unique filler token {i}" for i in range(3000))
        text = filler + ("\n" + _INCIDENT_ECHO + "\n") * 30
        assert is_repetition_dominated(text) is False

    def test_non_string_inputs(self):
        assert is_repetition_dominated("") is False
        assert is_repetition_dominated(None) is False
        assert is_repetition_dominated(12345) is False


class TestRunawayRepetition:
    def test_loop_without_newlines_after_distinct_lines_is_runaway(self):
        # A GLM-5.3-Flash thinking turn: distinct lines, then " hmm —" x890 and one 8-char token
        # repeated without a newline. The distinct lines before the loop must not veto the cut.
        loop_line = " hmm —" * 890 + " a" + "aaaaaaaa" * 5000
        text = _DISTINCT_LINES + loop_line

        assert len(text.splitlines()) > 5
        assert is_runaway_repetition(text) is True

    def test_distinct_rows_sharing_a_suffix_are_not_runaway(self):
        # Batch output: the shared suffix trips the window scan, but every row differs.
        text = "".join(
            f"| {i:05d} | status: pending review by the on-call engineer before the next release window |\n"
            for i in range(500)
        )

        assert is_repetition_dominated(text) is True
        assert is_runaway_repetition(text) is False

    def test_one_long_line_of_distinct_records_is_not_runaway(self):
        # Minified JSON after a few lines: one long line of records that differ only by their id.
        records = "".join(
            f'{{"id": {i}, "status": "pending review by the on-call engineer before the next release"}}, '
            for i in range(600)
        )
        text = _DISTINCT_LINES + "[" + records + "]\n"

        assert is_repetition_dominated(text) is True
        assert is_runaway_repetition(text) is False
