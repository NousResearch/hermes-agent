"""Unit tests for repetition-dominated model output detection."""

from __future__ import annotations

import random

import pytest

import json

from agent.repetition_guard import (
    LINE_CYCLE_MIN_LINES,
    MIN_FRAGMENT_LENGTH,
    is_line_cycle,
    is_repetition_dominated,
)

# The exact sentence from the #86581 incident (echoed hundreds of times by
# the model before the provider cut it off at finish_reason=length).
_INCIDENT_ECHO = "好，你幫我更改成 Google Gemini 4 31B。"


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


def _closing_loop(units: int, seed: int = 1) -> str:
    """The captured write_file loop: three fixed lines and one rotating closing line."""
    rng = random.Random(seed)
    closings = ["Complete.", "Final status: complete.", "Closeout.", "Launch.", "Final checklist delivered.", "Ready.", "Done."]
    return "".join(
        f"\n\n**End.**\n\n**Bluebell-9213bf**\n\n**14 May**\n\n**{rng.choice(closings)}**" for _ in range(units)
    )


class TestLineCycle:
    def test_should_flag_a_loop_whose_closing_line_varies(self):
        assert is_line_cycle(_closing_loop(600)) is True

    def test_should_flag_the_loop_inside_json_escaped_tool_arguments(self):
        arguments = json.dumps({"path": "/tmp/checklist.md", "content": _closing_loop(600)})
        assert "\n" not in arguments  # one long line, as streamed
        assert is_line_cycle(arguments[-32_000:]) is True

    def test_should_catch_the_loop_that_the_exact_repeat_scans_miss(self):
        arguments = json.dumps({"path": "/tmp/checklist.md", "content": _closing_loop(3_000)})
        assert is_repetition_dominated(arguments[:64_000]) is False
        assert is_line_cycle(arguments[-32_000:]) is True

    def test_should_ignore_a_cycle_below_the_line_floor(self):
        lines = ["**End.**", "**Bluebell**", "**14 May**"]
        text = "\n".join(lines[i % 3] for i in range(LINE_CYCLE_MIN_LINES - 1))
        assert is_line_cycle(text) is False

    def test_should_require_fewer_than_five_percent_distinct_lines(self):
        # 400 lines: 20 distinct is exactly 5% (kept), 19 distinct is below it (flagged).
        at_ratio = "\n".join(f"line {i % 20}" for i in range(400))
        below_ratio = "\n".join(f"line {i % 19}" for i in range(400))
        assert is_line_cycle(at_ratio) is False
        assert is_line_cycle(below_ratio) is True

    @pytest.mark.parametrize(
        "text",
        [
            "\n".join(f"{i},user{i}@example.com,{i * 7919 % 997},active" for i in range(2_000)),
            json.dumps([{"id": i, "name": f"item {i}", "tags": ["a", "b"], "active": True} for i in range(1_500)], indent=2),
            "\n".join(
                f"    def handler_{i}(self, request):\n        return self.dispatch(request, {i})\n" for i in range(600)
            ),
        ],
        ids=["csv_rows", "pretty_json_array", "repetitive_code"],
    )
    def test_should_keep_real_files_with_repetitive_structure(self, text):
        assert is_line_cycle(text[-32_000:]) is False
        assert is_line_cycle(json.dumps({"content": text})[-32_000:]) is False

    def test_should_count_escaped_carriage_returns_as_line_breaks(self):
        text = "**End.**\\r\\n**14 May**\\r\\n" * 300
        assert is_line_cycle(text) is True

    @pytest.mark.parametrize("value", [None, 42, b"**End.**\n" * 500])
    def test_should_ignore_non_string_input(self, value):
        assert is_line_cycle(value) is False
