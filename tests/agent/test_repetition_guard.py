"""Unit tests for the truncated-response repetition guard (issue #86581)."""

from __future__ import annotations

from agent.repetition_guard import (
    MIN_FRAGMENT_LENGTH,
    StreamingRepetitionGuard,
    is_repetition_dominated,
)
from agent.turn_truncation import _REPETITION_DOMINATED, _abort_reason

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

    def test_streaming_guard_stops_reasoning_loop_early(self):
        guard = StreamingRepetitionGuard()
        detected_at = None
        loop = "Deixa'm executar la crida read_file per comprovar-ho ara mateix. "
        for i in range(200):
            if guard.feed(loop):
                detected_at = (i + 1) * len(loop)
                break
        assert detected_at is not None
        assert detected_at < 4096  # stop well before the configured output-token ceiling

    def test_streaming_guard_allows_long_unique_reasoning(self):
        guard = StreamingRepetitionGuard()
        for i in range(200):
            assert guard.feed(
                f"Step {i} checks a different condition with unique marker quasar-{i}-nebula-{i}. "
            ) is False

    def test_truncation_abort_checks_structured_reasoning(self):
        class Agent:
            _has_content_after_think_block = staticmethod(lambda _text: False)
            _strip_think_blocks = staticmethod(lambda text: text)

        reasoning = (
            "Deixa'm executar la crida read_file per comprovar-ho ara mateix. " * 200
        )
        assert _abort_reason(Agent(), None, False, reasoning) == _REPETITION_DOMINATED
        assert _abort_reason(Agent(), None, True, reasoning) is None
