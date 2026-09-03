"""Unit tests for the truncated-response repetition guard (issue #86581)."""

from __future__ import annotations

from agent.repetition_guard import MIN_FRAGMENT_LENGTH, is_repetition_dominated

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

    def test_multipart_list_content_flagged(self):
        # Multipart list where text parts contain repeated sentences
        parts = [
            {"type": "text", "text": _INCIDENT_ECHO * 1000},
            {"type": "text", "text": _INCIDENT_ECHO * 1000},
        ]
        assert is_repetition_dominated(parts) is True

    def test_multipart_list_non_repeating_not_flagged(self):
        parts = [
            {"type": "text", "text": "Unique intro part describing architecture."},
            {"type": "text", "text": "Unique body part detailing verification steps."},
        ]
        assert is_repetition_dominated(parts) is False

    def test_multipart_with_images_and_tool_calls_not_flagged(self):
        """Non-text parts (images, tool_use) must not be stringified into false repetition signals."""
        parts = [
            {"type": "text", "text": "Here is the screenshot requested:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "tool_use", "id": "call_1", "name": "screenshot", "input": {}},
        ]
        assert is_repetition_dominated(parts) is False

    def test_multipart_repeated_table_syntax_below_dominance_not_flagged(self):
        """Legitimate repeated markdown tables or code syntax below dominance threshold must pass."""
        table_rows = ["| col1 | col2 | col3 |", "| --- | --- | --- |"] + [
            f"| data_{i} | value_{i} | result_{i} |" for i in range(100)
        ]
        parts = [
            {"type": "text", "text": "\n".join(table_rows)},
        ]
        assert is_repetition_dominated(parts) is False

