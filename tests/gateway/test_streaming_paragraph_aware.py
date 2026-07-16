"""Tests for paragraph-aware in-stream split logic.

The stream consumer splits long replies into multiple Telegram bubbles at
natural boundaries (paragraph > sentence > line > word) rather than at
arbitrary character counts. This avoids cutting mid-sentence and bounds
the per-bubble edit-animation time so Telegram's client doesn't replay
animations as successive edits land ("flashing").

See gateway/stream_consumer.py::find_paragraph_break for the heuristic.
"""

import pytest

from gateway.stream_consumer import find_paragraph_break


# ── find_paragraph_break unit tests ──────────────────────────────────────


class TestFindParagraphBreak:
    """Verify the split-point locator across the documented boundary hierarchy."""

    SOFT = 600   # max_edit_chars
    HARD = 1500  # hard_edit_chars

    def test_short_text_returns_none(self):
        """Text below soft_target is never split — short replies stay whole."""
        text = "Short reply. " * 10  # ~130 chars
        assert find_paragraph_break(text, self.SOFT, self.HARD) is None

    def test_paragraph_break_preferred(self):
        """\n\n beats any other boundary in the same window."""
        para1 = "word " * 200  # ~1000 chars, no internal paragraph break
        text = para1 + "\n\n" + ("word " * 200)
        result = find_paragraph_break(text, self.SOFT, self.HARD)
        assert result is not None
        # The split must land on the \n\n, not past it.
        assert result < len(para1) + 2  # +2 for \n\n itself
        assert text[result:result + 2] == "\n\n"

    def test_sentence_break_used_when_no_paragraph(self):
        """Single-paragraph long prose splits at the earliest sentence end."""
        sentence = "This is one long sentence with no paragraph breaks. "
        text = sentence * 30  # ~1500 chars, all one paragraph
        result = find_paragraph_break(text, self.SOFT, self.HARD)
        assert result is not None
        # Must land on a sentence-ending ". " somewhere in [soft, hard].
        assert self.SOFT <= result <= self.HARD
        assert text[result - 2:result] == ". "

    def test_line_break_used_when_no_paragraph_or_sentence(self):
        """Falls back to \\n when prose has line breaks but no sentence ends."""
        line = "no punctuation here just letters and spaces "  # ~46 chars
        text = (line + "\n") * 30  # ~1500 chars, only line breaks
        result = find_paragraph_break(text, self.SOFT, self.HARD)
        assert result is not None
        assert self.SOFT <= result <= self.HARD
        assert text[result] == "\n"

    def test_returns_none_when_text_too_short_for_soft_target(self):
        """soft_target >= len(text) → no split (would produce empty chunks)."""
        text = "x" * 500
        assert find_paragraph_break(text, 600, 1500) is None

    def test_hard_limit_zero_never_splits(self):
        """hard_limit=0 is the off-switch — never split."""
        text = "word " * 1000
        assert find_paragraph_break(text, 600, 0) is None

    def test_no_boundary_before_hard_limit(self):
        """Single-paragraph wall of text > hard_limit → no natural split."""
        # One giant sentence longer than hard_limit, no breaks at all.
        text = "a" * 2000
        # soft_target > 0, hard_limit > len(text) here is impossible since
        # text is 2000 and hard_limit defaults to 1500 — but the function
        # only searches [soft, hard], so 2000 chars with no boundaries
        # within the window → None.
        result = find_paragraph_break(text, 600, 1500)
        assert result is None

    def test_question_mark_treated_as_sentence_end(self):
        """? followed by space is a valid sentence boundary."""
        text = ("Short question? " * 200)  # ~3200 chars, sentence breaks
        result = find_paragraph_break(text, 600, 1500)
        assert result is not None
        assert self.SOFT <= result <= self.HARD
        assert text[result - 2:result] == "? "

    def test_exclamation_treated_as_sentence_end(self):
        """! followed by space is a valid sentence boundary."""
        text = ("Whoa! " * 200)  # ~1000 chars
        result = find_paragraph_break(text, 600, 1500)
        assert result is not None
        assert self.SOFT <= result <= self.HARD

    def test_punctuation_without_trailing_space_not_a_boundary(self):
        """'foo.' followed immediately by 'bar' (no space) is NOT a split."""
        text = ("word." * 500)  # 2500 chars, but no ". " anywhere
        result = find_paragraph_break(text, 600, 1500)
        assert result is None  # Falls through to no-boundary case

    def test_split_position_is_within_window(self):
        """Returned position must be in [soft_target, hard_limit)."""
        text = ("Sentence one. " * 50) + ("\n\n" + "Sentence two. " * 50)
        result = find_paragraph_break(text, 600, 1500)
        assert result is not None
        assert 600 <= result <= 1500


# ── Integration with StreamingConfig defaults ────────────────────────────


class TestStreamingConfigDefaults:
    """The two new knobs must default to sensible values and round-trip through to_dict/from_dict."""

    def test_defaults_match_documented_values(self):
        from gateway.config import (
            DEFAULT_STREAMING_HARD_EDIT_CHARS,
            DEFAULT_STREAMING_MAX_EDIT_CHARS,
            StreamingConfig,
        )
        cfg = StreamingConfig()
        assert cfg.max_edit_chars == DEFAULT_STREAMING_MAX_EDIT_CHARS == 600
        assert cfg.hard_edit_chars == DEFAULT_STREAMING_HARD_EDIT_CHARS == 1500

    def test_to_dict_includes_new_keys(self):
        from gateway.config import StreamingConfig
        cfg = StreamingConfig()
        d = cfg.to_dict()
        assert d["max_edit_chars"] == 600
        assert d["hard_edit_chars"] == 1500

    def test_from_dict_reads_new_keys(self):
        from gateway.config import StreamingConfig
        cfg = StreamingConfig.from_dict({
            "max_edit_chars": 400,
            "hard_edit_chars": 900,
        })
        assert cfg.max_edit_chars == 400
        assert cfg.hard_edit_chars == 900

    def test_from_dict_uses_defaults_when_keys_missing(self):
        """Older config.yaml files without the new keys must still load."""
        from gateway.config import (
            DEFAULT_STREAMING_HARD_EDIT_CHARS,
            DEFAULT_STREAMING_MAX_EDIT_CHARS,
            StreamingConfig,
        )
        cfg = StreamingConfig.from_dict({"enabled": True})
        assert cfg.max_edit_chars == DEFAULT_STREAMING_MAX_EDIT_CHARS
        assert cfg.hard_edit_chars == DEFAULT_STREAMING_HARD_EDIT_CHARS