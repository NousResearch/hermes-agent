"""Tests for `_is_placeholder_transcript` — the single source of truth for
rejecting literal "null" / "None" / "undefined" / "nan" / empty transcripts
emitted by misbehaving STT pipelines.

Contract:
  - Returns True ONLY for the canonical placeholder set, case-insensitive,
    after strip().
  - Returns False for any non-empty real content (even single chars).
  - Returns False for None / non-string inputs (None is treated as empty).

This guard is consumed by both the wake-capture path
(`_on_post_wake_audio_done`) and the barge-capture path
(`_voice_submit_barge_utterance`) — see `cli.py` for the call sites.
"""

import pytest

from cli import _is_placeholder_transcript, _STT_PLACEHOLDER_TRANSCRIPTS


class TestIsPlaceholderTranscript:
    """The placeholder set is the union of strings STT pipelines have
    been observed to emit instead of a real transcript when the upstream
    API errors silently."""

    @pytest.mark.parametrize(
        "value",
        ["null", "NULL", "Null", "  null  ", "None", "NONE", "undefined",
         "Undefined", "nan", "NaN", ""],
    )
    def test_canonical_placeholders_rejected(self, value):
        assert _is_placeholder_transcript(value) is True

    @pytest.mark.parametrize(
        "value",
        ["hello", "h", " no ", "nul", "non", "undefinedvariable",
         "n", "a", "1", "0", "nulls"],
    )
    def test_real_content_accepted(self, value):
        assert _is_placeholder_transcript(value) is False

    @pytest.mark.parametrize("value", [None, b"null", b""])
    def test_non_string_or_none_treated_as_empty(self, value):
        # The helper must NOT crash on bytes / None — STT shims have
        # been observed to return bytes (raw provider output) or None
        # (early-return short-circuit) on internal error.  Bytes that
        # decode to placeholder text should still be rejected.
        result = _is_placeholder_transcript(value)
        assert isinstance(result, bool)
        if value in (None, b""):
            assert result is True
        else:
            assert result is True  # b"null" decodes to "null"

    def test_set_is_frozen(self):
        """The set is exported as frozenset so callers can't drift the
        membership over the lifecycle of the process."""
        assert isinstance(_STT_PLACEHOLDER_TRANSCRIPTS, frozenset)