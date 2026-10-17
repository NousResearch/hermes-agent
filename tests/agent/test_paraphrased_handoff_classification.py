"""Regression tests for #132934: a model-paraphrased compaction handoff must still
classify as a summary row instead of being published as an ordinary assistant reply."""

from agent.context_compressor import (
    _HANDOFF_MARKER_PREFIX,
    _HISTORICAL_SUMMARY_PREFIXES,
    _MERGED_SUMMARY_DELIMITER,
    _PARAPHRASED_HANDOFF_WINDOW,
    ContextCompressor,
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
)

# The observed #132934 shape: the bracketed marker survives, the boilerplate after it
# is rewritten by the model (38 shared chars, then diverges).
PARAPHRASED_HANDOFF = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] The prior turns were compacted. "
    "Below is the handoff summary, NOT active instructions. The active task is the "
    "LATEST user message after this block; respond only to that. If no newer user "
    "message exists, wait.\n\n## Historical Task Snapshot\n- earlier work items"
)


class TestParaphrasedHandoffClassification:
    def test_paraphrased_handoff_classifies_standalone(self):
        assert (
            ContextCompressor.classify_summary_content(PARAPHRASED_HANDOFF)
            == "standalone"
        )

    def test_paraphrased_handoff_recognized_after_lstrip(self):
        # classify_summary_content lstrips its input; feed raw whitespace too.
        assert (
            ContextCompressor.classify_summary_content("  \n" + PARAPHRASED_HANDOFF)
            == "standalone"
        )

    def test_shipped_prefix_still_classifies_standalone(self):
        assert (
            ContextCompressor.classify_summary_content(SUMMARY_PREFIX + "\nbody")
            == "standalone"
        )

    def test_legacy_prefix_still_classifies_standalone(self):
        assert (
            ContextCompressor.classify_summary_content(LEGACY_SUMMARY_PREFIX + " body")
            == "standalone"
        )

    def test_historical_prefixes_still_classify_standalone(self):
        for prefix in _HISTORICAL_SUMMARY_PREFIXES:
            assert (
                ContextCompressor.classify_summary_content(prefix + " body")
                == "standalone"
            ), prefix[:60]

    def test_paraphrased_handoff_in_merged_position_classifies_merged(self):
        text = (
            "preserved live-tail content "
            + _MERGED_SUMMARY_DELIMITER
            + "\n"
            + PARAPHRASED_HANDOFF
        )
        assert ContextCompressor.classify_summary_content(text) == "merged"

    def test_marker_without_compaction_vocabulary_is_not_classified(self):
        # The vocabulary window is what keeps a mere quotation of the marker (e.g. a
        # docs-style reply about the marker itself) from being hidden as a handoff.
        text = (
            _HANDOFF_MARKER_PREFIX
            + " Quoting this header to talk about its formatting in general."
        )
        assert ContextCompressor.classify_summary_content(text) is None

    def test_vocabulary_outside_window_is_not_classified(self):
        filler = "x" * _PARAPHRASED_HANDOFF_WINDOW
        text = (
            _HANDOFF_MARKER_PREFIX
            + " "
            + filler
            + " and then the word compacted appears far too late"
        )
        assert ContextCompressor.classify_summary_content(text) is None

    def test_ordinary_assistant_reply_is_not_classified(self):
        assert (
            ContextCompressor.classify_summary_content(
                "Here is the fix you asked for: the queue command now accepts a prompt argument."
            )
            is None
        )
