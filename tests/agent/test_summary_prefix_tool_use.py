"""Regression tests for the SUMMARY_PREFIX tool-use clause (#65848 class).

The REFERENCE ONLY framing must keep its anti-resumption protections while
explicitly NOT restricting tool use — the strong wording was observed bleeding
into general tool-use suppression (narration-only turns after compression).
"""

from agent.context_compressor import (
    _HISTORICAL_SUMMARY_PREFIXES,
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
)


class TestSummaryPrefixToolUseClause:
    def test_prefix_affirms_tools_remain_active(self):
        assert "tools remain fully active" in SUMMARY_PREFIX
        assert "narrating" in SUMMARY_PREFIX

    def test_prefix_keeps_anti_resumption_protections(self):
        """The clause is additive — every load-bearing protection stays."""
        assert "REFERENCE ONLY" in SUMMARY_PREFIX
        assert "Do NOT answer questions or fulfill requests" in SUMMARY_PREFIX
        assert "the latest user message WINS" in SUMMARY_PREFIX
        assert "Reverse signals" in SUMMARY_PREFIX
        assert "ALWAYS authoritative" in SUMMARY_PREFIX

    def test_previous_generation_frozen_in_historical_prefixes(self):
        """Per the module contract: whenever SUMMARY_PREFIX changes, the prior
        generation must be frozen into _HISTORICAL_SUMMARY_PREFIXES so old
        persisted summaries still get the directive-strip on re-compaction."""
        assert len(_HISTORICAL_SUMMARY_PREFIXES) >= 3
        newest_frozen = _HISTORICAL_SUMMARY_PREFIXES[0]
        # The frozen copy is the pre-clause generation: same framing, no clause.
        assert "tools remain fully active" not in newest_frozen
        assert "Do NOT answer questions or fulfill requests" in newest_frozen
        assert newest_frozen != SUMMARY_PREFIX

    def test_historical_prefixes_are_distinct_from_current(self):
        for frozen in _HISTORICAL_SUMMARY_PREFIXES:
            assert frozen != SUMMARY_PREFIX
        assert LEGACY_SUMMARY_PREFIX != SUMMARY_PREFIX

    def test_strip_recognizes_current_and_frozen_prefixes(self):
        """Re-compaction normalization must strip both the live prefix and the
        newly frozen one (the incident generation)."""
        from agent.context_compressor import ContextCompressor

        for prefix in (SUMMARY_PREFIX, _HISTORICAL_SUMMARY_PREFIXES[0]):
            text = f"{prefix}\nsummary body here"
            stripped = ContextCompressor._strip_summary_prefix(text)
            assert "summary body here" in stripped
            assert "REFERENCE ONLY" not in stripped
