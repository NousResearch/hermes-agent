"""Tests for the CompactionResult metrics dataclass."""

from agent.compaction_result import CompactionResult


def test_compaction_result_records_basic_fields():
    r = CompactionResult(
        original_messages=50,
        compacted_messages=12,
        original_tokens=80_000,
        compacted_tokens=8_000,
        operations_deduped=4,
        triggered_by="token",
    )
    assert r.original_messages == 50
    assert r.compacted_messages == 12
    assert r.token_reduction_pct == 90.0
    assert r.message_reduction_pct == 76.0


def test_compaction_result_handles_zero_original():
    r = CompactionResult(
        original_messages=0,
        compacted_messages=0,
        original_tokens=0,
        compacted_tokens=0,
        operations_deduped=0,
        triggered_by="token",
    )
    assert r.token_reduction_pct == 0.0
    assert r.message_reduction_pct == 0.0


def test_compaction_result_summary_line():
    r = CompactionResult(
        original_messages=50,
        compacted_messages=12,
        original_tokens=80_000,
        compacted_tokens=8_000,
        operations_deduped=4,
        triggered_by="token",
    )
    line = r.summary_line()
    assert "80,000 → 8,000" in line
    assert "90%" in line
    assert "deduped 4" in line
    assert "trigger=token" in line
