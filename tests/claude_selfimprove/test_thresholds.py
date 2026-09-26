from __future__ import annotations

from claude_selfimprove import thresholds


def _row(**overrides):
    row = {
        "category": "repeated_fix",
        "target_kind": "rule",
        "confidence": 0.9,
        "session_ids": ["s1", "s2", "s3"],
        "occurrence_count": 2,
    }
    row.update(overrides)
    return row


def test_explicit_instruction_qualifies_from_one_occurrence():
    row = _row(category="explicit_instruction", session_ids=["s1"], occurrence_count=1)
    eligible, _ = thresholds.is_eligible(row)
    assert eligible is True


def test_explicit_instruction_with_zero_occurrences_is_not_eligible():
    row = _row(category="explicit_instruction", session_ids=[], occurrence_count=0)
    eligible, _ = thresholds.is_eligible(row)
    assert eligible is False


def test_rule_needs_three_sessions_and_two_tasks():
    row = _row(category="repeated_fix", target_kind="rule", session_ids=["s1", "s2"], occurrence_count=2)
    eligible, reason = thresholds.is_eligible(row)
    assert eligible is False
    assert "sessions" in reason

    row2 = _row(category="repeated_fix", target_kind="rule", session_ids=["s1", "s2", "s3"], occurrence_count=1)
    eligible2, _ = thresholds.is_eligible(row2)
    assert eligible2 is False  # enough sessions, not enough tasks

    row3 = _row(category="repeated_fix", target_kind="rule", session_ids=["s1", "s2", "s3"], occurrence_count=2)
    eligible3, _ = thresholds.is_eligible(row3)
    assert eligible3 is True


def test_claude_md_block_target_uses_rule_threshold():
    row = _row(category="repeated_procedure", target_kind="claude_md_block", session_ids=["s1", "s2", "s3"], occurrence_count=2)
    eligible, _ = thresholds.is_eligible(row)
    assert eligible is True


def test_skill_needs_three_confirmed_uses():
    row = _row(category="repeated_procedure", target_kind="skill", session_ids=["s1"], occurrence_count=2)
    eligible, _ = thresholds.is_eligible(row)
    assert eligible is False

    row2 = _row(category="repeated_procedure", target_kind="skill", session_ids=["s1"], occurrence_count=3)
    eligible2, _ = thresholds.is_eligible(row2)
    assert eligible2 is True  # note: sessions don't matter for skills, only uses


def test_low_confidence_blocks_regardless_of_evidence():
    row = _row(category="explicit_instruction", confidence=0.2, occurrence_count=10, session_ids=["s1"] * 10)
    eligible, reason = thresholds.is_eligible(row)
    assert eligible is False
    assert "confidence" in reason


def test_confidence_exactly_at_floor_passes():
    row = _row(category="explicit_instruction", confidence=thresholds.MIN_CONFIDENCE, occurrence_count=1)
    eligible, _ = thresholds.is_eligible(row)
    assert eligible is True
