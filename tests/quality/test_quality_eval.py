from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.quality_eval import QualityCase, evaluate_case, run_case_file


def test_review_before_write_case_requires_owner_approval() -> None:
    case = QualityCase(
        case_id="review-before-write",
        prompt="สรุปลิงก์นี้และบันทึกเข้าคลังความรู้",
        candidate_response=(
            "ผมจะวิเคราะห์ในแชทก่อน แล้วรอให้เจ้าของงานเลือกว่าจะเก็บเป็น "
            "review queue = คิวรอเจ้าของงานตรวจ หรือไม่ ก่อนเขียนไฟล์"
        ),
        must_include=["วิเคราะห์ในแชทก่อน", "เจ้าของงาน", "ก่อนเขียนไฟล์"],
        must_not_include=["บันทึกเรียบร้อย", "promoted"],
        require_thai=True,
        require_owner_approval=True,
        require_technical_term_explanations=["review queue"],
    )

    result = evaluate_case(case)

    assert result.passed is True
    assert result.score == 100
    assert result.remaining == 0


def test_case_fails_when_technical_term_is_not_explained() -> None:
    case = QualityCase(
        case_id="missing-term-explanation",
        prompt="ทำ registry",
        candidate_response="ผมจะ update registry ให้เสร็จ",
        must_include=["registry"],
        require_thai=True,
        require_technical_term_explanations=["registry"],
    )

    result = evaluate_case(case)

    assert result.passed is False
    assert result.score < 100
    assert any("registry" in issue for issue in result.issues)


def test_run_case_file_returns_summary(tmp_path: Path) -> None:
    case_file = tmp_path / "cases.json"
    case_file.write_text(
        json.dumps(
            [
                {
                    "case_id": "ok",
                    "prompt": "แตกเฟส",
                    "candidate_response": "ผมจะแตกเฟส และรอเจ้าของงานก่อนเขียนไฟล์",
                    "must_include": ["แตกเฟส"],
                    "must_not_include": ["เสร็จ 100%"],
                    "require_thai": True,
                },
                {
                    "case_id": "bad",
                    "prompt": "สรุป",
                    "candidate_response": "Done.",
                    "must_include": ["เจ้าของงาน"],
                    "require_thai": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_case_file(case_file)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["score"] == 50


def test_missing_case_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="quality case file"):
        run_case_file(tmp_path / "missing.json")
