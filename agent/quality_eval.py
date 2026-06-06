"""Deterministic quality checks for Hermes prompt and knowledge workflows.

This module is intentionally dependency-light. LLM-as-judge tools such as
DeepEval can be layered on top later, but these checks cover the owner rules
that must never depend on a model call: Thai-first responses, review before
write, no unsupported completion claims, and plain-language explanations for
technical terms.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_EXPLANATION_SEPARATORS = ("=", "คือ", "หมายถึง", "แปลว่า")
_OWNER_APPROVAL_HINTS = (
    "เจ้าของงาน",
    "รอ",
    "อนุมัติ",
    "เลือก",
    "ก่อนเขียน",
    "ก่อนบันทึก",
)


@dataclass
class QualityCase:
    case_id: str
    prompt: str
    candidate_response: str
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    require_thai: bool = False
    require_owner_approval: bool = False
    require_technical_term_explanations: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "QualityCase":
        return cls(
            case_id=str(data["case_id"]),
            prompt=str(data.get("prompt", "")),
            candidate_response=str(data.get("candidate_response", "")),
            must_include=[str(x) for x in data.get("must_include", [])],
            must_not_include=[str(x) for x in data.get("must_not_include", [])],
            require_thai=bool(data.get("require_thai", False)),
            require_owner_approval=bool(data.get("require_owner_approval", False)),
            require_technical_term_explanations=[
                str(x) for x in data.get("require_technical_term_explanations", [])
            ],
        )


@dataclass
class QualityResult:
    case_id: str
    passed: bool
    score: int
    remaining: int
    issues: list[str] = field(default_factory=list)


def _contains_thai(text: str) -> bool:
    return bool(_THAI_RE.search(text or ""))


def _term_is_explained(text: str, term: str) -> bool:
    lowered = text.lower()
    needle = term.lower()
    index = lowered.find(needle)
    if index < 0:
        return False
    window = lowered[index : index + max(len(term) + 80, 100)]
    return any(sep in window for sep in _EXPLANATION_SEPARATORS)


def evaluate_case(case: QualityCase) -> QualityResult:
    text = case.candidate_response or ""
    issues: list[str] = []

    for token in case.must_include:
        if token not in text:
            issues.append(f"missing required text: {token}")

    for token in case.must_not_include:
        if token in text:
            issues.append(f"forbidden text present: {token}")

    if case.require_thai and not _contains_thai(text):
        issues.append("response must contain Thai text")

    if case.require_owner_approval:
        if not all(hint in text for hint in ("เจ้าของงาน", "ก่อน")):
            issues.append("response must state owner review before writing")
        if not any(hint in text for hint in _OWNER_APPROVAL_HINTS):
            issues.append("response must include an approval/waiting signal")

    for term in case.require_technical_term_explanations:
        if not _term_is_explained(text, term):
            issues.append(f"technical term is not explained plainly: {term}")

    total_checks = (
        len(case.must_include)
        + len(case.must_not_include)
        + int(case.require_thai)
        + int(case.require_owner_approval)
        + len(case.require_technical_term_explanations)
    )
    total_checks = max(total_checks, 1)
    failed = min(len(issues), total_checks)
    score = int(round(((total_checks - failed) / total_checks) * 100))
    return QualityResult(
        case_id=case.case_id,
        passed=not issues,
        score=score,
        remaining=100 - score,
        issues=issues,
    )


def load_cases(path: Path) -> list[QualityCase]:
    if not path.exists():
        raise FileNotFoundError(f"quality case file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("quality case file must contain a JSON list")
    return [QualityCase.from_mapping(item) for item in raw]


def run_case_file(path: Path) -> dict[str, Any]:
    cases = load_cases(path)
    results = [evaluate_case(case) for case in cases]
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    score = int(round((passed / total) * 100)) if total else 100
    return {
        "ok": passed == total,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "score": score,
        "remaining": 100 - score,
        "results": [asdict(result) for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Hermes quality eval cases")
    parser.add_argument("case_file", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args(argv)

    summary = run_case_file(args.case_file)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"QUALITY_EVAL {summary['score']} {summary['remaining']}")
        for result in summary["results"]:
            print(
                f"{result['case_id']} {result['score']} {result['remaining']} "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )
            for issue in result["issues"]:
                print(f"  - {issue}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
