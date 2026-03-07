from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from swarm.types import SwarmResult, SwarmTask


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    weight: float = 1.0


@dataclass
class VerificationResult:
    task_id: str
    score: float  # 0.0 - 1.0
    passed: bool
    feedback: str
    checks: list[CheckResult] = field(default_factory=list)
    verified_by: str = "Verifier"
    timestamp: datetime = field(default_factory=datetime.utcnow)


class VerificationCheck(ABC):
    @abstractmethod
    def check(self, task: SwarmTask, result: SwarmResult) -> CheckResult:
        ...


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------

class NonEmptyOutputCheck(VerificationCheck):
    def check(self, task: SwarmTask, result: SwarmResult) -> CheckResult:
        output = result.output
        if output is None or (isinstance(output, str) and not output.strip()):
            return CheckResult("non_empty_output", False, "Output is empty or None")
        if isinstance(output, (list, dict)) and len(output) == 0:
            return CheckResult("non_empty_output", False, "Output is an empty collection")
        return CheckResult("non_empty_output", True, "Output is non-empty")


_ERROR_PATTERNS = [
    re.compile(r"(?i)\bTraceback\b.*\bcall\b"),
    re.compile(r"(?i)\bError\b:\s"),
    re.compile(r"(?i)\bFATAL\b"),
    re.compile(r"(?i)\bpanic\b:"),
    re.compile(r"(?i)\bSegmentation fault\b"),
    re.compile(r"(?i)\bPermission denied\b"),
]


class NoErrorCheck(VerificationCheck):
    def __init__(self, extra_patterns: list[re.Pattern[str]] | None = None):
        self.patterns = list(_ERROR_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)

    def check(self, task: SwarmTask, result: SwarmResult) -> CheckResult:
        text = str(result.output) if result.output is not None else ""
        for pat in self.patterns:
            m = pat.search(text)
            if m:
                snippet = text[max(0, m.start() - 30):m.end() + 30]
                return CheckResult(
                    "no_error", False,
                    f"Error pattern matched: {pat.pattern!r} near ...{snippet}...",
                )
        return CheckResult("no_error", True, "No error patterns detected")


class FilesExistCheck(VerificationCheck):
    def check(self, task: SwarmTask, result: SwarmResult) -> CheckResult:
        paths = list(result.artifacts) if result.artifacts else []
        # Also pull from task artifacts
        for ref in task.artifacts:
            paths.append(ref.path)
        if not paths:
            return CheckResult("files_exist", True, "No artifact paths to verify")
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            return CheckResult(
                "files_exist", False,
                f"Missing files: {missing}",
            )
        return CheckResult("files_exist", True, f"All {len(paths)} artifact(s) exist")


class SyntaxCheck(VerificationCheck):
    def check(self, task: SwarmTask, result: SwarmResult) -> CheckResult:
        text = result.output if isinstance(result.output, str) else None
        if text is None:
            return CheckResult("syntax", True, "Output is not a string; skipped syntax check")
        # Heuristic: looks like Python code
        if not _looks_like_python(text):
            return CheckResult("syntax", True, "Output does not appear to be Python; skipped")
        try:
            compile(text, "<verifier>", "exec")
        except SyntaxError as exc:
            return CheckResult("syntax", False, f"SyntaxError: {exc}")
        return CheckResult("syntax", True, "Python syntax OK")


def _looks_like_python(text: str) -> bool:
    indicators = ["def ", "class ", "import ", "from ", "if __name__"]
    return any(ind in text for ind in indicators)


class DiffSizeCheck(VerificationCheck):
    def __init__(self, max_lines: int = 2000):
        self.max_lines = max_lines

    def check(self, task: SwarmTask, result: SwarmResult) -> CheckResult:
        text = str(result.output) if result.output is not None else ""
        line_count = text.count("\n") + (1 if text else 0)
        if line_count > self.max_lines:
            return CheckResult(
                "diff_size", False,
                f"Output is {line_count} lines, exceeds limit of {self.max_lines}",
            )
        return CheckResult("diff_size", True, f"Output is {line_count} lines (limit {self.max_lines})")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CHECKS: list[VerificationCheck] = [
    NonEmptyOutputCheck(),
    NoErrorCheck(),
    FilesExistCheck(),
    SyntaxCheck(),
    DiffSizeCheck(),
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    def __init__(
        self,
        checks: list[VerificationCheck] | None = None,
        threshold: float = 0.7,
        name: str = "Verifier",
    ):
        self.checks = checks if checks is not None else list(DEFAULT_CHECKS)
        self.threshold = threshold
        self.name = name

    def verify(self, task: SwarmTask, result: SwarmResult) -> VerificationResult:
        check_results: list[CheckResult] = []
        for vc in self.checks:
            cr = vc.check(task, result)
            check_results.append(cr)

        total_weight = sum(cr.weight for cr in check_results)
        if total_weight == 0:
            score = 1.0
        else:
            score = sum(cr.weight for cr in check_results if cr.passed) / total_weight

        passed = score >= self.threshold
        feedback_lines = []
        for cr in check_results:
            status = "PASS" if cr.passed else "FAIL"
            feedback_lines.append(f"[{status}] {cr.name}: {cr.detail}")
        feedback = "\n".join(feedback_lines)

        return VerificationResult(
            task_id=task.id,
            score=round(score, 4),
            passed=passed,
            feedback=feedback,
            checks=check_results,
            verified_by=self.name,
        )


def format_report(vr: VerificationResult) -> str:
    verdict = "PASSED" if vr.passed else "FAILED"
    lines = [
        f"=== Verification Report ===",
        f"Task:      {vr.task_id}",
        f"Verdict:   {verdict}",
        f"Score:     {vr.score:.2%}",
        f"Verified:  {vr.verified_by} at {vr.timestamp:%Y-%m-%d %H:%M:%S}",
        f"",
        f"--- Checks ({len(vr.checks)}) ---",
    ]
    for cr in vr.checks:
        mark = "+" if cr.passed else "X"
        lines.append(f"  [{mark}] {cr.name} (w={cr.weight:.1f}): {cr.detail}")
    lines.append("")
    if not vr.passed:
        failed = [cr for cr in vr.checks if not cr.passed]
        lines.append(f"Failed checks ({len(failed)}):")
        for cr in failed:
            lines.append(f"  - {cr.name}: {cr.detail}")
    lines.append("=" * 27)
    return "\n".join(lines)
