"""Baseline/merge classification helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence, TextIO


class TestOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSENT = "ABSENT"


class Classification(str, Enum):
    REGRESSION = "MERGE REGRESSION"
    INHERITED_FLAKY = "INHERITED/FLAKY"
    UPSTREAM_TEST = "UPSTREAM TEST"
    ORDER_POLLUTION = "ORDER-POLLUTION"
    FLAKY = "FLAKY"


@dataclass(frozen=True)
class BisectResult:
    test: str
    baseline: TestOutcome
    merge: TestOutcome
    classification: Classification

    @property
    def baseline_passed(self) -> bool:
        return self.baseline == TestOutcome.PASS

    @property
    def merge_passed(self) -> bool:
        return self.merge == TestOutcome.PASS


Runner = Callable[[Path, Sequence[str]], TestOutcome | bool]


def normalize_outcome(value: TestOutcome | bool) -> TestOutcome:
    if isinstance(value, TestOutcome):
        return value
    return TestOutcome.PASS if value else TestOutcome.FAIL


def classify_results(*, baseline: TestOutcome | bool, merge: TestOutcome | bool) -> Classification:
    baseline_outcome = normalize_outcome(baseline)
    merge_outcome = normalize_outcome(merge)
    if baseline_outcome == TestOutcome.ABSENT:
        return Classification.UPSTREAM_TEST
    if baseline_outcome == TestOutcome.PASS and merge_outcome == TestOutcome.FAIL:
        return Classification.REGRESSION
    if baseline_outcome == TestOutcome.FAIL and merge_outcome == TestOutcome.FAIL:
        return Classification.INHERITED_FLAKY
    if baseline_outcome == TestOutcome.PASS and merge_outcome == TestOutcome.PASS:
        return Classification.ORDER_POLLUTION
    return Classification.INHERITED_FLAKY


def parse_pytest_outcome(returncode: int, output: str) -> TestOutcome:
    lowered = output.lower()
    if returncode == 0:
        return TestOutcome.PASS
    if returncode == 4 or "no tests ran" in lowered or "not found:" in lowered:
        return TestOutcome.ABSENT
    return TestOutcome.FAIL


def pytest_runner(repo: Path, tests: Sequence[str]) -> TestOutcome:
    script = repo / "scripts" / "run_tests.sh"
    proc = subprocess.run(
        [str(script), *tests],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return parse_pytest_outcome(proc.returncode, proc.stdout)


def bounded_jobs(requested: int | None, *, cpu_count: int | None = None) -> int:
    half_cpu = max(1, (cpu_count if cpu_count is not None else (os.cpu_count() or 2)) // 2)
    if requested is None:
        requested = 2
    return max(1, min(requested, half_cpu))


def parse_from_file(path: str, *, stdin: TextIO | None = None) -> list[str]:
    if path == "-":
        text = (stdin or sys.stdin).read()
    else:
        text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def classify_one(
    *,
    baseline_repo: Path,
    merge_repo: Path,
    test: str,
    runner: Runner,
) -> BisectResult:
    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(runner, baseline_repo, (test,))
        merge_future = executor.submit(runner, merge_repo, (test,))
        baseline = normalize_outcome(baseline_future.result())
        merge = normalize_outcome(merge_future.result())
    classification = classify_results(baseline=baseline, merge=merge)
    if classification == Classification.REGRESSION:
        rerun_merge = normalize_outcome(runner(merge_repo, (test,)))
        if rerun_merge == TestOutcome.PASS:
            classification = Classification.FLAKY
            merge = rerun_merge
    return BisectResult(test=test, baseline=baseline, merge=merge, classification=classification)


def classify_many(
    *,
    baseline_repo: Path,
    merge_repo: Path,
    tests: Sequence[str],
    runner: Runner,
    jobs: int = 2,
) -> list[BisectResult]:
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                classify_one,
                baseline_repo=baseline_repo,
                merge_repo=merge_repo,
                test=test,
                runner=runner,
            )
            for test in tests
        ]
        return [future.result() for future in futures]


def classify_baseline(
    *,
    baseline_repo: Path,
    merge_repo: Path,
    tests: Sequence[str],
    runner: Runner,
) -> BisectResult:
    joined = " ".join(tests)
    return classify_one(baseline_repo=baseline_repo, merge_repo=merge_repo, test=joined, runner=runner)


def format_table(results: Sequence[BisectResult]) -> str:
    rows = [("test", "baseline", "merge", "classification")]
    for result in results:
        rows.append((result.test, result.baseline.value, result.merge.value, result.classification.value))
    widths = [max(len(row[index]) for row in rows) for index in range(4)]
    lines: list[str] = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)
