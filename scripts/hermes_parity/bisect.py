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

from . import gitops

try:  # pragma: no cover - exercised only when the full agent package is importable
    from agent.redact import redact_sensitive_text
except Exception:  # pragma: no cover - parity tooling must also run in tiny fixture repos
    def redact_sensitive_text(text: str, *, force: bool = False) -> str:
        return text


class TestOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSENT = "ABSENT"


class Classification(str, Enum):
    CLEAN = "CLEAN"
    REGRESSION = "MERGE REGRESSION"
    REGRESSION_FLAKY = "MERGE REGRESSION (flaky)"
    REGRESSION_DEP = "MERGE REGRESSION (dep-driven)"
    FIXED_BY_MERGE = "FIXED-BY-MERGE"
    INHERITED = "INHERITED"
    INHERITED_FLAKY = "INHERITED (flaky)"
    NONDETERMINISTIC = "NONDETERMINISTIC"
    UPSTREAM_TEST = "UPSTREAM TEST"
    ORDER_POLLUTION = "ORDER-POLLUTION"
    FLAKY = "FLAKY"


@dataclass(frozen=True)
class PytestRun:
    outcome: TestOutcome
    output: str = ""


@dataclass(frozen=True)
class StableSide:
    outcome: TestOutcome
    stable: bool
    runs: tuple[TestOutcome, ...]
    output: str = ""


@dataclass(frozen=True)
class BisectResult:
    test: str
    baseline: TestOutcome
    merge: TestOutcome
    classification: Classification
    baseline_runs: tuple[TestOutcome, ...] = ()
    merge_runs: tuple[TestOutcome, ...] = ()
    baseline_output: str = ""
    merge_output: str = ""
    dep_output: str = ""

    @property
    def baseline_passed(self) -> bool:
        return self.baseline == TestOutcome.PASS

    @property
    def merge_passed(self) -> bool:
        return self.merge == TestOutcome.PASS


Runner = Callable[[Path, Sequence[str]], TestOutcome | bool | PytestRun]


def normalize_outcome(value: TestOutcome | bool | PytestRun) -> TestOutcome:
    if isinstance(value, PytestRun):
        return value.outcome
    if isinstance(value, TestOutcome):
        return value
    return TestOutcome.PASS if value else TestOutcome.FAIL


def normalize_run(value: TestOutcome | bool | PytestRun) -> PytestRun:
    if isinstance(value, PytestRun):
        return value
    return PytestRun(normalize_outcome(value))


def classify_results(*, baseline: TestOutcome | bool, merge: TestOutcome | bool) -> Classification:
    baseline_outcome = normalize_outcome(baseline)
    merge_outcome = normalize_outcome(merge)
    if baseline_outcome == TestOutcome.ABSENT:
        return Classification.UPSTREAM_TEST
    if baseline_outcome == TestOutcome.PASS and merge_outcome == TestOutcome.FAIL:
        return Classification.REGRESSION
    if baseline_outcome == TestOutcome.FAIL and merge_outcome == TestOutcome.PASS:
        return Classification.FIXED_BY_MERGE
    if baseline_outcome == TestOutcome.FAIL and merge_outcome == TestOutcome.FAIL:
        return Classification.INHERITED
    if baseline_outcome == TestOutcome.PASS and merge_outcome == TestOutcome.PASS:
        return Classification.CLEAN
    return Classification.INHERITED


def parse_pytest_outcome(returncode: int, output: str) -> TestOutcome:
    lowered = output.lower()
    if returncode == 0:
        return TestOutcome.PASS
    if returncode == 4 or "no tests ran" in lowered or "not found:" in lowered:
        return TestOutcome.ABSENT
    return TestOutcome.FAIL


def _python_for_repo(repo: Path) -> Path:
    for rel in (".venv/bin/python", "venv/bin/python"):
        candidate = repo / rel
        if candidate.exists():
            return candidate
    # In this fleet, parity worktrees usually share the parent checkout venv;
    # when requirements/lock files have not changed, that single shared venv is
    # faithful for both sides. A non-empty dependency diff triggers the explicit
    # dual-venv baseline probe in classify_one().
    shared = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"
    return shared if shared.exists() else Path(sys.executable)


def clean_pytest_env(repo: Path, python: Path) -> dict[str, str]:
    allow = ("HOME", "PATH", "TMPDIR", "TEMP", "TMP", "SSH_AUTH_SOCK", "GIT_CONFIG_GLOBAL")
    env = {key: value for key, value in os.environ.items() if key in allow and value}
    env["PYTHONPATH"] = str(repo)
    env["VIRTUAL_ENV"] = str(python.parent.parent)
    return env


def pytest_runner(repo: Path, tests: Sequence[str]) -> PytestRun:
    python = _python_for_repo(repo)
    proc = subprocess.run(
        [str(python), "-m", "pytest", *tests, "-q", "-o", "addopts=", "-p", "no:randomly"],
        cwd=repo,
        env=clean_pytest_env(repo, python),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return PytestRun(parse_pytest_outcome(proc.returncode, proc.stdout), proc.stdout)


def requirements_changed(repo: Path, baseline: str) -> bool:
    proc = gitops.run_git(
        repo,
        ["diff", "--name-only", f"{baseline}..HEAD", "--", "requirements*.txt", "pyproject.toml", "uv.lock"],
        check=False,
    )
    return bool(proc.stdout.strip())


def _stable(repo: Path, test: str, runner: Runner, *, repeats: int = 3) -> StableSide:
    runs: list[TestOutcome] = []
    output = ""
    for _ in range(repeats):
        run = normalize_run(runner(repo, (test,)))
        runs.append(run.outcome)
        if run.output:
            output = run.output
    stable = len(set(runs)) == 1
    return StableSide(runs[-1], stable, tuple(runs), output)


def _classify_stability(baseline: StableSide, merge: StableSide) -> Classification:
    if baseline.stable and merge.stable:
        return classify_results(baseline=baseline.outcome, merge=merge.outcome)
    if baseline.stable and not merge.stable:
        if baseline.outcome == TestOutcome.PASS:
            return Classification.REGRESSION_FLAKY
        if baseline.outcome == TestOutcome.FAIL:
            return Classification.INHERITED_FLAKY
    return Classification.NONDETERMINISTIC


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
    dep_runner: Runner | None = None,
    repeats: int = 3,
) -> BisectResult:
    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(_stable, baseline_repo, test, runner, repeats=repeats)
        merge_future = executor.submit(_stable, merge_repo, test, runner, repeats=repeats)
        baseline = baseline_future.result()
        merge = merge_future.result()
    classification = _classify_stability(baseline, merge)
    dep_output = ""
    if (
        dep_runner is not None
        and classification == Classification.REGRESSION
        and baseline.outcome == TestOutcome.PASS
        and merge.outcome == TestOutcome.FAIL
    ):
        dep_side = _stable(baseline_repo, test, dep_runner, repeats=repeats)
        dep_output = dep_side.output
        if dep_side.stable and dep_side.outcome == TestOutcome.FAIL:
            classification = Classification.REGRESSION_DEP
    return BisectResult(
        test=test,
        baseline=baseline.outcome,
        merge=merge.outcome,
        classification=classification,
        baseline_runs=baseline.runs,
        merge_runs=merge.runs,
        baseline_output=baseline.output,
        merge_output=merge.output,
        dep_output=dep_output,
    )


def classify_many(
    *,
    baseline_repo: Path,
    merge_repo: Path,
    tests: Sequence[str],
    runner: Runner,
    jobs: int = 2,
    dep_runner: Runner | None = None,
    repeats: int = 3,
) -> list[BisectResult]:
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                classify_one,
                baseline_repo=baseline_repo,
                merge_repo=merge_repo,
                test=test,
                runner=runner,
                dep_runner=dep_runner,
                repeats=repeats,
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
    dep_runner: Runner | None = None,
) -> BisectResult:
    joined = " ".join(tests)
    return classify_one(baseline_repo=baseline_repo, merge_repo=merge_repo, test=joined, runner=runner, dep_runner=dep_runner)


def scrub_output(text: str, *, home: Path | None = None, venv: Path | None = None, max_lines: int = 80) -> list[str]:
    lines = text.splitlines()[-max_lines:]
    scrubbed = "\n".join(lines)
    home = home or Path.home()
    scrubbed = scrubbed.replace(str(home), "$HOME")
    if venv:
        scrubbed = scrubbed.replace(str(venv), "$VENV")
    scrubbed = redact_sensitive_text(scrubbed, force=True)
    return scrubbed.splitlines()


def jsonable_result(result: BisectResult, *, max_lines: int = 80) -> dict[str, object]:
    return {
        "test": result.test,
        "baseline": result.baseline.value,
        "merge": result.merge.value,
        "classification": result.classification.value,
        "baseline_runs": [item.value for item in result.baseline_runs],
        "merge_runs": [item.value for item in result.merge_runs],
        "baseline_output_tail": scrub_output(result.baseline_output, max_lines=max_lines),
        "merge_output_tail": scrub_output(result.merge_output, max_lines=max_lines),
        "dep_output_tail": scrub_output(result.dep_output, max_lines=max_lines),
    }


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
