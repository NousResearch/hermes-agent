"""Eval runner — executes cases via AIAgent in isolated workdirs and scores results."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Callable, Optional

from .cases import get_case, get_suite, list_cases
from .executor import AgentResult, run_agent_for_eval
from .scoring import run_check, score_checks
from .types import (
    CaseResult,
    CaseStatus,
    CheckResult,
    EvalCase,
    RunSummary,
)

logger = logging.getLogger(__name__)

# Type alias for the executor function — allows injection for testing.
ExecutorFn = Callable[[str, str, int, Optional[str], int], AgentResult]


def run_case(
    case: EvalCase,
    run_id: str,
    workdir: Optional[str] = None,
    executor: Optional[ExecutorFn] = None,
    model: Optional[str] = None,
) -> CaseResult:
    """Execute a single eval case and return its scored result.

    1. Create/use an isolated workdir.
    2. Run the case's ``setup`` callback (if any).
    3. Invoke the AIAgent (via *executor*) with the case prompt.
    4. Run deterministic checks against the workdir artifacts.

    If *executor* is None the default ``run_agent_for_eval`` is used.
    Pass a custom executor in tests to avoid real model calls.
    """
    _exec = executor or run_agent_for_eval
    managed_dir = workdir is None
    if managed_dir:
        tmp = tempfile.mkdtemp(prefix=f"hermes-eval-{case.id}-")
        workdir = tmp

    start = time.monotonic()
    try:
        # Run setup if defined
        if case.setup:
            case.setup(workdir)

        # Invoke agent
        agent_result = _exec(
            case.prompt,
            workdir,
            case.timeout_seconds,
            model,
            30,  # max_iterations
        )

        # Run deterministic checks
        check_results: list[CheckResult] = []
        for check in case.deterministic_checks:
            check_results.append(run_check(check, workdir))

        score = score_checks(check_results)
        all_passed = all(r.passed for r in check_results)
        status = CaseStatus.PASSED if all_passed else CaseStatus.FAILED

        # If the agent itself errored, mark as ERROR regardless of checks
        if agent_result.error:
            status = CaseStatus.ERROR

        failures = [r for r in check_results if not r.passed]
        failure_summary = "; ".join(r.message for r in failures if r.message) if failures else ""
        if agent_result.error:
            err_prefix = f"agent error: {agent_result.error}"
            failure_summary = f"{err_prefix}; {failure_summary}" if failure_summary else err_prefix

        duration_ms = int((time.monotonic() - start) * 1000)
        return CaseResult(
            run_id=run_id,
            case_id=case.id,
            category=case.category.value,
            status=status,
            deterministic_score=score,
            total_score=score,
            duration_ms=duration_ms,
            failure_summary=failure_summary,
            check_results=check_results,
            raw_result=agent_result.raw,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return CaseResult(
            run_id=run_id,
            case_id=case.id,
            category=case.category.value,
            status=CaseStatus.ERROR,
            duration_ms=duration_ms,
            failure_summary=str(exc),
        )


def run_suite(
    suite_name: str,
    label: str = "",
    run_id: Optional[str] = None,
    executor: Optional[ExecutorFn] = None,
    model: Optional[str] = None,
) -> RunSummary:
    """Run all cases in a named suite and return an aggregated summary."""
    cases = get_suite(suite_name)
    return _run_cases(list(cases), suite_name, label, run_id, executor, model)


def run_single(
    case_id: str,
    label: str = "",
    run_id: Optional[str] = None,
    executor: Optional[ExecutorFn] = None,
    model: Optional[str] = None,
) -> RunSummary:
    """Run a single case by ID and return a summary."""
    case = get_case(case_id)
    return _run_cases([case], f"single:{case_id}", label, run_id, executor, model)


def _run_cases(
    cases: list[EvalCase],
    suite_name: str,
    label: str,
    run_id: Optional[str],
    executor: Optional[ExecutorFn],
    model: Optional[str],
) -> RunSummary:
    if run_id is None:
        run_id = RunSummary.new_run_id()

    results: list[CaseResult] = []
    for case in cases:
        result = run_case(case, run_id, executor=executor, model=model)
        results.append(result)

    passed = sum(1 for r in results if r.status == CaseStatus.PASSED)
    failed = len(results) - passed
    avg = sum(r.total_score for r in results) / len(results) if results else 0.0

    return RunSummary(
        run_id=run_id,
        suite_name=suite_name,
        label=label,
        case_count=len(results),
        passed_count=passed,
        failed_count=failed,
        avg_score=round(avg, 4),
        case_results=results,
    )
