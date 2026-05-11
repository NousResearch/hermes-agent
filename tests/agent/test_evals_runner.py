"""Tests for the eval runner."""

import os
import json
import pytest

from agent.evals.types import (
    CaseCategory,
    CaseStatus,
    CheckType,
    DeterministicCheck,
    EvalCase,
    RunSummary,
)
from agent.evals.runner import run_case, run_suite, run_single
from agent.evals.executor import AgentResult
from agent.evals.cases import get_suite, get_case, list_suites, list_cases


# ---------------------------------------------------------------------------
# Mock executor helpers
# ---------------------------------------------------------------------------

def _noop_executor(prompt, workdir, timeout, model, max_iter):
    """Executor that does nothing — simulates an agent that produces no artifacts."""
    return AgentResult(response_text="done", iterations=1, raw={"mock": True})


def _file_creating_executor(prompt, workdir, timeout, model, max_iter):
    """Executor that creates output.txt with the expected content."""
    with open(os.path.join(workdir, "output.txt"), "w") as f:
        f.write("hermes-eval-ok")
    return AgentResult(response_text="created file", iterations=2, raw={"mock": True})


def _patching_executor(prompt, workdir, timeout, model, max_iter):
    """Executor that patches greeting.txt as the eval case expects."""
    path = os.path.join(workdir, "greeting.txt")
    with open(path) as f:
        content = f.read()
    content = content.replace("World", "Hermes")
    with open(path, "w") as f:
        f.write(content)
    return AgentResult(response_text="patched", iterations=3, raw={"mock": True})


def _error_executor(prompt, workdir, timeout, model, max_iter):
    """Executor that returns an error."""
    return AgentResult(error="model unavailable", raw={"mock": True})


# ---------------------------------------------------------------------------
# Test: executor is invoked
# ---------------------------------------------------------------------------

class TestRunCaseWithExecutor:
    """Tests that verify the runner invokes the executor."""

    def test_executor_is_called(self, tmp_path):
        """Runner must call the executor function with the case prompt."""
        calls = []

        def tracking_executor(prompt, workdir, timeout, model, max_iter):
            calls.append({"prompt": prompt, "workdir": workdir, "timeout": timeout})
            return AgentResult(response_text="ok", iterations=1, raw={})

        case = EvalCase(
            id="test-track",
            name="Tracking test",
            category=CaseCategory.FILE_WORKSPACE,
            prompt="Do something specific",
            deterministic_checks=(
                DeterministicCheck(CheckType.FILE_NOT_EXISTS, "anything.txt"),
            ),
        )
        result = run_case(case, "run-t1", workdir=str(tmp_path), executor=tracking_executor)
        assert len(calls) == 1
        assert calls[0]["prompt"] == "Do something specific"
        assert calls[0]["workdir"] == str(tmp_path)

    def test_passing_case_with_executor(self, tmp_path):
        """Executor creates the expected file -> case passes."""
        case = EvalCase(
            id="test-pass",
            name="Test pass",
            category=CaseCategory.FILE_WORKSPACE,
            prompt="Create output.txt",
            deterministic_checks=(
                DeterministicCheck(CheckType.FILE_EXISTS, "output.txt"),
                DeterministicCheck(CheckType.CONTENT_EQUALS, "output.txt", "hermes-eval-ok"),
            ),
        )
        result = run_case(case, "run-001", workdir=str(tmp_path), executor=_file_creating_executor)
        assert result.status == CaseStatus.PASSED
        assert result.deterministic_score == 1.0
        assert result.raw_result.get("mock") is True

    def test_failing_case_noop_executor(self, tmp_path):
        """Noop executor produces no artifacts -> checks fail."""
        case = EvalCase(
            id="test-fail",
            name="Test fail",
            category=CaseCategory.FILE_WORKSPACE,
            prompt="Create output.txt",
            deterministic_checks=(
                DeterministicCheck(CheckType.FILE_EXISTS, "output.txt"),
            ),
        )
        result = run_case(case, "run-002", workdir=str(tmp_path), executor=_noop_executor)
        assert result.status == CaseStatus.FAILED
        assert result.deterministic_score == 0.0

    def test_error_executor_marks_error(self, tmp_path):
        """When executor returns an error, status is ERROR."""
        case = EvalCase(
            id="test-err",
            name="Test error",
            category=CaseCategory.FILE_WORKSPACE,
            prompt="Create output.txt",
            deterministic_checks=(
                DeterministicCheck(CheckType.FILE_NOT_EXISTS, "anything.txt"),
            ),
        )
        result = run_case(case, "run-003", workdir=str(tmp_path), executor=_error_executor)
        assert result.status == CaseStatus.ERROR
        assert "model unavailable" in result.failure_summary

    def test_setup_runs_before_executor(self, tmp_path):
        """Setup callback runs before the executor; executor sees setup artifacts."""
        order = []

        def setup(wd):
            order.append("setup")
            with open(os.path.join(wd, "setup.txt"), "w") as f:
                f.write("ready")

        def ordered_executor(prompt, workdir, timeout, model, max_iter):
            order.append("executor")
            assert os.path.exists(os.path.join(workdir, "setup.txt"))
            return AgentResult(response_text="ok", iterations=1, raw={})

        case = EvalCase(
            id="test-order",
            name="Order test",
            category=CaseCategory.FILE_WORKSPACE,
            prompt="Check",
            setup=setup,
            deterministic_checks=(
                DeterministicCheck(CheckType.FILE_EXISTS, "setup.txt"),
            ),
        )
        run_case(case, "run-004", workdir=str(tmp_path), executor=ordered_executor)
        assert order == ["setup", "executor"]

    def test_patch_case_with_executor(self, tmp_path):
        """Executor patches greeting.txt -> checks pass."""
        from agent.evals.cases import _setup_file_for_patching
        case = EvalCase(
            id="test-patch",
            name="Patch test",
            category=CaseCategory.FILE_WORKSPACE,
            prompt="Replace World with Hermes",
            setup=_setup_file_for_patching,
            deterministic_checks=(
                DeterministicCheck(CheckType.CONTENT_CONTAINS, "greeting.txt", "Hello, Hermes!"),
                DeterministicCheck(CheckType.CONTENT_NOT_CONTAINS, "greeting.txt", "World"),
            ),
        )
        result = run_case(case, "run-005", workdir=str(tmp_path), executor=_patching_executor)
        assert result.status == CaseStatus.PASSED

    def test_managed_workdir(self):
        """When no workdir given, runner creates a temp dir."""
        case = EvalCase(
            id="test-managed",
            name="Test managed dir",
            category=CaseCategory.RELIABILITY,
            prompt="No-op",
            deterministic_checks=(
                DeterministicCheck(CheckType.FILE_NOT_EXISTS, "anything.txt"),
            ),
        )
        result = run_case(case, "run-006", executor=_noop_executor)
        assert result.status == CaseStatus.PASSED


# ---------------------------------------------------------------------------
# Test: case/suite registry
# ---------------------------------------------------------------------------

class TestCaseRegistry:
    def test_smoke_suite_exists(self):
        assert "smoke" in list_suites()

    def test_smoke_suite_has_6_cases(self):
        cases = get_suite("smoke")
        assert len(cases) == 6

    def test_get_case_by_id(self):
        case = get_case("file-create-and-read")
        assert case.id == "file-create-and-read"
        assert case.category == CaseCategory.FILE_WORKSPACE

    def test_get_case_missing_raises(self):
        with pytest.raises(KeyError):
            get_case("nonexistent-case")

    def test_get_suite_missing_raises(self):
        with pytest.raises(KeyError):
            get_suite("nonexistent-suite")

    def test_all_cases_have_checks(self):
        for case in list_cases():
            assert len(case.deterministic_checks) > 0, f"{case.id} has no checks"


# ---------------------------------------------------------------------------
# Test: suite-level execution with mock executor
# ---------------------------------------------------------------------------

class TestRunSuite:
    def test_run_suite_returns_summary(self):
        summary = run_suite("smoke", executor=_noop_executor)
        assert isinstance(summary, RunSummary)
        assert summary.suite_name == "smoke"
        assert summary.case_count == 6
        assert summary.run_id
        assert summary.passed_count + summary.failed_count == summary.case_count

    def test_run_single(self):
        summary = run_single("file-create-and-read", executor=_noop_executor)
        assert summary.case_count == 1
        assert summary.suite_name == "single:file-create-and-read"

    def test_custom_run_id(self):
        summary = run_suite("smoke", run_id="custom-123", executor=_noop_executor)
        assert summary.run_id == "custom-123"
        for cr in summary.case_results:
            assert cr.run_id == "custom-123"

    def test_executor_passed_to_all_cases(self):
        """Verify each case in the suite receives the executor."""
        calls = []

        def counting_executor(prompt, workdir, timeout, model, max_iter):
            calls.append(prompt)
            return AgentResult(response_text="ok", iterations=1, raw={})

        summary = run_suite("smoke", executor=counting_executor)
        assert len(calls) == 6  # one per smoke case
