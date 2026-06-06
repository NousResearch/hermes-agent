#!/usr/bin/env python3
"""
Tests for executors/review_agent.py — ReviewAgent and QAAgent prompt builders
and parsers.

Scope:
  - ReviewAgent.build_prompt: required, with context, with diff truncation,
    with main_run_prompt_snapshot (which is sliced to user-part via
    "--- End Context ---" separator)
  - ReviewAgent.parse_findings: JSON array, individual objects, malformed,
    empty input graceful path
  - ReviewAgent.recommend_executor: priority order, fallback to hermes-local
  - ReviewAgent.build_report: severity counts, duration calculation
  - QAAgent.build_prompt: with and without test_commands
  - QAAgent.parse_results: JSON object, heuristic ("N passed/failed/skipped"),
    no patterns
  - QAAgent.recommend_executor
  - Convenience factories create_review_agent / create_qa_agent

Strictly no subprocess, no model invocation, no real files, no DB.
"""
from __future__ import annotations

import datetime
import json

import pytest

from executors import review_agent
from executors.review_agent import (
    QAAgent,
    ReviewAgent,
    create_qa_agent,
    create_review_agent,
)
from executors.types import (
    FindingCategory,
    ProjectContext,
    QAReport,
    QAStatus,
    ReviewFinding,
    ReviewReport,
    ReviewStatus,
    Severity,
)


# ---------------------------------------------------------------------------
# 1. ReviewAgent.build_prompt
# ---------------------------------------------------------------------------


class TestReviewBuildPrompt:
    def test_minimal_inputs(self) -> None:
        agent = ReviewAgent()
        prompt = agent.build_prompt(
            task_goal="Refactor auth",
            main_run_executor="claude-code",
            changed_files=[],
            diff="",
        )
        assert "--- Review Context ---" in prompt
        assert "Task Goal: Refactor auth" in prompt
        assert "Main Run Executor: claude-code" in prompt
        assert "--- End Review Context ---" in prompt
        # No changed files block, no diff block
        assert "Changed Files:" not in prompt
        assert "Diff:" not in prompt

    def test_with_changed_files_and_diff(self) -> None:
        agent = ReviewAgent()
        prompt = agent.build_prompt(
            task_goal="add tests",
            main_run_executor="opencode",
            changed_files=["a.py", "b/test_b.py"],
            diff="+x = 1\n-y = 2\n",
        )
        assert "Changed Files:" in prompt
        assert "  a.py" in prompt
        assert "  b/test_b.py" in prompt
        assert "Diff:" in prompt
        assert "+x = 1" in prompt

    def test_with_context(self) -> None:
        agent = ReviewAgent()
        ctx = ProjectContext(
            architecture_notes="Service-based architecture",
            coding_conventions="PEP 8",
        )
        prompt = agent.build_prompt(
            task_goal="g",
            main_run_executor="e",
            changed_files=[],
            diff="",
            context=ctx,
        )
        assert "Architecture: Service-based architecture" in prompt
        assert "Coding Conventions: PEP 8" in prompt

    def test_with_main_run_prompt_snapshot_sliced(self) -> None:
        agent = ReviewAgent()
        snapshot = (
            "--- System Context ---\nsecret stuff\n--- End Context ---\n"
            "User wants to refactor login.\n"
        )
        prompt = agent.build_prompt(
            task_goal="g",
            main_run_executor="e",
            changed_files=[],
            diff="",
            main_run_prompt_snapshot=snapshot,
        )
        # The "--- End Context ---" separator slices out system context;
        # only the user part after the separator should appear.
        assert "User wants to refactor login." in prompt
        assert "secret stuff" not in prompt

    def test_diff_truncation(self) -> None:
        agent = ReviewAgent()
        big_diff = "\n".join([f"+line {i}" for i in range(3000)])
        prompt = agent.build_prompt(
            task_goal="g",
            main_run_executor="e",
            changed_files=[],
            diff=big_diff,
        )
        assert "more lines truncated" in prompt
        # First few lines should still be present
        assert "+line 0" in prompt
        # Should NOT contain the last line (truncated)
        assert "+line 2999" not in prompt

    def test_contains_review_instructions(self) -> None:
        agent = ReviewAgent()
        prompt = agent.build_prompt("g", "e", [], "")
        assert "severity" in prompt
        assert "category" in prompt
        assert "JSON array" in prompt
        assert "Do NOT modify any code" in prompt


# ---------------------------------------------------------------------------
# 2. ReviewAgent.parse_findings
# ---------------------------------------------------------------------------


class TestReviewParseFindings:
    def test_json_array_well_formed(self) -> None:
        agent = ReviewAgent()
        out = '[{"severity": "high", "category": "security", "title": "XSS", "description": "unescaped input"}]'
        findings, err = agent.parse_findings("run-1", out)
        assert err is None
        assert len(findings) == 1
        assert findings[0].severity == Severity.HIGH
        assert findings[0].category == FindingCategory.SECURITY
        assert "XSS" in findings[0].title

    def test_json_array_mixed_with_prose(self) -> None:
        agent = ReviewAgent()
        out = (
            "Here is my review:\n"
            '[{"severity": "low", "category": "style", "title": "naming", "description": "use snake_case"}]\n'
            "Hope that helps."
        )
        findings, err = agent.parse_findings("run-1", out)
        assert err is None
        assert len(findings) == 1
        assert findings[0].severity == Severity.LOW

    def test_malformed_json_falls_back_to_object_match(self) -> None:
        agent = ReviewAgent()
        # No array but contains a single JSON object with severity
        out = 'found: {"severity": "medium", "category": "maintainability", "title": "duplication"}'
        findings, err = agent.parse_findings("run-1", out)
        assert err is None
        assert len(findings) == 1
        assert findings[0].severity == Severity.MEDIUM

    def test_unknown_severity_defaults_to_medium(self) -> None:
        agent = ReviewAgent()
        out = '[{"severity": "apocalyptic", "category": "style", "title": "x"}]'
        findings, _ = agent.parse_findings("run-1", out)
        assert findings[0].severity == Severity.MEDIUM

    def test_unknown_category_defaults_to_maintainability(self) -> None:
        agent = ReviewAgent()
        out = '[{"severity": "low", "category": "smell", "title": "x"}]'
        findings, _ = agent.parse_findings("run-1", out)
        assert findings[0].category == FindingCategory.MAINTAINABILITY

    def test_empty_output_returns_empty_list_with_error(self) -> None:
        agent = ReviewAgent()
        findings, err = agent.parse_findings("run-1", "")
        assert findings == []
        assert err is not None
        assert "No findings" in err

    def test_unstructured_output_becomes_info_finding(self) -> None:
        agent = ReviewAgent()
        out = "this is unstructured prose with no JSON"
        findings, err = agent.parse_findings("run-1", out)
        # Non-empty non-JSON output produces a single info finding
        # titled "Unstructured review output", with the error string.
        assert len(findings) == 1
        assert findings[0].severity == Severity.INFO
        assert "Unstructured" in findings[0].title
        assert err is not None
        assert "Could not parse" in err

    def test_unstructured_with_content(self) -> None:
        agent = ReviewAgent()
        out = "Some prose that contains a JSON-like {brace} but is not valid JSON."
        findings, err = agent.parse_findings("run-1", out)
        # The regex `\{[^{}]*\}` will not match nested braces; the
        # initial JSON array regex also won't match. Since output is
        # non-empty, the function should produce an info finding.
        # (It may also match the single-object regex; either way it
        # produces >= 1 finding.)
        assert len(findings) >= 1
        # If we did get an info finding, the title says so.
        if len(findings) == 1:
            assert "Unstructured" in findings[0].title or err is not None


# ---------------------------------------------------------------------------
# 3. ReviewAgent.recommend_executor
# ---------------------------------------------------------------------------


class TestReviewRecommendExecutor:
    def test_priority_order(self) -> None:
        agent = ReviewAgent()
        eid, reason = agent.recommend_executor(["hermes-local", "opencode", "claude-code"])
        assert eid == "claude-code"
        assert "Claude Code" in reason

    def test_opencode_second(self) -> None:
        agent = ReviewAgent()
        eid, _ = agent.recommend_executor(["opencode"])
        assert eid == "opencode"

    def test_hermes_local_third(self) -> None:
        agent = ReviewAgent()
        eid, _ = agent.recommend_executor(["hermes-local"])
        assert eid == "hermes-local"

    def test_fallback_when_none_available(self) -> None:
        agent = ReviewAgent()
        eid, reason = agent.recommend_executor([])
        assert eid == "hermes-local"
        # Reason text uses "falling back" (two words), not "fallback"
        assert "falling back" in reason.lower()
        assert "hermes-local" in reason.lower()


# ---------------------------------------------------------------------------
# 4. ReviewAgent.build_report
# ---------------------------------------------------------------------------


class TestReviewBuildReport:
    def test_severity_counts(self) -> None:
        agent = ReviewAgent()
        findings = [
            ReviewFinding(id="1", run_id="r", severity=Severity.CRITICAL, title="c"),
            ReviewFinding(id="2", run_id="r", severity=Severity.HIGH, title="h"),
            ReviewFinding(id="3", run_id="r", severity=Severity.HIGH, title="h2"),
            ReviewFinding(id="4", run_id="r", severity=Severity.LOW, title="l"),
        ]
        started = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        completed = started + datetime.timedelta(seconds=42)
        report = agent.build_report(
            review_run_id="rv-1",
            executor="claude-code",
            findings=findings,
            status=ReviewStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
        )
        assert isinstance(report, ReviewReport)
        assert report.total_findings == 4
        assert report.critical_count == 1
        assert report.high_count == 2
        assert report.low_count == 1
        assert report.medium_count == 0
        assert report.info_count == 0
        assert report.duration_seconds == 42.0

    def test_zero_duration_when_no_timestamps(self) -> None:
        agent = ReviewAgent()
        report = agent.build_report("r", "e", [], ReviewStatus.PASSED)
        assert report.duration_seconds == 0.0

    def test_error_propagated(self) -> None:
        agent = ReviewAgent()
        report = agent.build_report(
            "r", "e", [], ReviewStatus.FAILED, error="boom"
        )
        assert report.error == "boom"


# ---------------------------------------------------------------------------
# 5. QAAgent.build_prompt
# ---------------------------------------------------------------------------


class TestQABuildPrompt:
    def test_no_test_commands(self) -> None:
        agent = QAAgent()
        prompt = agent.build_prompt(
            task_goal="verify",
            changed_files=["a.py"],
            test_commands=[],
        )
        assert "--- QA Context ---" in prompt
        assert "Task Goal: verify" in prompt
        assert "No test commands configured" in prompt
        assert "Execute the following test commands" not in prompt

    def test_with_test_commands(self) -> None:
        agent = QAAgent()
        prompt = agent.build_prompt(
            task_goal="verify",
            changed_files=["a.py", "b.py"],
            test_commands=[("unit", "pytest -q"), ("lint", "ruff check .")],
            worktree_path="/tmp/wt",
        )
        assert "Test Commands:" in prompt
        assert "  unit: pytest -q" in prompt
        assert "  lint: ruff check ." in prompt
        assert "Worktree Path: /tmp/wt" in prompt
        assert "  $ pytest -q" in prompt
        assert "  $ ruff check ." in prompt

    def test_contains_qa_instructions(self) -> None:
        agent = QAAgent()
        prompt = agent.build_prompt("g", [], [("unit", "pytest -q")])
        assert "test_passed" in prompt
        assert "test_failed" in prompt
        assert "Do NOT modify any code" in prompt


# ---------------------------------------------------------------------------
# 6. QAAgent.parse_results
# ---------------------------------------------------------------------------


class TestQAParseResults:
    def test_json_object_well_formed(self) -> None:
        agent = QAAgent()
        out = json.dumps({
            "test_passed": 10,
            "test_failed": 1,
            "test_skipped": 0,
            "failed_test_details": "1 failed: test_x",
            "risks": [
                {"severity": "low", "title": "minor", "description": "x", "affected_areas": ["a"]}
            ],
            "coverage_delta": 0.5,
        })
        report, err = agent.parse_results("qa-1", out)
        assert err is None
        assert isinstance(report, QAReport)
        assert report.test_passed == 10
        assert report.test_failed == 1
        assert report.test_skipped == 0
        assert report.coverage_delta == 0.5
        assert len(report.risks) == 1
        assert report.risks[0].severity == Severity.LOW

    def test_heuristic_fallback(self) -> None:
        agent = QAAgent()
        out = "pytest output: 42 passed, 3 failed, 1 skipped in 1.23s"
        report, err = agent.parse_results("qa-1", out)
        assert err is None
        assert report.test_passed == 42
        assert report.test_failed == 3
        assert report.test_skipped == 1

    def test_heuristic_partial(self) -> None:
        agent = QAAgent()
        out = "summary: 7 passed"
        report, err = agent.parse_results("qa-1", out)
        assert err is None
        assert report.test_passed == 7
        assert report.test_failed == 0
        assert report.test_skipped == 0

    def test_no_patterns_returns_minimal_report(self) -> None:
        agent = QAAgent()
        out = "completely unparseable executor output"
        report, err = agent.parse_results("qa-1", out)
        assert err is None
        assert report.test_passed == 0
        assert report.test_failed == 0
        assert report.test_output.startswith("completely unparseable")

    def test_risks_default_to_medium(self) -> None:
        agent = QAAgent()
        out = json.dumps({
            "test_passed": 1,
            "test_failed": 0,
            "test_skipped": 0,
            "risks": [{"severity": "imminent", "title": "t", "description": "d"}],
        })
        report, _ = agent.parse_results("qa-1", out)
        assert report.risks[0].severity == Severity.MEDIUM


# ---------------------------------------------------------------------------
# 7. QAAgent.recommend_executor
# ---------------------------------------------------------------------------


class TestQARecommendExecutor:
    def test_opencode_first(self) -> None:
        agent = QAAgent()
        eid, _ = agent.recommend_executor(["claude-code", "opencode", "deepseek-tui"])
        assert eid == "opencode"

    def test_deepseek_second(self) -> None:
        agent = QAAgent()
        eid, _ = agent.recommend_executor(["deepseek-tui"])
        assert eid == "deepseek-tui"

    def test_fallback_when_none(self) -> None:
        agent = QAAgent()
        eid, reason = agent.recommend_executor([])
        assert eid == "hermes-local"
        assert "falling back" in reason.lower()
        assert "hermes-local" in reason.lower()


# ---------------------------------------------------------------------------
# 8. Convenience factories
# ---------------------------------------------------------------------------


class TestFactories:
    def test_create_review_agent(self) -> None:
        a = create_review_agent()
        assert isinstance(a, ReviewAgent)

    def test_create_qa_agent(self) -> None:
        a = create_qa_agent()
        assert isinstance(a, QAAgent)


# ---------------------------------------------------------------------------
# 9. Module-level priority constants are well-formed
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_review_priority_includes_claude(self) -> None:
        assert "claude-code" in review_agent._REVIEW_EXECUTOR_PRIORITY

    def test_qa_priority_includes_opencode(self) -> None:
        assert "opencode" in review_agent._QA_EXECUTOR_PRIORITY
