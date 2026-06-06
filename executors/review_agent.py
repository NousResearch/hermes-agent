#!/usr/bin/env python3
"""
ReviewAgent and QAAgent — build prompts and parse outputs for review/QA runs.

Review run: analyzes main run diff + changed files for correctness, security,
  performance, maintainability, style, test coverage.  Default executor: claude-code.

QA run: runs test commands in the worktree, collects results, identifies risks.
  Default executor: opencode.

Both agents are READ-ONLY — they build prompts but do not execute runs directly.
Execution is handled by the existing adapter layer.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from executors.types import (
    ExecutorId,
    FindingCategory,
    ProjectContext,
    QARisk,
    QAReport,
    QAStatus,
    ReviewFinding,
    ReviewReport,
    ReviewStatus,
    RunType,
    Severity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recommended executors
# ---------------------------------------------------------------------------

_REVIEW_EXECUTOR_PRIORITY: List[ExecutorId] = ["claude-code", "opencode", "hermes-local"]
_QA_EXECUTOR_PRIORITY: List[ExecutorId] = ["opencode", "deepseek-tui", "claude-code"]

# ---------------------------------------------------------------------------
# Max diff lines injected into prompt
# ---------------------------------------------------------------------------

_MAX_DIFF_LINES = 2000
_MAX_FAILED_TEST_OUTPUT = 500


class ReviewAgent:
    """Builds review prompts and parses structured findings from executor output."""

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        task_goal: str,
        main_run_executor: str,
        changed_files: List[str],
        diff: str,
        context: Optional[ProjectContext] = None,
        main_run_prompt_snapshot: Optional[str] = None,
    ) -> str:
        """Build the review prompt for injection into a review run.

        Args:
            task_goal: The original task goal from the main run.
            main_run_executor: Which executor ran the main run.
            changed_files: List of changed file paths (with status prefix).
            diff: Unified diff of changes (truncated to _MAX_DIFF_LINES lines).
            context: Optional workspace context for architecture/coding conventions.
            main_run_prompt_snapshot: Original prompt snapshot from the main run.

        Returns:
            The full review prompt string.
        """
        parts: List[str] = []
        parts.append("--- Review Context ---")
        parts.append(f"Task Goal: {task_goal}")
        parts.append(f"Main Run Executor: {main_run_executor}")

        if context:
            if context.architecture_notes:
                parts.append(f"Architecture: {context.architecture_notes[:500]}")
            if context.coding_conventions:
                parts.append(f"Coding Conventions: {context.coding_conventions}")

        if changed_files:
            parts.append("\nChanged Files:")
            for f in changed_files:
                parts.append(f"  {f}")

        if diff:
            diff_lines = diff.strip().split("\n")
            if len(diff_lines) > _MAX_DIFF_LINES:
                diff_lines = diff_lines[:_MAX_DIFF_LINES]
                diff_lines.append(f"... ({len(diff.strip().split(chr(10))) - _MAX_DIFF_LINES} more lines truncated)")
            parts.append("\nDiff:")
            parts.append("\n".join(diff_lines))

        if main_run_prompt_snapshot:
            # Only include the user prompt part, not the full injected context
            user_part = main_run_prompt_snapshot
            if "--- End Context ---" in user_part:
                user_part = user_part.split("--- End Context ---")[-1].strip()
            parts.append(f"\nMain Run Prompt: {user_part[:500]}")

        parts.append("--- End Review Context ---")
        parts.append("")
        parts.append(self._review_instructions())

        return "\n".join(parts)

    @staticmethod
    def _review_instructions() -> str:
        return (
            "Review the above changes. For each finding, output a JSON object "
            "with these fields:\n"
            '  {"severity": "critical|high|medium|low|info",\n'
            '   "category": "correctness|security|performance|maintainability|style|test_coverage",\n'
            '   "file_path": "optional/path",\n'
            '   "line_start": optional_number,\n'
            '   "line_end": optional_number,\n'
            '   "title": "one-line summary",\n'
            '   "description": "detailed explanation",\n'
            '   "suggestion": "optional fix suggestion"}\n\n'
            "Output findings as a JSON array. Do NOT modify any code.\n\n"
            "Checklist:\n"
            "1. Correctness — logic errors, boundary conditions, edge cases\n"
            "2. Security — injection risks, hardcoded secrets, access control\n"
            "3. Performance — N+1 queries, unnecessary allocations, lock contention\n"
            "4. Maintainability — naming, structure, duplication, complexity\n"
            "5. Style — conventions, formatting, consistency\n"
            "6. Test coverage — missing tests for new/changed code"
        )

    # ------------------------------------------------------------------
    # Recommended executor
    # ------------------------------------------------------------------

    def recommend_executor(self, available: List[ExecutorId]) -> Tuple[ExecutorId, str]:
        """Return the best available review executor and reason."""
        for eid in _REVIEW_EXECUTOR_PRIORITY:
            if eid in available:
                reasons = {
                    "claude-code": "Claude Code excels at code review and design reasoning",
                    "opencode": "OpenCode is a capable local review agent",
                    "hermes-local": "Hermes Local (built-in fallback)",
                }
                return eid, reasons.get(eid, "Default review executor")
        return "hermes-local", "No review executors available — falling back to hermes-local"

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def parse_findings(
        self, review_run_id: str, executor_output: str
    ) -> Tuple[List[ReviewFinding], Optional[str]]:
        """Parse review findings from executor output.

        Attempts to find a JSON array in the output, then falls back to
        heuristic line-by-line parsing.

        Returns:
            (findings list, error or None if parsing succeeded)
        """
        # Try to find JSON array
        json_match = re.search(r"\[.*\]", executor_output, re.DOTALL)
        if json_match:
            try:
                raw = json.loads(json_match.group(0))
                if isinstance(raw, list):
                    findings = []
                    for item in raw:
                        if isinstance(item, dict):
                            findings.append(self._dict_to_finding(review_run_id, item))
                    if findings:
                        return findings, None
            except json.JSONDecodeError:
                pass

        # Fallback: try to find individual JSON objects
        objects = re.findall(r"\{[^{}]*\}", executor_output)
        if objects:
            findings = []
            for obj_str in objects:
                try:
                    item = json.loads(obj_str)
                    if isinstance(item, dict) and "severity" in item:
                        findings.append(self._dict_to_finding(review_run_id, item))
                except json.JSONDecodeError:
                    continue
            if findings:
                return findings, None

        # If no structured output found, create a single info finding with the raw output
        if executor_output.strip():
            f = ReviewFinding(
                id=f"{review_run_id}-raw",
                run_id=review_run_id,
                severity=Severity.INFO,
                category=FindingCategory.MAINTAINABILITY,
                title="Unstructured review output",
                description=executor_output[:2000],
            )
            return [f], "Could not parse structured findings from executor output"

        return [], "No findings or output from review run"

    @staticmethod
    def _dict_to_finding(review_run_id: str, d: Dict[str, Any]) -> ReviewFinding:
        severity_map = {s.value: s for s in Severity}
        category_map = {c.value: c for c in FindingCategory}

        return ReviewFinding(
            id=f"{review_run_id}-{d.get('title', 'unknown')[:20].replace(' ', '-')}"
                f"-{str(uuid.uuid4())[:8]}",
            run_id=review_run_id,
            severity=severity_map.get(d.get("severity", "medium"), Severity.MEDIUM),
            category=category_map.get(d.get("category", "maintainability"), FindingCategory.MAINTAINABILITY),
            file_path=d.get("file_path"),
            line_start=d.get("line_start"),
            line_end=d.get("line_end"),
            title=d.get("title", "Untitled finding"),
            description=d.get("description", ""),
            suggestion=d.get("suggestion"),
        )

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def build_report(
        self,
        review_run_id: str,
        executor: str,
        findings: List[ReviewFinding],
        status: ReviewStatus,
        started_at: Optional[datetime.datetime] = None,
        completed_at: Optional[datetime.datetime] = None,
        error: Optional[str] = None,
    ) -> ReviewReport:
        """Build a ReviewReport from parsed findings."""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for f in findings:
            severity_counts[f.severity.value] += 1

        duration = 0.0
        if started_at and completed_at:
            duration = (completed_at - started_at).total_seconds()

        return ReviewReport(
            review_run_id=review_run_id,
            status=status,
            executor=executor,
            total_findings=len(findings),
            critical_count=severity_counts["critical"],
            high_count=severity_counts["high"],
            medium_count=severity_counts["medium"],
            low_count=severity_counts["low"],
            info_count=severity_counts["info"],
            findings=findings,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            error=error,
        )


class QAAgent:
    """Builds QA prompts and parses test results from executor output."""

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        task_goal: str,
        changed_files: List[str],
        test_commands: List[Tuple[str, str]],  # (label, command)
        worktree_path: Optional[str] = None,
    ) -> str:
        """Build the QA prompt.

        Args:
            task_goal: The original task goal from the main run.
            changed_files: List of changed file paths.
            test_commands: List of (label, command) tuples from workspace context.
            worktree_path: Path to the worktree where tests should run.

        Returns:
            The full QA prompt string.
        """
        parts: List[str] = []
        parts.append("--- QA Context ---")
        parts.append(f"Task Goal: {task_goal}")

        if changed_files:
            parts.append("\nChanged Files:")
            for f in changed_files:
                parts.append(f"  {f}")

        if test_commands:
            parts.append("\nTest Commands:")
            for label, cmd in test_commands:
                parts.append(f"  {label}: {cmd}")

        if worktree_path:
            parts.append(f"\nWorktree Path: {worktree_path}")

        parts.append("--- End QA Context ---")
        parts.append("")

        if test_commands:
            parts.append("Execute the following test commands in the worktree:")
            for label, cmd in test_commands:
                parts.append(f"  $ {cmd}")
        else:
            parts.append("No test commands configured in workspace context.")

        parts.append("")
        parts.append(self._qa_instructions())
        return "\n".join(parts)

    @staticmethod
    def _qa_instructions() -> str:
        return (
            "After running the tests, output a JSON object with:\n"
            '  {"test_passed": number,\n'
            '   "test_failed": number,\n'
            '   "test_skipped": number,\n'
            '   "failed_test_details": "summary of failures",\n'
            '   "risks": [\n'
            '     {"severity": "high|medium|low",\n'
            '      "title": "risk title",\n'
            '      "description": "detail",\n'
            '      "affected_areas": ["module1", "module2"]}\n'
            '   ],\n'
            '   "coverage_delta": optional_float}\n\n'
            "Do NOT modify any code. Only run tests and report results."
        )

    # ------------------------------------------------------------------
    # Recommended executor
    # ------------------------------------------------------------------

    def recommend_executor(self, available: List[ExecutorId]) -> Tuple[ExecutorId, str]:
        """Return the best available QA executor and reason."""
        for eid in _QA_EXECUTOR_PRIORITY:
            if eid in available:
                reasons = {
                    "opencode": "OpenCode is ideal for local test execution",
                    "deepseek-tui": "DeepSeek TUI is fast for smoke tests",
                    "claude-code": "Claude Code (fallback QA executor)",
                }
                return eid, reasons.get(eid, "Default QA executor")
        return "hermes-local", "No QA executors available — falling back to hermes-local"

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def parse_results(
        self, qa_run_id: str, executor_output: str
    ) -> Tuple[QAReport, Optional[str]]:
        """Parse QA results from executor output.

        Attempts JSON parsing first, then heuristic fallback.

        Returns:
            (QAReport, error or None if parsing succeeded)
        """
        json_match = re.search(r"\{.*\}", executor_output, re.DOTALL)
        if json_match:
            try:
                raw = json.loads(json_match.group(0))
                if isinstance(raw, dict) and "test_passed" in raw:
                    risks = []
                    for r in raw.get("risks", []):
                        severity_map = {s.value: s for s in Severity}
                        risks.append(QARisk(
                            severity=severity_map.get(r.get("severity", "medium"), Severity.MEDIUM),
                            title=r.get("title", ""),
                            description=r.get("description", ""),
                            affected_areas=r.get("affected_areas", []),
                        ))

                    return QAReport(
                        qa_run_id=qa_run_id,
                        status=QAStatus.COMPLETED,
                        test_passed=raw.get("test_passed", 0),
                        test_failed=raw.get("test_failed", 0),
                        test_skipped=raw.get("test_skipped", 0),
                        test_output=raw.get("failed_test_details", executor_output[:_MAX_FAILED_TEST_OUTPUT]),
                        risks=risks,
                        coverage_delta=raw.get("coverage_delta"),
                    ), None
            except json.JSONDecodeError:
                pass

        # Fallback: extract test statistics from patterns
        passed_match = re.search(r"(\d+)\s+passed", executor_output)
        failed_match = re.search(r"(\d+)\s+failed", executor_output)
        skipped_match = re.search(r"(\d+)\s+skipped", executor_output)

        if passed_match or failed_match:
            return QAReport(
                qa_run_id=qa_run_id,
                status=QAStatus.COMPLETED,
                test_passed=int(passed_match.group(1)) if passed_match else 0,
                test_failed=int(failed_match.group(1)) if failed_match else 0,
                test_skipped=int(skipped_match.group(1)) if skipped_match else 0,
                test_output=executor_output[:_MAX_FAILED_TEST_OUTPUT],
            ), None

        return QAReport(
            qa_run_id=qa_run_id,
            status=QAStatus.COMPLETED,
            test_output=executor_output[:_MAX_FAILED_TEST_OUTPUT],
        ), None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_review_agent() -> ReviewAgent:
    return ReviewAgent()


def create_qa_agent() -> QAAgent:
    return QAAgent()
