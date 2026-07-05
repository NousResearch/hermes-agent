"""Task Evaluator — 5 methods (test_pass, file_exists, content_match, command_output, llm_judge)."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.evolution.task_definition import (
    SuccessCriterion,
    SuccessCriterionType,
    TaskDefinition,
)
from agent.evolution.trajectory_collector import EvalCheck, EvalResult, Trajectory

logger = logging.getLogger(__name__)

# Timeout for evaluation commands
DEFAULT_COMMAND_TIMEOUT = 120  # seconds
# Max output bytes to capture from eval commands
MAX_OUTPUT_BYTES = 100_000


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class EvaluationContext:
    """Context passed to evaluation methods."""
    working_dir: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    trajectory: Optional[Trajectory] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class TaskEvaluator:
    """Evaluates a task attempt against its success criteria."""

    def __init__(self, llm_judge_fn: Optional[Any] = None):
        """Initialize the evaluator.

        Args:
            llm_judge_fn: Optional async callable for LLM-based judging.
              Signature: async (rubric: str, trajectory_json: str) -> Dict[str, Any]
              If None, llm_judge criteria are skipped with a warning.
        """
        self._llm_judge_fn = llm_judge_fn

    def evaluate(
        self,
        task: TaskDefinition,
        trajectory: Trajectory,
        ctx: Optional[EvaluationContext] = None,
    ) -> EvalResult:
        """Evaluate a completed trajectory against all success criteria.

        Args:
            task: The task definition with success criteria.
            trajectory: The completed agent trajectory.
            ctx: Optional evaluation context (working dir, env vars, etc.).

        Returns:
            EvalResult with aggregate score and per-criterion details.
        """
        if ctx is None:
            ctx = EvaluationContext(trajectory=trajectory)

        checks: List[EvalCheck] = []
        total_weight = 0.0
        weighted_score = 0.0

        for criterion in task.success_criteria:
            check = self._evaluate_criterion(criterion, ctx)
            checks.append(check)
            weighted_score += check.score * criterion.weight
            total_weight += criterion.weight

        if total_weight == 0:
            overall_score = 0.0
        else:
            overall_score = weighted_score / total_weight

        # Task passes if ALL criteria pass (logical AND)
        all_passed = all(c.passed for c in checks)

        return EvalResult(passed=all_passed, score=overall_score, checks=checks)

    def _evaluate_criterion(self, criterion: SuccessCriterion, ctx: EvaluationContext) -> EvalCheck:
        """Evaluate a single success criterion."""
        method_map = {
            SuccessCriterionType.TEST_PASS: self._eval_test_pass,
            SuccessCriterionType.FILE_EXISTS: self._eval_file_exists,
            SuccessCriterionType.CONTENT_MATCH: self._eval_content_match,
            SuccessCriterionType.COMMAND_OUTPUT: self._eval_command_output,
            SuccessCriterionType.LLM_JUDGE: self._eval_llm_judge,
            SuccessCriterionType.MANUAL: self._eval_manual,
        }
        handler = method_map.get(criterion.type)
        if handler is None:
            return EvalCheck(type=criterion.type.value, passed=False, detail=f"Unknown criterion type: {criterion.type}")

        try:
            return handler(criterion, ctx)
        except Exception as e:
            logger.warning("Evaluation failed for criterion %s: %s", criterion.type, e)
            return EvalCheck(type=criterion.type.value, passed=False, detail=f"Evaluation error: {e}")

    # -- Evaluation methods ---------------------------------------------------

    def _eval_test_pass(self, criterion: SuccessCriterion, ctx: EvaluationContext) -> EvalCheck:
        """Run a shell command and check exit code."""
        if not criterion.command:
            return EvalCheck(type="test_pass", passed=False, detail="No command specified")

        try:
            result = subprocess.run(
                criterion.command,
                shell=True,
                capture_output=True,
                timeout=DEFAULT_COMMAND_TIMEOUT,
                cwd=ctx.working_dir or None,
                env={**ctx.env} if ctx.env else None,
            )
            passed = result.returncode == 0
            detail = (
                f"Exit code: {result.returncode}\n"
                f"stdout: {_truncate_output(result.stdout.decode('utf-8', errors='replace'))}\n"
                f"stderr: {_truncate_output(result.stderr.decode('utf-8', errors='replace'))}"
            )
            return EvalCheck(
                type="test_pass",
                passed=passed,
                detail=detail[:1000],
                score=1.0 if passed else 0.0,
            )
        except subprocess.TimeoutExpired:
            return EvalCheck(type="test_pass", passed=False, detail=f"Command timed out after {DEFAULT_COMMAND_TIMEOUT}s")
        except Exception as e:
            return EvalCheck(type="test_pass", passed=False, detail=f"Command execution failed: {e}")

    def _eval_file_exists(self, criterion: SuccessCriterion, ctx: EvaluationContext) -> EvalCheck:
        """Check if a file exists."""
        if not criterion.path:
            return EvalCheck(type="file_exists", passed=False, detail="No path specified")

        # Resolve relative to working dir
        path = Path(criterion.path)
        if not path.is_absolute() and ctx.working_dir:
            path = Path(ctx.working_dir) / path

        exists = path.exists()
        detail = f"File {'exists' if exists else 'does not exist'}: {path}"
        if exists:
            try:
                size = path.stat().st_size
                detail += f" ({size} bytes)"
            except OSError:
                pass
        return EvalCheck(type="file_exists", passed=exists, detail=detail, score=1.0 if exists else 0.0)

    def _eval_content_match(self, criterion: SuccessCriterion, ctx: EvaluationContext) -> EvalCheck:
        """Check file content against a regex pattern."""
        if not criterion.pattern:
            return EvalCheck(type="content_match", passed=False, detail="No pattern specified")

        # Determine the file to check
        file_path = criterion.path
        if not file_path:
            return EvalCheck(type="content_match", passed=False, detail="No file path specified for content_match")

        path = Path(file_path)
        if not path.is_absolute() and ctx.working_dir:
            path = Path(ctx.working_dir) / path

        if not path.exists():
            return EvalCheck(type="content_match", passed=False, detail=f"File does not exist: {path}")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(criterion.pattern, content, re.MULTILINE | re.DOTALL)
            passed = len(matches) > 0
            detail = f"Pattern '{criterion.pattern[:100]}': {len(matches)} match(es) found in {path}"
            if matches:
                detail += f"\nFirst match: {str(matches[0])[:200]}"
            return EvalCheck(type="content_match", passed=passed, detail=detail[:1000], score=1.0 if passed else 0.0)
        except Exception as e:
            return EvalCheck(type="content_match", passed=False, detail=f"Content read/scan error: {e}")

    def _eval_command_output(self, criterion: SuccessCriterion, ctx: EvaluationContext) -> EvalCheck:
        """Run a command and check if output matches expected."""
        if not criterion.command:
            return EvalCheck(type="command_output", passed=False, detail="No command specified")

        try:
            result = subprocess.run(
                criterion.command,
                shell=True,
                capture_output=True,
                timeout=DEFAULT_COMMAND_TIMEOUT,
                cwd=ctx.working_dir or None,
                env={**ctx.env} if ctx.env else None,
            )
            output = result.stdout.decode("utf-8", errors="replace")
            if criterion.expected_output:
                # Partial match: expected output appears somewhere in the output
                passed = criterion.expected_output in output
                detail = (
                    f"Expected: '{criterion.expected_output[:200]}'\n"
                    f"Found: {passed}\n"
                    f"Output: {_truncate_output(output)}"
                )
                score = 1.0 if passed else 0.0
            else:
                # No expected output specified: just check it ran successfully
                passed = result.returncode == 0
                detail = f"Exit: {result.returncode}\nOutput: {_truncate_output(output)}"
                score = 1.0 if passed else 0.0
            return EvalCheck(type="command_output", passed=passed, detail=detail[:1000], score=score)
        except subprocess.TimeoutExpired:
            return EvalCheck(type="command_output", passed=False, detail=f"Timed out after {DEFAULT_COMMAND_TIMEOUT}s")
        except Exception as e:
            return EvalCheck(type="command_output", passed=False, detail=f"Error: {e}")

    def _eval_llm_judge(self, criterion: SuccessCriterion, ctx: EvaluationContext) -> EvalCheck:
        """Use an auxiliary LLM to judge qualitative criteria."""
        if not self._llm_judge_fn:
            return EvalCheck(
                type="llm_judge",
                passed=False,
                detail="LLM judge function not configured",
            )
        if not criterion.rubric:
            return EvalCheck(type="llm_judge", passed=False, detail="No rubric specified")

        try:
            import asyncio
            trajectory_json = ctx.trajectory.to_json() if ctx.trajectory else "{}"
            result = asyncio.get_event_loop().run_until_complete(
                self._llm_judge_fn(criterion.rubric, trajectory_json)
            )
            passed = bool(result.get("passed", False))
            score = float(result.get("score", 1.0 if passed else 0.0))
            detail = str(result.get("reasoning", result.get("detail", "")))
            return EvalCheck(type="llm_judge", passed=passed, detail=detail[:1000], score=score)
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)
            return EvalCheck(type="llm_judge", passed=False, detail=f"LLM judge error: {e}")

    def _eval_manual(self, criterion: SuccessCriterion, ctx: EvaluationContext) -> EvalCheck:
        """Manual evaluation — always returns pending/incomplete."""
        return EvalCheck(
            type="manual",
            passed=False,
            detail="Manual verification required — awaiting human judgment",
            score=0.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_output(text: str, max_chars: int = 500) -> str:
    """Truncate command output for storage in evaluation results."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text)} total chars]"


# ---------------------------------------------------------------------------
# Quick evaluation (no LLM judge needed)
# ---------------------------------------------------------------------------


def quick_evaluate(task: TaskDefinition, working_dir: str = "") -> EvalResult:
    """Evaluate a task using only deterministic criteria (no LLM judge).

    Useful for CI/CD pipelines and fast regression checks.
    """
    evaluator = TaskEvaluator(llm_judge_fn=None)
    # Filter out llm_judge and manual criteria
    deterministic_criteria = [
        c for c in task.success_criteria
        if c.type not in (SuccessCriterionType.LLM_JUDGE, SuccessCriterionType.MANUAL)
    ]
    if not deterministic_criteria:
        return EvalResult(passed=True, score=1.0, checks=[])
    # Create a temporary task with only deterministic criteria
    temp_task = TaskDefinition(
        name=task.name,
        description=task.description,
        success_criteria=deterministic_criteria,
    )
    ctx = EvaluationContext(working_dir=working_dir)
    return evaluator.evaluate(temp_task, Trajectory(), ctx)
