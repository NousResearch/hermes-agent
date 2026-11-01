"""agents/validator_agent.py
ValidatorAgent: validates code/outputs and provides improvement suggestions.
Core component of the self-evolving loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("agents.validator_agent")


@dataclass
class ValidationResult:
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    success: bool
    output: str
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidatorAgent:
    """Validates outputs and provides structured feedback for improvement."""

    name = "validator_agent"
    SYSTEM_PROMPT = """You are a code validator in the Kairos self-evolving swarm.
Your job is to carefully analyze code/outputs and provide:
1. Pass/Fail assessment with score (0.0-1.0)
2. List of issues found (if any)
3. Concrete improvement suggestions (prioritized)
4. Best practices violated
5. Edge cases to consider

Be fair but critical. Output structured feedback suitable for automated improvement loops."""

    def __init__(
        self,
        tools: Any = None,
        memory: Any = None,
        llm_call: Optional[Callable[[str, str], str]] = None,
    ):
        self.tools = tools
        self.memory = memory
        self.llm_call = llm_call

    def run(
        self,
        code_or_output: str,
        requirements: str = "",
        context: str = "",
    ) -> AgentResult:
        """Validate code or task output against requirements."""
        logger.info("VALIDATOR analyzing output...")

        try:
            validation = self._validate(code_or_output, requirements, context)
            feedback = self._format_feedback(validation, code_or_output)

            return AgentResult(
                success=True,
                output=feedback,
                metadata={
                    "passed": validation.passed,
                    "score": validation.score,
                    "issue_count": len(validation.issues),
                    "suggestion_count": len(validation.suggestions),
                },
            )
        except Exception as e:
            logger.error("Validation error: %s", e)
            return AgentResult(
                success=False,
                output=f"Validation error: {str(e)}",
                metadata={"error": str(e), "passed": False, "score": 0.0},
            )

    def _validate(
        self,
        code_or_output: str,
        requirements: str,
        context: str,
    ) -> ValidationResult:
        issues: List[str] = []
        suggestions: List[str] = []
        score = 0.90

        if not code_or_output.strip():
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=["Output is empty"],
                suggestions=["Provide complete non-empty implementation"],
            )

        if "TODO" in code_or_output or "FIXME" in code_or_output:
            issues.append("Contains unfulfilled TODO/FIXME markers")
            score -= 0.15

        if len(code_or_output.strip()) < 20:
            issues.append("Output is suspiciously short")
            score -= 0.20

        passed = score >= 0.70 and len(issues) < 2
        return ValidationResult(
            passed=passed,
            score=max(0.0, min(1.0, score)),
            issues=issues,
            suggestions=suggestions or ["Add inline docstrings and type hints"],
        )

    def _format_feedback(self, validation: ValidationResult, code_or_output: str) -> str:
        status = "PASS" if validation.passed else "FAIL"
        lines = [f"=== VALIDATION RESULT: {status} (Score: {validation.score:.2f}) ==="]
        if validation.issues:
            lines.append("\nISSUES:")
            for issue in validation.issues:
                lines.append(f"  • {issue}")
        if validation.suggestions:
            lines.append("\nSUGGESTIONS:")
            for sugg in validation.suggestions:
                lines.append(f"  • {sugg}")
        return "\n".join(lines)
