"""agents/validator_agent.py
ValidatorAgent: validates code/outputs and provides improvement suggestions.
Core component of the self-evolving loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("agents.validator_agent")


@dataclass
class ValidationResult:
    """Result of validation check."""

    passed: bool
    score: float  # 0.0 to 1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    success: bool
    output: str
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
        """
        Validate code or task output against requirements.

        Args:
            code_or_output: The code/output to validate
            requirements: The original requirements/spec
            context: Additional context about the task

        Returns:
            AgentResult with validation feedback
        """
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
            logger.error(f"Validation failed: {e}")
            return AgentResult(
                success=False,
                output=f"Validation error: {str(e)}",
                metadata={"error": str(e)},
            )

    def _validate(
        self,
        code_or_output: str,
        requirements: str,
        context: str,
    ) -> ValidationResult:
        """Perform validation using LLM or heuristics."""
        issues = []
        suggestions = []
        score = 1.0

        # Basic heuristic checks
        score = self._heuristic_checks(code_or_output, issues, suggestions)

        # LLM-based validation if available
        if self.llm_call:
            try:
                llm_feedback = self._llm_validate(code_or_output, requirements, context)
                if llm_feedback:
                    parsed = self._parse_llm_feedback(llm_feedback)
                    issues.extend(parsed.get("issues", []))
                    suggestions.extend(parsed.get("suggestions", []))
                    score = min(score, parsed.get("score", score))
            except Exception as e:
                logger.warning(f"LLM validation failed: {e}")

        passed = score >= 0.7

        return ValidationResult(
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata={"validation_type": "code"},
        )

    def _heuristic_checks(
        self,
        code: str,
        issues: list[str],
        suggestions: list[str],
    ) -> float:
        """Basic code quality heuristics."""
        score = 1.0

        # Check for common issues
        if not code or len(code.strip()) < 10:
            issues.append("Code is too short or empty")
            score -= 0.3

        if "TODO" in code or "FIXME" in code:
            suggestions.append("Remove TODO/FIXME comments before finalizing")
            score -= 0.05

        if "print(" in code and "__main__" not in code:
            suggestions.append("Consider using logging instead of print statements")
            score -= 0.02

        if len(code.split("\n")) > 500:
            suggestions.append("Consider breaking into smaller functions")
            score -= 0.05

        return max(0.0, score)

    def _llm_validate(self, code: str, requirements: str, context: str) -> str:
        """Get LLM validation feedback."""
        prompt = f"""Validate this code:

Requirements: {requirements}

Context: {context}

Code:
{code[:2000]}

Provide structured feedback with:
1. Pass/Fail (and score 0.0-1.0)
2. Issues (bullet list)
3. Suggestions (bullet list)
4. Best practices"""

        try:
            result = self.llm_call(self.SYSTEM_PROMPT, prompt)
            return result
        except Exception as e:
            logger.warning(f"LLM validation error: {e}")
            return ""

    def _parse_llm_feedback(self, feedback: str) -> dict[str, Any]:
        """Parse LLM feedback into structured data."""
        parsed = {
            "score": 0.7,
            "issues": [],
            "suggestions": [],
        }

        lines = feedback.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "score" in line.lower() and any(c.isdigit() for c in line):
                try:
                    score = float(line.split()[-1].strip("[](),"))
                    parsed["score"] = min(1.0, max(0.0, score))
                except ValueError:
                    pass

            if any(
                x in line.lower()
                for x in ["issue", "problem", "error", "fail"]
            ):
                current_section = "issues"
            elif any(x in line.lower() for x in ["suggestion", "recommend", "improve"]):
                current_section = "suggestions"

            if line.startswith(("-", "ΓÇó", "*")) or (
                line and line[0].isdigit() and "." in line
            ):
                if current_section:
                    parsed[current_section].append(line.lstrip("-ΓÇó* 0123456789. "))

        return parsed

    def _format_feedback(self, validation: ValidationResult, code: str) -> str:
        """Format validation result as readable feedback."""
        lines = [
            f"Γ£ô VALIDATION RESULT: {'PASS' if validation.passed else 'FAIL'} (Score: {validation.score:.2f})",
            "",
        ]

        if validation.issues:
            lines.append("ΓÜá∩╕Å  ISSUES FOUND:")
            for issue in validation.issues[:5]:
                lines.append(f"  ΓÇó {issue}")

        if validation.suggestions:
            lines.append("\n≡ƒÆí SUGGESTIONS FOR IMPROVEMENT:")
            for i, sugg in enumerate(validation.suggestions[:5], 1):
                lines.append(f"  {i}. {sugg}")

        lines.append(f"\n≡ƒôè CODE LENGTH: {len(code)} chars")
        lines.append(f"≡ƒôè LINES: {len(code.split(chr(10)))}")

        return "\n".join(lines)
