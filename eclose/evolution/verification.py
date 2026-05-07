from dataclasses import dataclass
from eclose.evolution.execution import ExecutionResult


@dataclass
class VerificationResult:
    passed: bool
    score: float
    feedback: str
    improvement_suggestions: list[str]


class VerificationLayer:
    """Verifies execution results and generates feedback."""

    def verify(self, execution_result: ExecutionResult) -> VerificationResult:
        """Verify execution and assess results."""
        passed = execution_result.verification.get("passed", False)
        steps_completed = len(execution_result.results.get("steps_completed", []))
        steps_failed = len(execution_result.results.get("steps_failed", []))
        total_steps = steps_completed + steps_failed

        score = steps_completed / total_steps if total_steps > 0 else 0.0

        return VerificationResult(
            passed=passed,
            score=score,
            feedback=self._generate_feedback(execution_result),
            improvement_suggestions=self._get_suggestions(execution_result),
        )

    def _generate_feedback(self, result: ExecutionResult) -> str:
        if result.status == "success":
            return "Evolution executed successfully."
        elif result.status == "partial":
            return f"Partial success: {len(result.results.get('steps_failed', []))} steps failed."
        else:
            return "Evolution failed."

    def _get_suggestions(self, result: ExecutionResult) -> list[str]:
        if result.status == "failed":
            return ["Review failed steps and retry", "Consider simpler approach"]
        return []