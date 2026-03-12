from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FailureAssessment:
    failure_class: str
    confidence: float
    recommendation: str
    retryable: bool


@dataclass
class RetryPolicy:
    max_retries: int
    backoff_seconds: list[float]
    reason: str


def classify_failure(
    *,
    tool_name: str,
    error_type: Optional[str],
    error_summary: Optional[str],
    retryable_hint: bool,
) -> FailureAssessment:
    """Classify a tool failure into the super-agent failure taxonomy."""
    summary = (error_summary or "").lower()
    etype = (error_type or "").lower()

    if "reasoning" in summary or "halluc" in summary or "reasoning" in etype:
        return FailureAssessment(
            failure_class="reasoning_failure",
            confidence=0.74,
            recommendation="Reframe objective and constraints, then regenerate the plan before retry.",
            retryable=False,
        )

    if retryable_hint or "timeout" in summary or "rate" in summary or "connection" in summary:
        return FailureAssessment(
            failure_class="provider_failure",
            confidence=0.8,
            recommendation="Retry with backoff and preserve prior reasoning state.",
            retryable=True,
        )

    if "json" in summary or "schema" in summary or "invalid" in summary:
        return FailureAssessment(
            failure_class="schema_mismatch",
            confidence=0.78,
            recommendation="Ask model to repair tool args/schema before reissuing call.",
            retryable=True,
        )

    if tool_name in {"terminal", "patch", "write_file"} and ("permission" in summary or "denied" in summary):
        return FailureAssessment(
            failure_class="unsafe_action_block",
            confidence=0.82,
            recommendation="Request explicit approval before retrying side-effecting action.",
            retryable=False,
        )

    if "missing" in summary or "not found" in summary:
        return FailureAssessment(
            failure_class="missing_context",
            confidence=0.7,
            recommendation="Gather required context/input, then retry once.",
            retryable=True,
        )

    return FailureAssessment(
        failure_class="tool_misuse",
        confidence=0.6,
        recommendation="Use tool-specific correction prompt and require argument sanity-check.",
        retryable=True,
    )


def get_retry_policy(*, tool_name: str, failure: FailureAssessment) -> RetryPolicy:
    """Return tool-aware retry policy for bounded live retries."""
    provider_tool = tool_name.startswith("web_") or tool_name.startswith("browser_") or tool_name.startswith("mcp")

    if failure.failure_class == "provider_failure" and failure.retryable:
        if provider_tool:
            return RetryPolicy(max_retries=2, backoff_seconds=[0.2, 0.5], reason="provider_backoff")
        return RetryPolicy(max_retries=1, backoff_seconds=[0.25], reason="single_transient_retry")

    if failure.failure_class == "schema_mismatch" and failure.retryable:
        return RetryPolicy(max_retries=1, backoff_seconds=[0.0], reason="schema_repair_retry")

    if failure.failure_class == "missing_context" and failure.retryable:
        return RetryPolicy(max_retries=1, backoff_seconds=[0.0], reason="context_retry")

    return RetryPolicy(max_retries=0, backoff_seconds=[], reason=f"{failure.failure_class}_non_retry")


def build_shadow_critique_event(
    *,
    tool_name: str,
    error_type: Optional[str],
    error_summary: Optional[str],
    retryable_hint: bool,
) -> Dict[str, Any]:
    """Build a shadow-mode critique payload without changing live behavior."""
    assessment = classify_failure(
        tool_name=tool_name,
        error_type=error_type,
        error_summary=error_summary,
        retryable_hint=retryable_hint,
    )
    return {
        "tool_name": tool_name,
        "failure_class": assessment.failure_class,
        "confidence": assessment.confidence,
        "recommendation": assessment.recommendation,
        "retryable": assessment.retryable,
        "shadow_mode": True,
    }
