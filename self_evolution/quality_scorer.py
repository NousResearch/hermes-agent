"""
Self Evolution Plugin — Quality Scorer
=======================================

Computes a composite quality score for each session:

  session_quality = 0.4 * completion_rate
                  + 0.2 * efficiency_score
                  + 0.15 * cost_efficiency
                  + 0.25 * satisfaction_proxy

Zero API cost — pure computation from already-collected session data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from self_evolution.models import QualityScore

logger = logging.getLogger(__name__)

# ── Weights ──────────────────────────────────────────────────────────────

W_COMPLETION = 0.40
W_EFFICIENCY = 0.20
W_COST = 0.15
W_SATISFACTION = 0.25

# Ideal iteration counts by task complexity
IDEAL_ITERATIONS = {
    "simple": 3,
    "medium": 8,
    "complex": 15,
}
DEFAULT_IDEAL_ITERATIONS = 8


def compute_score(session_data: dict) -> QualityScore:
    """Compute a composite quality score from session data.

    Args:
        session_data: dict with keys like:
            - completed, interrupted, partial
            - iterations, max_iterations
            - tool_call_count, message_count
            - input_tokens, output_tokens, estimated_cost_usd
            - duration_seconds
            - model, platform
            - messages (list)

    Returns:
        QualityScore with individual and composite scores.
    """
    session_id = session_data.get("session_id", "")

    completion = _completion_rate(session_data)
    efficiency = _efficiency_score(session_data)
    cost = _cost_efficiency(session_data)
    satisfaction = _satisfaction_proxy(session_data)

    composite = (
        W_COMPLETION * completion
        + W_EFFICIENCY * efficiency
        + W_COST * cost
        + W_SATISFACTION * satisfaction
    )

    return QualityScore(
        session_id=session_id,
        composite=round(composite, 3),
        completion_rate=round(completion, 3),
        efficiency_score=round(efficiency, 3),
        cost_efficiency=round(cost, 3),
        satisfaction_proxy=round(satisfaction, 3),
        task_category=_detect_task_category(session_data),
        model=session_data.get("model", ""),
    )


# ── Individual Score Components ──────────────────────────────────────────

def _completion_rate(session_data: dict) -> float:
    """1.0 if completed, 0.5 if interrupted, 0.0 if failed."""
    if session_data.get("completed"):
        return 1.0
    if session_data.get("interrupted"):
        return 0.5
    if session_data.get("partial"):
        return 0.3
    return 0.0


def _efficiency_score(session_data: dict) -> float:
    """Ideal iterations / actual iterations, capped at 1.0."""
    iterations = session_data.get("iterations", 0) or session_data.get("tool_call_count", 0)
    if iterations <= 0:
        return 1.0

    category = _detect_task_category(session_data)
    ideal = IDEAL_ITERATIONS.get(category, DEFAULT_IDEAL_ITERATIONS)

    return min(1.0, ideal / max(iterations, 1))


def _cost_efficiency(session_data: dict) -> float:
    """Baseline cost / actual cost, capped at 1.0.

    Uses message count as a proxy for expected work.
    """
    messages = session_data.get("message_count", 1) or 1
    tool_calls = session_data.get("tool_call_count", 0) or 0
    iterations = session_data.get("iterations", 0) or 0

    # Expected: roughly 2 tool calls per user message
    expected_tool_calls = messages * 2

    if expected_tool_calls <= 0:
        return 1.0

    return min(1.0, expected_tool_calls / max(tool_calls, 1))


def _satisfaction_proxy(session_data: dict) -> float:
    """Estimate satisfaction from behavioral signals.

    Signals:
    - Single-turn session (user got what they needed) = high
    - Multi-turn but completed = medium-high
    - User corrections detected = lower
    - Budget exhausted = low
    """
    messages = session_data.get("message_count", 1) or 1
    completed = session_data.get("completed", False)
    max_iterations = session_data.get("max_iterations", 0)
    iterations = session_data.get("iterations", 0)

    score = 0.7  # baseline

    # Single-turn completion is a strong positive signal
    if messages <= 2 and completed:
        score = 0.9
    elif completed:
        score = 0.75
    elif messages > 10:
        score = 0.5

    # Budget exhaustion is a negative signal
    if max_iterations and iterations >= max_iterations:
        score -= 0.2

    return max(0.0, min(1.0, score))


# ── Task Category Detection ──────────────────────────────────────────────

def _detect_task_category(session_data: dict) -> str:
    """Detect task category from tool usage patterns."""
    tool_names = session_data.get("tool_names", [])
    if isinstance(tool_names, str):
        tool_names = tool_names.split(",")

    tool_set = set(t.lower() for t in tool_names) if tool_names else set()

    coding_tools = {"terminal", "bash", "write", "edit", "file_write", "file_edit"}
    web_tools = {"web_search", "browser", "browser_navigate", "scrape", "fetch"}
    file_tools = {"read", "file_read", "grep", "glob", "find"}

    if tool_set & coding_tools:
        return "coding"
    if tool_set & web_tools:
        return "web_research"
    if tool_set & file_tools:
        return "file_analysis"

    return "general"
