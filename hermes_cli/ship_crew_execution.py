"""Deterministic per-task context/output and execution guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ExecutionBudget:
    complexity_tier: str
    max_context_chars: int
    max_output_chars: int
    max_retries: int
    max_review_attempts: int
    goal_max_turns: int


_DEFAULTS: dict[str, ExecutionBudget] = {
    "T0": ExecutionBudget("T0", 6_000, 2_000, 0, 1, 1),
    "T1": ExecutionBudget("T1", 12_000, 4_000, 1, 1, 3),
    "T2": ExecutionBudget("T2", 20_000, 8_000, 2, 2, 8),
    "T3": ExecutionBudget("T3", 32_000, 12_000, 2, 2, 12),
    "T4": ExecutionBudget("T4", 48_000, 16_000, 3, 3, 20),
}


class BudgetExceededError(ValueError):
    """Raised when an output cannot be accepted under its declared budget."""


def resolve_execution_budget(
    complexity_tier: str = "T1", overrides: Optional[Mapping[str, Any]] = None
) -> ExecutionBudget:
    tier = str(complexity_tier).upper()
    if tier not in _DEFAULTS:
        raise ValueError(f"unknown complexity tier {complexity_tier!r}")
    base = _DEFAULTS[tier]
    values = {name: getattr(base, name) for name in base.__dataclass_fields__}
    for name in (
        "max_context_chars",
        "max_output_chars",
        "max_retries",
        "max_review_attempts",
        "goal_max_turns",
    ):
        if overrides and name in overrides and overrides[name] is not None:
            value = int(overrides[name])
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            values[name] = value
    # A zero retry/review budget is valid; a zero output/context budget is not.
    if values["max_context_chars"] <= 0 or values["max_output_chars"] <= 0:
        raise ValueError("context/output budgets must be positive")
    return ExecutionBudget(**values)


def compact_context(text: str, max_chars: int) -> str:
    """Keep a deterministic head/tail window with an explicit omission marker."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = str(text)
    if len(text) <= max_chars:
        return text
    marker = f"\n… [context compacted: {len(text) - max_chars} chars omitted] …\n"
    if len(marker) >= max_chars:
        return marker[:max_chars]
    remaining = max_chars - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + marker + text[-tail:] if tail else text[:head] + marker


def accept_output(text: str, budget: ExecutionBudget) -> str:
    """Return output if it fits; callers must retry or quarantine on overflow."""
    output = str(text)
    if len(output) > budget.max_output_chars:
        raise BudgetExceededError(
            f"output exceeds {budget.max_output_chars} chars for {budget.complexity_tier}"
        )
    return output


def retry_allowed(failure_count: int, budget: ExecutionBudget) -> bool:
    """Use a strict bound: N means N total failures are tolerated."""
    return int(failure_count) < budget.max_retries


def review_allowed(review_attempts: int, budget: ExecutionBudget) -> bool:
    """Never auto-approve after the independent review attempt budget."""
    return int(review_attempts) < budget.max_review_attempts


def goal_turn_allowed(turn: int, budget: ExecutionBudget) -> bool:
    return 0 <= int(turn) < budget.goal_max_turns
