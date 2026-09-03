"""Model-relative lifecycle decisions for bounded long-running turns.

Historical call/cache counters are recorded by callers for observability only.
They intentionally do not participate in the admission decision: only live
context headroom and the configured turn budget can drain a session.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import sys


class LifecycleState(StrEnum):
    HEALTHY = "healthy"
    HEAVY = "heavy"
    DRAINING = "draining"


@dataclass(frozen=True)
class LifecycleBudget:
    context_window_tokens: int
    prompt_tokens: int
    reserved_output_tokens: int
    reserved_tool_result_tokens: int
    reserved_checkpoint_tokens: int
    max_iterations: int
    iterations_used: int
    api_calls: int = 0
    cache_read_tokens: int = 0
    compactions: int = 0
    in_flight_workers: int = 0
    closeout_iterations: int = 2
    heavy_utilization_ratio: float = 0.70


@dataclass(frozen=True)
class LifecycleDecision:
    state: LifecycleState
    context_utilization: float
    remaining_context_tokens: int
    remaining_iterations: int | None
    accept_new_tools: bool
    accept_new_delegations: bool
    allow_in_flight_completion: bool = True
    #: Tokens deliberately withheld from ordinary work so a checkpoint and a
    #: final response always fit. Published for status/doctor observability.
    reserved_headroom_tokens: int = 0


def evaluate_lifecycle(budget: LifecycleBudget) -> LifecycleDecision:
    """Classify a turn without fixed token thresholds.

    ``draining`` starts while the checkpoint reserve still fits. That is the
    last safe point to stop admitting new work and persist a continuation.
    """
    window = max(0, int(budget.context_window_tokens))
    prompt = max(0, int(budget.prompt_tokens))
    reserve_output = max(0, int(budget.reserved_output_tokens))
    reserve_tools = max(0, int(budget.reserved_tool_result_tokens))
    reserve_checkpoint = max(0, int(budget.reserved_checkpoint_tokens))
    utilization = (prompt / window) if window else 1.0
    remaining_context = max(0, window - prompt)

    max_iterations = int(budget.max_iterations)
    if max_iterations >= sys.maxsize:
        remaining_iterations = None
        closeout_exhausted = False
    else:
        remaining_iterations = max(0, max_iterations - max(0, int(budget.iterations_used)))
        closeout_exhausted = remaining_iterations <= max(1, int(budget.closeout_iterations))

    checkpoint_boundary = reserve_output + reserve_tools + reserve_checkpoint
    heavy_boundary = reserve_output + reserve_tools
    draining = (
        not window
        or remaining_context <= checkpoint_boundary
        # A turn-count closeout protects a real checkpoint/output reserve. With
        # zero reserve it becomes a model-independent early fence, so ratio-only
        # policy remains healthy until live context says otherwise.
        or (checkpoint_boundary > 0 and closeout_exhausted)
    )
    heavy = draining or remaining_context <= heavy_boundary or utilization >= float(budget.heavy_utilization_ratio)
    state = LifecycleState.DRAINING if draining else LifecycleState.HEAVY if heavy else LifecycleState.HEALTHY
    return LifecycleDecision(
        state=state,
        context_utilization=utilization,
        remaining_context_tokens=remaining_context,
        remaining_iterations=remaining_iterations,
        accept_new_tools=state is LifecycleState.HEALTHY,
        accept_new_delegations=state is LifecycleState.HEALTHY,
        reserved_headroom_tokens=checkpoint_boundary,
    )
