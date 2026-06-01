"""continuous_mode.py — Continuous / Converge / Supervised mode support for HITL harness.

Extends the cron scheduler to support GOAL.md-driven autonomous improvement loops
that run overnight or until a target score is reached.

Modes:
- converge: run until target score reached or max iterations / stall limit
- continuous: run indefinitely (or until max_iterations), reporting progress
- supervised: human gate required for high-stakes actions

Usage (from cron job):
    from agent.hitl_harness.continuous_mode import run_continuous_goal
    result = run_continuous_goal(goal_path, mode="converge", max_iterations=50)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.hitl_harness import HITLHarness, HITLHarnessConfig
from agent.hitl_harness.fitness_engine import FitnessEngine, FitnessResult
from agent.hitl_harness.fitness_wizard import FitnessFunctionWizard, FitnessGoal
from agent.hitl_harness.iterations_logger import IterationsLogger

logger = logging.getLogger(__name__)


@dataclass
class ContinuousRunResult:
    """Result of a continuous/converge mode run."""

    mode: str
    iterations: int
    best_combined_score: float
    final_primary_score: float
    final_instrument_trust_score: float
    halted: bool
    halt_reason: str
    log_path: str
    action_history: List[str] = field(default_factory=list)


def _load_goal(goal_path: str) -> Optional[FitnessGoal]:
    """Load a FitnessGoal from a GOAL.md file or JSON."""
    path = Path(goal_path).expanduser()
    if not path.exists():
        logger.warning("Goal file not found: %s", goal_path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FitnessGoal(**data)
    except Exception as e:
        logger.warning("Failed to load goal from %s: %s", goal_path, e)
        return None


def run_continuous_goal(
    goal_path: str,
    mode: str = "converge",
    max_iterations: int = 50,
    target_score: float = 0.95,
    hitl_config: Optional[Dict[str, Any]] = None,
    agent: Any = None,
) -> ContinuousRunResult:
    """Run an autonomous improvement loop against a GOAL.md.

    Args:
        goal_path: Path to the GOAL.md file
        mode: "converge" | "continuous" | "supervised"
        max_iterations: Hard cap on iterations
        target_score: For converge mode, stop when combined_score >= this
        hitl_config: Optional HITLHarnessConfig override dict
        agent: Optional AIAgent instance (creates a minimal fake if None)

    Returns:
        ContinuousRunResult with final scores and history
    """
    goal = _load_goal(goal_path)
    if goal is None:
        return ContinuousRunResult(
            mode=mode,
            iterations=0,
            best_combined_score=0.0,
            final_primary_score=0.0,
            final_instrument_trust_score=0.0,
            halted=True,
            halt_reason="Goal file not found or invalid",
            log_path="",
        )

    # Build or reuse harness
    cfg = HITLHarnessConfig.from_agent_config({"hitl": hitl_config or {"enabled": True}})
    harness = HITLHarness(agent, cfg) if agent else _make_minimal_harness(cfg)

    # Load goal into wizard
    harness.fitness_wizard.current_goal = goal
    harness.fitness_wizard.enabled = True

    # Initialize fitness engine
    engine = FitnessEngine(goal=goal, deepeval_adapter=harness.deepeval)

    logger.info(
        "Starting %s mode for goal '%s' (target=%.2f, max_iter=%d)",
        mode, goal.name, target_score, max_iterations,
    )

    iteration = 0
    best_score = 0.0
    action_history: List[str] = []
    halt_reason = ""
    primary = 0.0
    instrument = 0.0
    decision = None

    while iteration < max_iterations:
        iteration += 1
        harness._iteration_count = iteration

        # Build a synthetic trajectory for evaluation
        trajectory = {
            "messages": [],
            "tool_calls": [],
            "final_response": f"iteration_{iteration}",
        }

        # Evaluate current state
        fitness_result = engine.evaluate(trajectory, {"iteration": iteration, "mode": mode})
        primary = fitness_result.primary_score
        instrument = fitness_result.instrument_trust_score
        combined = fitness_result.combined_score

        # State locking decision
        decision = harness.optimizer.evaluate_step(primary, instrument_trust=instrument)

        # Log iteration
        harness.iterations_logger.log_iteration(
            primary_score=primary,
            instrument_trust_score=instrument,
            action=decision.action,
            iteration=iteration,
            locked_constraints=decision.locked_constraints,
            explanation=decision.summary,
            metadata={
                "mode": mode,
                "goal_id": goal.id,
                "metric_breakdown": fitness_result.metric_breakdown,
            },
        )

        if combined > best_score:
            best_score = combined

        action_history.append(decision.action)

        # Mode-specific stopping logic
        if mode == "converge" and combined >= target_score:
            halt_reason = f"Target score {target_score} reached at iteration {iteration}"
            logger.info(halt_reason)
            break

        if decision is not None and decision.action == "halt":
            halt_reason = decision.summary
            break

        if mode == "supervised" and iteration % 5 == 0:
            # In supervised mode, pause every 5 iterations for human review
            logger.info("Supervised mode: pausing for human review at iteration %d", iteration)
            # In a real implementation, this would emit a notification/wait for approval
            # For now, we just log and continue

        # Simulate work delay (in real use, the agent would make changes here)
        time.sleep(0.1)

    else:
        halt_reason = f"Max iterations ({max_iterations}) reached"

    logger.info(
        "Run complete: mode=%s iterations=%d best_score=%.4f reason=%s",
        mode, iteration, best_score, halt_reason,
    )

    return ContinuousRunResult(
        mode=mode,
        iterations=iteration,
        best_combined_score=best_score,
        final_primary_score=primary,
        final_instrument_trust_score=instrument,
        halted=decision.action == "halt" or iteration >= max_iterations,
        halt_reason=halt_reason,
        log_path=str(harness.iterations_logger.log_path),
        action_history=action_history,
    )


def _make_minimal_harness(cfg: HITLHarnessConfig) -> HITLHarness:
    """Create a minimal harness for standalone continuous mode (no real agent)."""
    class _FakeAgent:
        session_id = "continuous"
        model = "continuous-mode"
    return HITLHarness(_FakeAgent(), cfg)


def format_run_report(result: ContinuousRunResult) -> str:
    """Format a run result as a markdown report for delivery."""
    lines = [
        "# Continuous Goal Run Report",
        "",
        f"**Mode:** {result.mode}",
        f"**Iterations:** {result.iterations}",
        f"**Best Combined Score:** {result.best_combined_score:.4f}",
        f"**Final Primary Score:** {result.final_primary_score:.4f}",
        f"**Final Instrument Trust:** {result.final_instrument_trust_score:.4f}",
        f"**Halted:** {result.halted}",
        f"**Halt Reason:** {result.halt_reason}",
        f"**Log Path:** {result.log_path}",
        "",
        "## Action History",
        "",
    ]
    for i, action in enumerate(result.action_history, 1):
        lines.append(f"- Iteration {i}: {action}")
    return "\n".join(lines)
