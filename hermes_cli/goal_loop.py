"""Utilities for the /goal autonomous supervisor loop."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GoalDecision:
    """Supervisor decision for one goal-loop iteration."""

    complete: bool
    feedback: str


def parse_goal_supervisor_decision(text: str | None) -> GoalDecision:
    """Parse the supervisor's compact COMPLETE/CONTINUE decision.

    The supervisor prompt asks for one of:
      COMPLETE: <why>
      CONTINUE: <specific next-step instruction>

    Be permissive so providers that add light formatting still work.
    """

    raw = (text or "").strip()
    normalized = raw.lstrip("`*# \n\t").strip()
    upper = normalized.upper()
    if upper.startswith("COMPLETE"):
        feedback = normalized.split(":", 1)[1].strip() if ":" in normalized else normalized
        return GoalDecision(True, feedback or "Supervisor marked the goal complete.")
    if upper.startswith("CONTINUE"):
        feedback = normalized.split(":", 1)[1].strip() if ":" in normalized else normalized
        return GoalDecision(False, feedback or "Continue toward the goal and address any remaining gaps.")

    # Conservative fallback: ambiguous supervisor output means continue once with
    # the raw critique rather than silently declaring success.
    return GoalDecision(False, raw or "Supervisor returned an empty decision; continue and verify completion explicitly.")


def get_goal_max_loops(default: int = 6) -> int:
    """Return HERMES_GOAL_MAX_LOOPS as a safe positive integer."""

    try:
        return max(1, int(os.getenv("HERMES_GOAL_MAX_LOOPS", str(default))))
    except (TypeError, ValueError):
        return max(1, default)


def make_goal_worker_prompt(goal: str, iteration: int, previous_response: str = "", supervisor_feedback: str = "") -> str:
    """Build the prompt sent to the worker agent for a goal iteration."""

    if iteration <= 1:
        return (
            "You are running under /goal autonomous mode. Work directly toward this goal and keep going until the "
            "task is actually complete, using available tools for implementation and verification. Do not stop at a "
            "plan if you can act. When you believe the goal is complete, provide a concise final summary with evidence.\n\n"
            f"GOAL:\n{goal}"
        )

    return (
        "You are continuing a /goal autonomous loop. The supervisor judged the previous attempt incomplete. "
        "Address the feedback directly, continue implementation/verification with tools, and finish with concise evidence.\n\n"
        f"ORIGINAL GOAL:\n{goal}\n\n"
        f"SUPERVISOR FEEDBACK / NEXT STEP:\n{supervisor_feedback}\n\n"
        f"PREVIOUS WORKER RESPONSE:\n{previous_response[-6000:]}"
    )


def make_goal_supervisor_prompt(goal: str, iteration: int, worker_response: str) -> str:
    """Build the no-tools supervisor prompt for judging a worker iteration."""

    return (
        "You are the supervisor for a Hermes /goal loop. Decide whether the worker's latest response proves the "
        "goal is complete. Be strict: require concrete evidence such as tests, commit/push/CI status, generated files, "
        "or a clear explanation of an unrecoverable blocker. If more tool work is likely useful, choose CONTINUE.\n\n"
        "Reply in exactly one of these formats:\n"
        "COMPLETE: one-sentence reason\n"
        "CONTINUE: specific next instruction for the worker\n\n"
        f"GOAL:\n{goal}\n\n"
        f"ITERATION: {iteration}\n\n"
        f"WORKER RESPONSE:\n{worker_response[-12000:]}"
    )
