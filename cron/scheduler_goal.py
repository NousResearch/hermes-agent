"""Goal-mode helpers for cron's otherwise one-shot scheduler path."""

from __future__ import annotations

from typing import Any, Callable


def goal_prompt_from_job(job: dict[str, Any]) -> str | None:
    """Return a non-empty ``/goal`` objective from a cron job's raw prompt.

    Cron normally prepends its execution preamble before calling the agent, so
    slash-command dispatch cannot see a leading ``/goal``.  Detect it before
    that assembly and leave malformed or ordinary prompts on the old path.
    """
    parts = str(job.get("prompt") or "").strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0] != "/goal" or not parts[1].strip():
        return None
    return parts[1].strip()


def cron_goal_session_id(job_id: str) -> str:
    """Stable GoalManager session key for one cron job across scheduled fires."""
    return f"cron-goal:{job_id}"


def run_goal_turns(
    manager: Any,
    goal: str,
    *,
    initial_prompt: str,
    run_turn: Callable[[str], dict[str, Any]],
    response_from_result: Callable[[dict[str, Any]], str],
) -> tuple[dict[str, Any], str, str]:
    """Drive a cron goal until its existing GoalManager reaches a boundary.

    The manager owns turn budgets, judge failures, wait barriers, and terminal
    states. ``initial_prompt`` is the already assembled cron prompt for a new
    goal; continuations use GoalManager's compact prompt. Cron only supplies
    the synchronous turn runner and returns one final delivery payload instead
    of emitting gateway progress messages.
    """
    state = manager.state
    if state is not None and getattr(state, "goal", goal) != goal:
        manager.set(goal)
        prompt = initial_prompt
    elif manager.is_active() and manager.is_waiting():
        return {}, "", "⏳ Goal remains parked; the next scheduled fire will retry when its wait barrier clears."
    elif manager.is_active():
        prompt = manager.next_continuation_prompt() or goal
    elif manager.has_goal():
        return {}, "", "⏸ Goal remains paused; resume it explicitly before the next scheduled fire."
    elif getattr(state, "status", None) == "done":
        return {}, "", "✓ Goal is already complete; set a new goal before the next scheduled fire."
    else:
        manager.set(goal)
        prompt = initial_prompt

    while True:
        result = run_turn(prompt)
        response = response_from_result(result)
        decision = manager.evaluate_after_turn(response, user_initiated=True)
        status = str(decision.get("message") or "")
        prompt = decision.get("continuation_prompt")
        if not decision.get("should_continue") or not prompt:
            return result, response, status
