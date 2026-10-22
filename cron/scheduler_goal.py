"""Goal-mode helpers for cron's otherwise one-shot scheduler path."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


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


def goal_job_is_cancelled(job_id: str, cancel_event: Any = None) -> bool:
    """Fail closed at a turn boundary using the owning profile's current store.

    Read raw records: get_job() normalizes display state and can hide a
    contradictory enabled=True/state=paused record before scheduler self-heal.
    """
    if cancel_event is not None and cancel_event.is_set():
        return True
    from cron.jobs import is_job_runnable, load_jobs

    try:
        job = next((job for job in load_jobs() if job["id"] == job_id), None)
        return job is None or not is_job_runnable(job)
    except Exception:
        logger.warning("Cannot read cron goal job %s; stopping further turns", job_id, exc_info=True)
        return True


def finish_unstarted_goal_execution(job: dict[str, Any], execution_id: str) -> None:
    """Release a rejected goal claim without changing ordinary ownership handling."""
    if goal_prompt_from_job(job) is not None:
        from cron.executions import finish_execution

        finish_execution(
            execution_id, success=False,
            error="Goal job is already running; execution was not started.",
        )


def _pause_interrupted_goal(manager: Any) -> str:
    # Leave done/budget-paused goals intact; pause() preserves progress and budget.
    if manager.is_active():
        manager.pause("cron job interrupted (cancelled, paused, disabled, or unavailable)")
    return "⏸ Goal interrupted; resume the cron job and its goal explicitly to continue."


def run_goal_turns(
    manager: Any,
    goal: str,
    *,
    initial_prompt: str,
    run_turn: Callable[[str], dict[str, Any]],
    response_from_result: Callable[[dict[str, Any]], str],
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Drive a cron goal until its existing GoalManager reaches a boundary.

    The manager owns turn budgets, judge failures, wait barriers, and terminal
    states. ``initial_prompt`` is the already assembled cron prompt for this
    fire's first turn; only later turns use GoalManager's compact prompt. Cron
    only supplies the synchronous turn runner and returns one final delivery
    payload instead of emitting gateway progress messages. Cancellation is checked
    before initialization and between turns; it never resets an existing budget.
    """
    if is_cancelled is not None and is_cancelled():
        return {}, "", _pause_interrupted_goal(manager)

    state = manager.state
    if state is not None and getattr(state, "goal", goal) != goal:
        manager.set(goal)
        prompt = initial_prompt
    elif manager.is_active() and manager.is_waiting():
        return {}, "", "⏳ Goal remains parked; the next scheduled fire will retry when its wait barrier clears."
    elif manager.is_active():
        prompt = initial_prompt
    elif manager.has_goal():
        return {}, "", "⏸ Goal remains paused; resume it explicitly before the next scheduled fire."
    elif getattr(state, "status", None) == "done":
        return {}, "", "✓ Goal is already complete; set a new goal before the next scheduled fire."
    else:
        manager.set(goal)
        prompt = initial_prompt

    result: dict[str, Any] = {}
    response = ""
    while True:
        if is_cancelled is not None and is_cancelled():
            return result, response, _pause_interrupted_goal(manager)
        try:
            result = run_turn(prompt)
            response = response_from_result(result)
        except Exception:
            # The existing watchdog may interrupt an in-flight turn. Preserve its
            # failure reporting while preventing a later fire from resuming the goal.
            if is_cancelled is not None and is_cancelled():
                _pause_interrupted_goal(manager)
            raise
        decision = manager.evaluate_after_turn(response, user_initiated=True)
        status = str(decision.get("message") or "")
        prompt = decision.get("continuation_prompt")
        if is_cancelled is not None and is_cancelled() and manager.is_active():
            return result, response, _pause_interrupted_goal(manager)
        if not decision.get("should_continue") or not prompt:
            return result, response, status
