"""Cron ``/goal`` prompts run a bounded, judge-terminated goal loop."""

from cron.scheduler_goal import (
    cron_goal_session_id,
    goal_prompt_from_job,
    run_goal_turns,
)


def test_goal_prompt_from_job_keeps_plain_cron_prompts_unchanged():
    assert goal_prompt_from_job({"prompt": "summarize the inbox"}) is None
    assert goal_prompt_from_job({"prompt": "/goal ship the release"}) == "ship the release"
    assert goal_prompt_from_job({"prompt": "/goal\tship the release"}) == "ship the release"
    assert goal_prompt_from_job({"prompt": "/goal"}) is None


def test_cron_goal_session_is_stable_per_job():
    assert cron_goal_session_id("nightly") == "cron-goal:nightly"


def test_run_goal_turns_continues_until_the_judge_finishes(monkeypatch):
    decisions = iter((
        {"should_continue": True, "continuation_prompt": "continue", "message": "keep going"},
        {"should_continue": False, "continuation_prompt": None, "message": "goal achieved"},
    ))
    prompts = []

    class Manager:
        def is_active(self):
            return False

        def set(self, goal):
            assert goal == "ship the release"

        def evaluate_after_turn(self, response, **_kwargs):
            return next(decisions)

    def run_turn(prompt):
        prompts.append(prompt)
        return {"final_response": f"response {len(prompts)}"}

    result, response, status = run_goal_turns(
        Manager(), "ship the release", run_turn=run_turn,
        response_from_result=lambda result: result["final_response"],
    )

    assert prompts == ["ship the release", "continue"]
    assert result["final_response"] == "response 2"
    assert response == "response 2"
    assert status == "goal achieved"


def test_run_goal_turns_does_not_burn_a_turn_while_a_goal_is_parked():
    class Manager:
        state = type("State", (), {"goal": "ship the release"})()

        def is_active(self):
            return True

        def is_waiting(self):
            return True

    result, response, status = run_goal_turns(
        Manager(), "ship the release", run_turn=lambda _prompt: AssertionError(),
        response_from_result=lambda result: result["final_response"],
    )

    assert result == {}
    assert response == ""
    assert "parked" in status
