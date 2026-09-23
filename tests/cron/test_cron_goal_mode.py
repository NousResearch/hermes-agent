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
        state = None

        def is_active(self):
            return False

        def has_goal(self):
            return False

        def set(self, goal):
            assert goal == "ship the release"

        def evaluate_after_turn(self, response, **_kwargs):
            return next(decisions)

    def run_turn(prompt):
        prompts.append(prompt)
        return {"final_response": f"response {len(prompts)}"}

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context", run_turn=run_turn,
        response_from_result=lambda result: result["final_response"],
    )

    assert prompts == ["assembled initial context", "continue"]
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

        def has_goal(self):
            return True

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda _prompt: AssertionError(),
        response_from_result=lambda result: result["final_response"],
    )

    assert result == {}
    assert response == ""
    assert "parked" in status


def test_run_goal_turns_restores_context_after_a_wait_barrier_clears():
    prompts = []
    decisions = iter((
        {"should_continue": True, "continuation_prompt": "continuation only", "message": "keep going"},
        {"should_continue": False, "continuation_prompt": None, "message": "done"},
    ))

    class Manager:
        state = type("State", (), {"goal": "ship the release"})()

        def is_active(self):
            return True

        def is_waiting(self):
            # This goal was parked on fire A, but its barrier is now clear on
            # fire B. The fire must retain B's freshly assembled context.
            return False

        def has_goal(self):
            return True

        def evaluate_after_turn(self, _response, **_kwargs):
            return next(decisions)

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda prompt: prompts.append(prompt) or {"final_response": "response"},
        response_from_result=lambda result: result["final_response"],
    )

    assert prompts == ["assembled initial context", "continuation only"]
    assert result["final_response"] == response == "response"
    assert status == "done"


def test_run_goal_turns_preserves_a_paused_goal():
    class Manager:
        state = type("State", (), {"status": "paused"})()

        def is_active(self):
            return False

        def has_goal(self):
            return True

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda _prompt: AssertionError(),
        response_from_result=lambda result: result["final_response"],
    )

    assert result == {}
    assert response == ""
    assert "paused" in status


def test_run_goal_turns_preserves_a_completed_goal():
    class Manager:
        state = type("State", (), {"status": "done"})()

        def is_active(self):
            return False

        def has_goal(self):
            return False

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda _prompt: AssertionError(),
        response_from_result=lambda result: result["final_response"],
    )

    assert result == {}
    assert response == ""
    assert "complete" in status


def test_run_goal_turns_replaces_a_changed_active_goal_with_assembled_context():
    prompts = []

    class Manager:
        state = type("State", (), {"goal": "old goal"})()

        def is_active(self):
            return True

        def has_goal(self):
            return True

        def set(self, goal):
            assert goal == "ship the release"

        def evaluate_after_turn(self, _response, **_kwargs):
            return {"should_continue": False, "continuation_prompt": None, "message": "done"}

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda prompt: prompts.append(prompt) or {"final_response": "response"},
        response_from_result=lambda result: result["final_response"],
    )

    assert prompts == ["assembled initial context"]
    assert result["final_response"] == response == "response"
    assert status == "done"


def test_goal_execution_claim_blocks_overlap_until_the_running_claim_finishes(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    first = executions.create_execution("goal-job", source="builtin")
    second = executions.create_execution("goal-job", source="builtin")
    other = executions.create_execution("other-goal-job", source="builtin")

    assert executions.mark_execution_running(first["id"], exclusive_job=True) is not None
    assert executions.mark_execution_running(second["id"], exclusive_job=True) is None
    assert executions.mark_execution_running(other["id"], exclusive_job=True) is not None
    assert executions.finish_execution(first["id"], success=True) is not None
    assert executions.mark_execution_running(second["id"], exclusive_job=True) is not None


def test_goal_execution_adoption_blocks_overlap_until_the_running_claim_finishes(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    first = executions.create_execution("goal-job", source="builtin")
    second = executions.create_execution("goal-job", source="builtin")
    assert executions.mark_execution_handoff_pending(first["id"]) is not None
    assert executions.mark_execution_handoff_pending(second["id"]) is not None

    assert executions.adopt_claimed_execution(first["id"], exclusive_job=True) is not None
    assert executions.adopt_claimed_execution(second["id"], exclusive_job=True) is None
    assert executions.finish_execution(first["id"], success=True) is not None
    assert executions.adopt_claimed_execution(second["id"], exclusive_job=True) is not None
