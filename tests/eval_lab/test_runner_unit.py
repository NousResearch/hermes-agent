from agent.eval_lab.runner import LocalEvalRunner
from agent.eval_lab.schemas import EvalScenario, TrajectoryGroup


class FakeAgent:
    def __init__(self):
        self.calls = 0

    def run_conversation(self, user_message):
        self.calls += 1
        return {
            "final_response": f"answer {self.calls}",
            "messages": [
                {"role": "assistant", "content": f"answer {self.calls}"},
            ],
            "metadata": {"model": "fake"},
        }


def _scenario():
    return EvalScenario(
        id="fake_scenario",
        title="Fake scenario",
        prompt="Return a fake answer.",
        tags=["unit"],
        expected_artifacts=[],
        blocked_actions=[],
        success_criteria=["completed"],
    )


def test_runner_produces_group_with_attempts():
    agent = FakeAgent()
    runner = LocalEvalRunner(agent=agent)

    group = runner.run(_scenario(), attempt_count=2)

    assert isinstance(group, TrajectoryGroup)
    assert group.scenario_id == "fake_scenario"
    assert len(group.attempts) == 2
    assert [attempt.status for attempt in group.attempts] == ["completed", "completed"]
    assert [attempt.final_response for attempt in group.attempts] == ["answer 1", "answer 2"]
    assert group.attempts[0].steps[0].role == "assistant"
    assert agent.calls == 2


def test_runner_captures_attempt_failure_without_aborting_group():
    class FlakyAgent:
        def __init__(self):
            self.calls = 0

        def run_conversation(self, user_message):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return {"final_response": "recovered", "messages": []}

    runner = LocalEvalRunner(agent=FlakyAgent())

    group = runner.run(_scenario(), attempt_count=2)

    assert [attempt.status for attempt in group.attempts] == ["failed", "completed"]
    assert group.attempts[0].steps[0].error == "boom"
    assert group.attempts[1].final_response == "recovered"
