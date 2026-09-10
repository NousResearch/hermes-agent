from unittest.mock import patch

from hermes_cli import goals


def test_judge_sees_acceptance_criteria_after_long_context():
    goal = 'Context. ' * 300 + 'Required: verify the immutable Git target.'
    contract = goals.GoalContract(verification='Evidence. ' * 300 + 'Required: independent review.')
    subgoal = 'Details. ' * 300 + 'Required: preserve unrelated work.'
    with patch.object(goals, '_call_goal_judge_llm', return_value='{"done": false, "reason": "review pending"}') as call:
        goals.judge_goal(goal, 'tests passed', contract=contract, subgoals=[subgoal])
    prompt = call.call_args.args[2]
    assert goal in prompt
    assert contract.verification in prompt
    assert subgoal in prompt
    assert '[truncated]' not in prompt


def test_oversized_acceptance_is_not_judged_from_partial_instructions():
    with patch.object(goals, '_call_goal_judge_llm') as call:
        verdict, reason, *_ = goals.judge_goal('criterion ' * 10000, 'done')
    assert verdict == 'continue'
    assert 'split into bounded tasks' in reason
    call.assert_not_called()
