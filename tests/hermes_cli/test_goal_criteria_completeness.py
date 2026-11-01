"""The judge must receive every authoritative criterion, even in long goals."""

import pytest

from hermes_cli import goals


@pytest.mark.parametrize("surface", ["goal", "contract", "subgoals", "contract_subgoals"])
def test_judge_preserves_complete_criteria(monkeypatch, surface):
    marker = "MANDATORY_FINAL_ACCEPTANCE_CRITERION"
    long_text = "A required acceptance condition. " * 170 + marker
    goal = long_text if surface == "goal" else "Finish the work"
    contract = (
        goals.GoalContract(verification=long_text, stop_when="Stop on unavailable authority")
        if surface in {"contract", "contract_subgoals"} else None
    )
    subgoals = [long_text] if surface in {"subgoals", "contract_subgoals"} else None
    captured = []

    def capture(_call, _system, prompt, _timeout):
        captured.append(prompt)
        return '{"verdict":"continue","reason":"test"}'

    monkeypatch.setattr(goals, "_call_goal_judge_llm", capture)
    goals.judge_goal(goal, "Work remains", contract=contract, subgoals=subgoals, timeout=1)

    assert len(captured) == 1
    assert long_text in captured[0]
    if contract:
        assert contract.render_block() in captured[0]
    if subgoals:
        assert captured[0].count(marker) == (2 if contract else 1)


def test_response_preview_remains_bounded(monkeypatch):
    captured = []
    def capture(_call, _system, prompt, _timeout):
        captured.append(prompt)
        return '{"verdict":"continue","reason":"test"}'
    monkeypatch.setattr(goals, "_call_goal_judge_llm", capture)
    response = "X" * (goals._JUDGE_RESPONSE_SNIPPET_CHARS * 2)
    goals.judge_goal("Finish the work", response, timeout=1)
    assert response not in captured[0]
    assert "X" * goals._JUDGE_RESPONSE_SNIPPET_CHARS in captured[0]
