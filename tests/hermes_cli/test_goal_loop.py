from hermes_cli.goal_loop import (
    get_goal_max_loops,
    make_goal_supervisor_prompt,
    make_goal_worker_prompt,
    parse_goal_supervisor_decision,
)


def test_parse_complete_decision():
    decision = parse_goal_supervisor_decision("COMPLETE: tests passed and CI is green")
    assert decision.complete is True
    assert "tests passed" in decision.feedback


def test_parse_continue_decision():
    decision = parse_goal_supervisor_decision("CONTINUE: run the failing test and fix it")
    assert decision.complete is False
    assert "failing test" in decision.feedback


def test_parse_ambiguous_decision_continues():
    decision = parse_goal_supervisor_decision("Needs more evidence")
    assert decision.complete is False
    assert decision.feedback == "Needs more evidence"


def test_goal_max_loops_env_is_safe(monkeypatch):
    monkeypatch.setenv("HERMES_GOAL_MAX_LOOPS", "not-an-int")
    assert get_goal_max_loops() == 6
    monkeypatch.setenv("HERMES_GOAL_MAX_LOOPS", "0")
    assert get_goal_max_loops() == 1
    monkeypatch.setenv("HERMES_GOAL_MAX_LOOPS", "3")
    assert get_goal_max_loops() == 3


def test_worker_prompt_first_iteration_mentions_goal():
    prompt = make_goal_worker_prompt("ship the feature", 1)
    assert "GOAL" in prompt
    assert "ship the feature" in prompt
    assert "Do not stop at a plan" in prompt


def test_worker_prompt_followup_includes_feedback_and_previous_response():
    prompt = make_goal_worker_prompt(
        "ship the feature",
        2,
        previous_response="old summary",
        supervisor_feedback="missing verification",
    )
    assert "ORIGINAL GOAL" in prompt
    assert "missing verification" in prompt
    assert "old summary" in prompt


def test_supervisor_prompt_requires_compact_decision():
    prompt = make_goal_supervisor_prompt("ship", 1, "I ran tests")
    assert "COMPLETE:" in prompt
    assert "CONTINUE:" in prompt
    assert "I ran tests" in prompt
