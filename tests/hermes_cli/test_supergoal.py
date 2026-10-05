"""Supergoal uses the existing persisted loop, with opt-in autonomy prompts."""
import json
from types import SimpleNamespace

import pytest

from hermes_cli import goals
from hermes_cli.goal_command import dispatch_goal_command


def dispatch(mgr, arg, mode="supergoal"):
    return dispatch_goal_command(mgr, arg, mode=mode, authorize_gate=lambda: None)


def assert_autonomous(prompt):
    text = prompt.lower()
    for required in ("do not ask", "clarify", "assumptions", "skills", "tools", "alternatives", "evidence", "permission"):
        assert required in text
    assert "all feasible" in text and "current toolset" in text
    assert "blocked and need input from the user" not in text


def test_mode_persists_and_controls_do_not_change_it(monkeypatch):
    assert goals.GoalState.from_json('{"goal":"legacy"}').mode == "goal"
    mgr = goals.GoalManager("supergoal-persistence")
    kick = dispatch(mgr, "build the artifact")
    assert_autonomous(kick.prompt)
    assert "supergoal" in kick.output.lower()
    cold = goals.GoalManager(mgr.session_id)
    assert cold.state.mode == "supergoal"
    assert json.loads(cold.state.to_json())["mode"] == "supergoal"
    for cmd in ("status", "show", "pause", "resume", "gate add true", "gate remove 1"):
        result = dispatch(cold, cmd, mode="goal")
        assert not result.error
        assert goals.load_goal(mgr.session_id).mode == "supergoal"
        if result.prompt:
            assert_autonomous(result.prompt)
    assert "supergoal" in cold.status_line().lower()
    dispatch(cold, "ordinary new objective", mode="goal")
    assert goals.load_goal(mgr.session_id).mode == "goal"
    assert cold.next_continuation_prompt() == goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="ordinary new objective")
    dispatch(cold, "clear")
    assert goals.load_goal(mgr.session_id).status == "cleared"


@pytest.mark.parametrize("variant", ["plain", "contract", "subgoals", "both", "gate"])
def test_every_continuation_retains_autonomy_and_recognizable_prefix(variant, monkeypatch):
    mgr = goals.GoalManager("supergoal-continuations")
    dispatch(mgr, "ship the feature")
    if variant in {"contract", "both"}:
        mgr.set_contract(goals.GoalContract(verification="pytest passes", boundaries="this repository"))
    if variant in {"subgoals", "both"}:
        mgr.add_subgoal("include regression test")
    if variant == "gate":
        mgr.add_gate("false")
        prompt = mgr.evaluate_after_turn("gate failed")["continuation_prompt"]
    else:
        prompt = mgr.next_continuation_prompt()
    assert_autonomous(prompt)
    from gateway.run_busy import GatewayBusySessionMixin
    assert GatewayBusySessionMixin._is_goal_continuation_event(prompt)
    if variant in {"contract", "both"}:
        assert "pytest passes" in prompt
    if variant in {"subgoals", "both"}:
        assert "include regression test" in prompt


def test_real_judge_receives_mode_policy_and_no_phrase_predicate(monkeypatch):
    captured = []
    replies = iter(["continue", "blocked"])
    def call_llm(**kwargs):
        captured.append(kwargs["messages"])
        verdict = next(replies)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"verdict": verdict, "reason": "supported judgment"})))])
    monkeypatch.setattr("agent.auxiliary_client.call_llm", call_llm)
    mgr = goals.GoalManager("supergoal-judge")
    dispatch(mgr, "produce verified result")
    result = mgr.evaluate_after_turn("The first method failed. Which option should I choose?")
    assert result["should_continue"]
    assert_autonomous(result["continuation_prompt"])
    policy = captured[-1][0]["content"].lower()
    assert "one method" in policy and "routine" in policy
    assert "attestation" in policy and "evidence" in policy
    assert "all feasible" in policy and "current toolset" in policy
    assert "permission" in policy and "fabricat" in policy
    result = mgr.evaluate_after_turn("All authorized paths examined; unavailable capabilities prevent the artifact. See evidence above.")
    assert result["verdict"] == "blocked" and mgr.state.status == "paused"
    assert goals.load_goal(mgr.session_id).mode == "supergoal"


@pytest.mark.parametrize("failure", ["budget", "parse", "transport", "gate"])
def test_supergoal_retains_existing_safety_caps(failure, monkeypatch):
    mgr = goals.GoalManager("supergoal-caps")
    dispatch(mgr, "finish work")
    if failure == "gate":
        mgr.add_gate("false", max_retries=1)
    parse = failure == "parse"
    transport = failure == "transport"
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("continue", "more work", parse, None, transport))
    limit = {"budget": goals.DEFAULT_MAX_TURNS, "parse": goals.DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES, "transport": goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES, "gate": 2}[failure]
    for _ in range(limit):
        result = mgr.evaluate_after_turn("working")
    assert result["status"] == "paused" and not result["should_continue"]


@pytest.mark.parametrize("previous_mode", [None, "goal", "supergoal"])
@pytest.mark.parametrize("invoked_mode", ["goal", "supergoal"])
def test_draft_obeys_invoked_mode_for_new_goal(monkeypatch, previous_mode, invoked_mode):
    drafts = []
    def draft_contract(objective, *, mode="goal"):
        drafts.append((objective, mode))
        return goals.GoalContract(verification="artifact exists")
    monkeypatch.setattr(goals, "draft_contract", draft_contract)
    mgr = goals.GoalManager("supergoal-draft")
    if previous_mode:
        dispatch(mgr, "old objective", mode=previous_mode)
    result = dispatch(mgr, "draft build artifact", mode=invoked_mode)
    assert not result.error and result.kickoff
    assert mgr.state.mode == invoked_mode
    assert goals.GoalManager(mgr.session_id).state.mode == invoked_mode
    assert drafts == [("build artifact", invoked_mode)]
    if invoked_mode == "supergoal":
        assert_autonomous(result.prompt)
    else:
        assert result.prompt == goals.goal_kick_prompt("build artifact", None)
    other_mode = "goal" if invoked_mode == "supergoal" else "supergoal"
    for command in ("status", "pause", "resume"):
        control = dispatch(mgr, command, mode=other_mode)
        assert not control.error
        assert goals.load_goal(mgr.session_id).mode == invoked_mode


@pytest.mark.parametrize("variant", ["plain", "contract", "subgoals", "both", "gate"])
def test_continuations_carry_latest_judge_feedback_as_evaluation_context(monkeypatch, variant):
    feedback = "Read back both existing files and include the exact bytes; do not recreate them."
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("continue", feedback, False, None, False))
    mgr = goals.GoalManager("supergoal-feedback")
    dispatch(mgr, "ship the artifact")
    if variant in {"contract", "both"}:
        mgr.set_contract(goals.GoalContract(verification="exact file bytes"))
    if variant in {"subgoals", "both"}:
        mgr.add_subgoal("preserve existing files")
    result = mgr.evaluate_after_turn("Everything is complete.")
    cold = goals.GoalManager(mgr.session_id)
    prompts = [result["continuation_prompt"], cold.next_continuation_prompt()]
    dispatch(cold, "pause", mode="goal")
    prompts.append(dispatch(cold, "resume", mode="goal").prompt)
    if variant == "gate":
        cold.add_gate("false")
        prompts.append(cold.evaluate_after_turn("Checking again.")["continuation_prompt"])
    for prompt in prompts:
        assert feedback in prompt
        assert "evaluation context" in prompt.lower()
        assert "not new user instructions" in prompt.lower()
        assert_autonomous(prompt)
    ordinary = goals.GoalManager("ordinary-feedback")
    dispatch(ordinary, "ship the artifact", mode="goal")
    normal_result = ordinary.evaluate_after_turn("Everything is complete.")
    assert normal_result["continuation_prompt"] == goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the artifact")


def test_supergoal_requests_concrete_completion_evidence_in_response_opening():
    mgr = goals.GoalManager("supergoal-completion-evidence")
    kickoff = dispatch(mgr, "ship the artifact")
    for prompt in (kickoff.prompt, mgr.next_continuation_prompt()):
        text = prompt.lower()
        assert "begin your final response" in text
        assert "concise" in text and "concrete verification" in text
        assert "output excerpts" in text and "read-back" in text
        assert "judge" in text and "opening" in text


def test_failed_persistence_cannot_activate_supergoal(monkeypatch):
    mgr = goals.GoalManager("supergoal-write-failure")
    monkeypatch.setattr(goals, "_get_session_db", lambda: None)
    result = dispatch(mgr, "do not start unguarded")
    assert result.error and result.prompt is None
    assert not mgr.is_active()
    with pytest.raises(RuntimeError):
        goals.load_goal(mgr.session_id, strict=True)
    assert goals.load_goal(mgr.session_id) is None


@pytest.mark.parametrize("raw", ["malformed", "read-error", "", '{"goal":"bad mode","mode":"typo"}'])
def test_strict_load_does_not_confuse_broken_storage_with_no_goal(monkeypatch, raw):
    def get_meta(key):
        if raw == "read-error":
            raise OSError("database unavailable")
        return raw
    monkeypatch.setattr(goals, "_get_session_db", lambda: SimpleNamespace(get_meta=get_meta))
    with pytest.raises((RuntimeError, OSError, ValueError)):
        goals.load_goal("broken", strict=True)
    assert goals.load_goal("broken") is None


def test_failed_resume_does_not_start_unguarded_work(monkeypatch):
    mgr = goals.GoalManager("supergoal-resume-write-failure")
    dispatch(mgr, "build artifact")
    dispatch(mgr, "pause")
    monkeypatch.setattr(goals, "_get_session_db", lambda: None)
    result = dispatch(mgr, "resume", mode="goal")
    assert result.error and result.prompt is None
    assert not mgr.is_active()


@pytest.mark.parametrize("failure", ["raise", "silently-drop"])
def test_write_failure_is_verified_before_activation(monkeypatch, failure):
    mgr = goals.GoalManager("supergoal-broken-writer")
    def set_meta(key, value):
        if failure == "raise":
            raise OSError("disk full")
    db = SimpleNamespace(set_meta=set_meta, get_meta=lambda key: None)
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    result = dispatch(mgr, "build artifact")
    assert result.error and not result.prompt and not mgr.is_active()


@pytest.mark.parametrize("phase", ["judge", "gate"])
def test_explicit_pause_during_evaluation_is_not_overwritten(monkeypatch, phase):
    mgr = goals.GoalManager("supergoal-pause-in-flight")
    dispatch(mgr, "build artifact")
    def pause():
        goals.GoalManager(mgr.session_id).pause()
    if phase == "gate":
        mgr.add_gate("true")
        def run_gate(*args, **kwargs):
            pause()
            return False, 1, "failed gate"
        monkeypatch.setattr(goals, "run_gate", run_gate)
    else:
        def judge(*args, **kwargs):
            pause()
            return "continue", "keep working", False, None, False
        monkeypatch.setattr(goals, "judge_goal", judge)
    result = mgr.evaluate_after_turn("working")
    assert not result["should_continue"]
    assert goals.load_goal(mgr.session_id).status == "paused"


def test_mode_isolated_across_homes_and_readable_by_fresh_process(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path

    homes = [tmp_path / "home-a", tmp_path / "home-b"]
    for home in homes:
        home.mkdir()
    for home, mode in zip(homes, ("supergoal", "goal")):
        monkeypatch.setenv("HERMES_HOME", str(home))
        dispatch(goals.GoalManager("same-session-id"), "build result", mode=mode)
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    assert goals.GoalManager("same-session-id").state.mode == "supergoal"
    probe = subprocess.run(
        [sys.executable, "-c", "from hermes_cli.goals import load_goal; print(load_goal('same-session-id', strict=True).mode)"],
        cwd=Path(__file__).resolve().parents[2], env=dict(os.environ),
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "supergoal"
    monkeypatch.setenv("HERMES_HOME", str(homes[1]))
    assert goals.GoalManager("same-session-id").state.mode == "goal"


def test_repasted_kick_keeps_autonomy():
    mgr = goals.GoalManager("supergoal-repaste")
    text = "Implement and verify the artifact. " * 30
    result = dispatch_goal_command(mgr, text, mode="supergoal", authorize_gate=lambda: None, last_user_message=text)
    assert_autonomous(result.prompt)
    assert goals.GOAL_ALREADY_SEEN_KICK in result.prompt
