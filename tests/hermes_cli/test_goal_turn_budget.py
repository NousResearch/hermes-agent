"""Per-goal command budgets preserve profile defaults and persisted state."""
from dataclasses import asdict
from pathlib import Path

import pytest

from hermes_cli import goals
from hermes_cli.goal_command import dispatch_goal_command


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield goals.GoalManager(session_id="turn-budget")
    # pytest may reuse a passed case's path; retire its open DB before that happens.
    for db in goals._DB_CACHE.values():
        goals._release_session_db(db)
    goals._DB_CACHE.clear()


def dispatch(mgr, arg):
    return dispatch_goal_command(mgr, arg, authorize_gate=lambda: None)


@pytest.mark.parametrize("arg, objective", [
    ("--turns 60 ship the thing", "ship the thing"),
    ("--turns=60 ship the thing", "ship the thing"),
    ("--turns 60 draft ship the thing", "ship the thing"),
    ("--turns 60 -- pause the cron", "pause the cron"),
])
def test_set_turn_budget(mgr, monkeypatch, arg, objective):
    drafted = []
    def draft(text):
        drafted.append(text)
        return goals.GoalContract(verification="tests pass")
    monkeypatch.setattr(goals, "draft_contract", draft)
    result = dispatch(mgr, arg)
    assert not result.error
    assert (mgr.state.max_turns, mgr.state.goal) == (60, objective)
    assert "60-turn budget" in result.output
    assert result.kickoff and result.prompt == objective
    assert drafted == ([objective] if " draft " in arg else [])
    assert mgr.state.has_contract() == bool(drafted)
    assert goals.GoalManager(session_id=mgr.session_id).state.max_turns == 60
    dispatch(mgr, "a different goal")
    assert mgr.state.max_turns == 20


@pytest.mark.parametrize("arg", [
    "--turns", "--turns=", "--turns 0 x", "--turns -2 x", "--turns abc x",
    "--turns 60", "--turns 60 pause", "--turns 60 status please",
    "--turns 60 clear", "--turns 60 gate add true", "--turns 60 draft",
    "--turns 60 --",
])
def test_invalid_turn_budget_leaves_goal_untouched(mgr, arg):
    mgr.set("original", max_turns=7)
    before = asdict(mgr.state)
    result = dispatch(mgr, arg)
    assert result.error
    assert asdict(mgr.state) == before
    assert asdict(goals.load_goal(mgr.session_id)) == before
    if arg == "--turns 60":
        assert "Usage: /goal --turns N <objective>" in result.output
    if arg in {"--turns", "--turns=", "--turns 0 x", "--turns -2 x", "--turns abc x"}:
        assert "--turns must be a positive integer" in result.output


@pytest.mark.parametrize("verb", ["resume", "continue", "unpause"])
def test_resume_with_fresh_budget_after_budget_pause(mgr, monkeypatch, verb):
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("continue", "more work", False, None, False))
    mgr.set("ship the thing", max_turns=2)
    assert mgr.evaluate_after_turn("step one")["should_continue"]
    decision = mgr.evaluate_after_turn("step two")
    assert decision["status"] == "paused"
    result = dispatch(mgr, f"{verb} 40")
    assert not result.error
    assert (mgr.state.status, mgr.state.turns_used, mgr.state.max_turns) == ("active", 0, 40)
    assert "40-turn budget" in result.output
    assert result.prompt == mgr.next_continuation_prompt()
    assert result.prompt and not result.kickoff
    reloaded = goals.GoalManager(session_id=mgr.session_id).state
    assert (reloaded.status, reloaded.turns_used, reloaded.max_turns) == ("active", 0, 40)


def test_budget_pause_advertises_resume_turns(mgr, monkeypatch):
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("continue", "more work", False, None, False))
    mgr.set("ship the thing", max_turns=1)
    assert "/goal resume <turns>" in mgr.evaluate_after_turn("step")["message"]


@pytest.mark.parametrize("verb", ["resume", "continue", "unpause"])
@pytest.mark.parametrize("suffix", ["last goal", "3 bugs", "abc"])
def test_resume_words_keep_ignore_note_and_budget(mgr, verb, suffix):
    mgr.set("original", max_turns=7)
    mgr.pause()
    result = dispatch(mgr, f"{verb} {suffix}")
    assert not result.error
    assert f"(ignored {suffix!r}" in result.output
    assert (mgr.state.goal, mgr.state.status, mgr.state.max_turns) == ("original", "active", 7)


@pytest.mark.parametrize("verb", ["resume", "continue", "unpause"])
@pytest.mark.parametrize("number", ["0", "-1"])
def test_invalid_resume_budget_changes_nothing(mgr, verb, number):
    mgr.set("original", max_turns=7)
    mgr.pause()
    before = asdict(mgr.state)
    assert dispatch(mgr, f"{verb} {number}").error
    assert asdict(mgr.state) == before
    assert asdict(goals.load_goal(mgr.session_id)) == before


def test_verbatim_turns_prefix_and_profile_default(mgr):
    result = dispatch(mgr, "-- --turns 60 literal objective")
    assert not result.error
    assert (mgr.state.goal, mgr.state.max_turns) == ("--turns 60 literal objective", 20)
    custom = goals.GoalManager(session_id="custom-budget", default_max_turns=9)
    dispatch(custom, "--turns 3 small goal")
    assert custom.state.max_turns == 3
    dispatch(custom, "next goal")
    assert custom.state.max_turns == 9


def test_gateway_proxy_forwards_resume_budget(mgr, monkeypatch):
    from hermes_cli.gateway_automation import GatewayGoalManager
    mgr.set("original", max_turns=2)
    mgr.pause()
    proxy = object.__new__(GatewayGoalManager)
    def rpc(call, *args, **kwargs):
        proxy._state = getattr(mgr, call)(*args, **kwargs)
        return proxy._state
    monkeypatch.setattr(proxy, "_rpc", rpc)
    result = dispatch(proxy, "resume 40")
    assert "40-turn budget" in result.output
    assert mgr.state.max_turns == 40


@pytest.mark.parametrize("arg, on_gateway", [
    ("draft ship it", True), ("--turns 60 draft ship it", True), ("--turns=60 draft ship it", True),
    ("--turns 60 Draft ship it", True), ("DRAFT ship it", True),
    ("--turns 60 ship it", False), ("--turns 0 draft ship it", False),
    ("-- draft ship it", False), ("--turns 60 -- draft ship it", False), ("drafty ship it", False),
])
def test_classic_cli_drafts_budgeted_goals_on_the_gateway(mgr, monkeypatch, arg, on_gateway):
    from hermes_cli import gateway_automation
    from hermes_cli.goal_command import GoalCommandResult
    proxy = object.__new__(gateway_automation.GatewayGoalManager)
    proxy._state = mgr.state
    sent = []
    monkeypatch.setattr(proxy, "command", lambda a, last=None: sent.append(a) or GoalCommandResult("drafted"))
    def rpc(call, *args, **kwargs):
        proxy._state = getattr(mgr, call)(*args, **kwargs)
        return proxy._state
    monkeypatch.setattr(proxy, "_rpc", rpc)
    monkeypatch.setattr(goals, "draft_contract", lambda text: pytest.fail("drafted in the CLI process"))
    result = gateway_automation.dispatch_goal_command(proxy, arg, authorize_gate=lambda: None)
    assert sent == ([arg] if on_gateway else [])
    if arg == "--turns 60 ship it":
        assert mgr.state.max_turns == 60 and not result.error


def test_resume_budget_reply_has_one_next_step_line(mgr):
    from agent.i18n import t
    mgr.set("original", max_turns=2)
    mgr.pause()
    # CLI and TUI render the code default and print their own next-step line after it.
    assert "\n" not in dispatch(mgr, "resume 40").output
    mgr.pause()
    gateway = dispatch_goal_command(mgr, "resume 40", authorize_gate=lambda: None,
                                    render=lambda key, default, **values: t(key, **values))
    assert gateway.output.count("Continuing now") == 1 and "40-turn budget" in gateway.output
