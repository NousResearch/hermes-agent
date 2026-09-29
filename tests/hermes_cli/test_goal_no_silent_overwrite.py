"""A control-shaped /goal must never silently replace the active goal; criteria go to subgoals.

Regression: on 2026-09-28 `/goal resume last goal` replaced a standing goal with the literal string
"resume last goal" (parse_contract treats a headline without `outcome:` as the outcome), silently —
the loss was only visible by reading ``state_meta['goal:<session>']``. Extra criteria were also
being concatenated onto the headline instead of landing on the judge-rendered subgoal surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import goals
from hermes_cli.goal_command import dispatch_goal_command, is_goal_control

GOAL = "fix everything with free deepseek agents"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()   # pre-warm: never race the loop-thread bootstrap window
    yield home
    try:
        reset_hermes_home_override(token)
    except Exception:
        pass
    goals._DB_CACHE.clear()


def _raw_row(session_id: str) -> dict:
    return json.loads(goals._get_session_db().get_meta(f"goal:{session_id}") or "{}")


def _mgr(session_id: str, goal: str | None = GOAL) -> goals.GoalManager:
    mgr = goals.GoalManager(session_id=session_id)
    if goal is not None:
        mgr.set(goal)
    return mgr


def _run(mgr: goals.GoalManager, arg: str):
    return dispatch_goal_command(mgr, arg, authorize_gate=lambda: None)


# ── Resume-shaped input resumes; it is never goal text ────────────────


def test_resume_shaped_input_leaves_an_active_goal_byte_identical(hermes_home):
    mgr = _mgr("goal-resume-words")
    before = _raw_row(mgr.session_id)

    result = _run(mgr, "resume last goal")

    after = _raw_row(mgr.session_id)
    assert after["goal"] == GOAL == before["goal"]
    assert after == before, "an active goal must not be rewritten by a resume-shaped command"
    assert "already active" in result.output
    assert result.prompt is None


def test_resume_shaped_input_resumes_a_paused_goal_with_its_text(hermes_home):
    mgr = _mgr("goal-resume-paused")
    mgr.pause(reason="user-paused")

    result = _run(mgr, "continue")

    row = _raw_row(mgr.session_id)
    assert row["goal"] == GOAL
    assert row["status"] == "active"
    assert row["paused_reason"] is None
    assert result.prompt and GOAL in result.prompt


def test_resume_shaped_input_without_a_goal_creates_no_goal(hermes_home):
    mgr = goals.GoalManager(session_id="goal-resume-nothing")

    result = _run(mgr, "resume last goal")

    assert _raw_row(mgr.session_id) == {}
    assert "No goal to resume" in result.output
    assert "'last goal' is not goal text" in result.output   # says plainly what was ignored


# ── A replace keeps the displaced text recoverable ────────────────────


def test_replacing_a_goal_keeps_the_old_text_recoverable(hermes_home):
    mgr = _mgr("goal-replace")
    result = _run(mgr, "ship the payments API")

    row = _raw_row(mgr.session_id)
    assert row["goal"] == "ship the payments API"
    assert [entry["goal"] for entry in row["superseded"]] == [GOAL]
    assert GOAL in result.output
    assert GOAL in _run(mgr, "history").output


def test_a_row_written_before_history_existed_loads_clean(hermes_home):
    raw = json.loads(goals.GoalState(goal="legacy objective").to_json())
    raw.pop("superseded")
    goals._get_session_db().set_meta("goal:goal-legacy-row", json.dumps(raw))

    state = goals.load_goal("goal-legacy-row")

    assert state.goal == "legacy objective"
    assert state.superseded == []


# ── Criteria land on the subgoal surface, not in the headline ─────────


def test_criteria_added_to_an_existing_goal_leave_the_headline_untouched(hermes_home):
    mgr = _mgr("goal-criteria-existing")

    result = _run(mgr, "criteria: free routes only\ncriteria: no paid models")

    row = _raw_row(mgr.session_id)
    assert row["goal"] == GOAL
    assert row["subgoals"] == ["free routes only", "no paid models"]
    assert mgr.state.render_subgoals_block() == "- 1. free routes only\n- 2. no paid models"
    assert "- 1. free routes only" in result.output
    assert "no paid models" in (mgr.next_continuation_prompt() or "")


def test_criteria_supplied_with_a_new_goal_become_its_subgoals(hermes_home):
    mgr = _mgr("goal-criteria-new", goal=None)

    _run(mgr, "ship the API\ncriteria: tests pass\ncriteria: no downtime")

    row = _raw_row(mgr.session_id)
    assert row["goal"] == "ship the API"
    assert row["subgoals"] == ["tests pass", "no downtime"]


def test_criteria_without_a_goal_is_refused_not_turned_into_a_goal(hermes_home):
    mgr = goals.GoalManager(session_id="goal-criteria-nothing")

    result = _run(mgr, "criteria: tests pass")

    assert _raw_row(mgr.session_id) == {}
    assert result.error is True


# ── Bare control verbs never swallow the goal either ──────────────────


def test_a_terminal_verb_with_arguments_refuses_instead_of_replacing(hermes_home):
    mgr = _mgr("goal-terminal-args")

    result = _run(mgr, "clear the flaky tests")

    row = _raw_row(mgr.session_id)
    assert result.error is True
    assert row["goal"] == GOAL
    assert row["status"] == "active"


def test_a_verbatim_prefix_sets_goal_text_that_starts_with_a_control_word(hermes_home):
    mgr = _mgr("goal-verbatim")

    _run(mgr, "-- clear the flaky tests")

    row = _raw_row(mgr.session_id)
    assert row["goal"] == "clear the flaky tests"
    assert [entry["goal"] for entry in row["superseded"]] == [GOAL]


@pytest.mark.parametrize("command,control", [
    ("resume last goal", True), ("continue", True), ("unpause it", True),
    ("criteria: free routes only", True), ("wait 4321 build", True), ("history", True),
    ("ship the API", False), ("drafting docs", False),
])
def test_is_goal_control_classifies_resume_and_criteria_input(command, control):
    """Mid-run dispatch uses this to tell a control command from a new goal."""
    assert is_goal_control(command) is control


# ── The real user-facing surfaces report it, they don't overwrite ─────


def test_cli_surface_reports_the_no_op_without_touching_the_goal(hermes_home, capsys):
    import queue

    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    mgr = _mgr("goal-cli-surface")
    cli = object.__new__(CLICommandsMixin)
    cli._get_goal_manager = lambda: mgr
    cli._pending_input = queue.Queue()
    cli.conversation_history = []

    cli._handle_goal_command("/goal resume last goal")

    assert _raw_row(mgr.session_id)["goal"] == GOAL
    assert "already active" in capsys.readouterr().out
    assert cli._pending_input.empty()


def test_gateway_surface_reports_the_no_op_without_touching_the_goal(hermes_home):
    import asyncio
    from types import SimpleNamespace

    from gateway.run_busy import GatewayBusySessionMixin
    from gateway.slash_commands_goals import GatewayGoalCommandsMixin

    class Runner(GatewayBusySessionMixin, GatewayGoalCommandsMixin):
        pass

    mgr = _mgr("goal-gateway-surface")
    runner = object.__new__(Runner)

    async def manager(event):
        return mgr, None

    async def execute(fn, *args):
        return fn(*args)

    runner._get_goal_manager_for_event = manager
    runner._run_in_executor_with_context = execute
    runner._adapter_and_key_for = lambda event: (None, None)
    runner._enqueue_goal_turn = lambda event, text, **kwargs: None
    runner._resume_caller_is_admin = lambda source: True
    event = SimpleNamespace(get_command_args=lambda: "resume last goal", source=None)

    response = asyncio.run(runner._handle_goal_command(event))

    assert _raw_row(mgr.session_id)["goal"] == GOAL
    assert "already active" in response
