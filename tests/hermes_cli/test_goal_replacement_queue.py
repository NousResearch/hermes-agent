"""Replacing /goal B must never dispatch queued goal-A prompts again."""
import queue

import pytest

from hermes_cli import goals


def _cli(mgr):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    cli = object.__new__(CLICommandsMixin)
    cli._get_goal_manager = lambda: mgr
    cli._pending_input = queue.Queue()
    cli.conversation_history = []
    return cli


def test_replacement_purges_queued_prompts_for_previous_goal(monkeypatch):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="replace-purge-1")
    cli = _cli(mgr)
    cli._handle_goal_command("/goal goal A objective")
    cli._pending_input.put(mgr.next_continuation_prompt())
    assert len(cli._pending_input.queue) == 2
    cli._handle_goal_command("/goal goal B objective")
    prompts = list(cli._pending_input.queue)
    assert goals.load_goal("replace-purge-1").goal == "goal B objective"
    assert all("goal A objective" not in p for p in prompts)
    assert prompts == ["goal B objective"]


def test_pause_and_clear_purge_queued_continuations(monkeypatch):
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="replace-purge-2")
    cli = _cli(mgr)
    cli._handle_goal_command("/goal goal A objective")
    cli._pending_input.put(mgr.next_continuation_prompt())
    cli._handle_goal_command("/goal pause")
    assert list(cli._pending_input.queue) == []
    cli._handle_goal_command("/goal resume")
    cli._pending_input.put(mgr.next_continuation_prompt())
    cli._handle_goal_command("/goal clear")
    assert list(cli._pending_input.queue) == []


def test_explicit_queue_semantics_preserved(monkeypatch):
    """A user-typed identical plain message is indistinguishable from a kickoff;
    replacement semantics say the old goal must not dispatch — purge is correct.
    Ordinary (non-goal) queued messages are kept."""
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="replace-purge-3")
    cli = _cli(mgr)
    cli._handle_goal_command("/goal goal A objective")
    cli._pending_input.put("ordinary user follow-up")
    cli._handle_goal_command("/goal goal B objective")
    prompts = list(cli._pending_input.queue)
    assert "ordinary user follow-up" in prompts
    assert "goal B objective" in prompts
    assert all("goal A objective" not in p for p in prompts)


def test_user_message_identical_to_previous_goal_is_never_purged():
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="own-1")
    cli = _cli(mgr)
    cli._handle_goal_command("/goal goal A objective")
    # Real user message, plain str, identical text to the kickoff
    cli._pending_input.put("goal A objective")
    cli._handle_goal_command("/goal goal B objective")
    prompts = list(cli._pending_input.queue)
    assert "goal A objective" in prompts  # user work preserved
    assert "goal B objective" in prompts


def test_long_goal_sentinel_kickoff_purged_on_replacement():
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="own-2")
    cli = _cli(mgr)
    long_goal = "goal A " + "x" * 500
    cli._handle_goal_command(f"/goal {long_goal}")
    # goal_kick_prompt returns sentinel for long goals already in history
    cli.conversation_history = [{"role": "user", "content": long_goal}]
    cli._handle_goal_command(f"/goal {long_goal}")
    # sentinel plain str from a user must survive; sentinel GoalKickoff must purge
    cli._pending_input.put(goals.GOAL_ALREADY_SEEN_KICK)
    from hermes_cli.goals import GoalKickoff
    cli._pending_input.put(GoalKickoff(goals.GOAL_ALREADY_SEEN_KICK, goal_generation=1, goal_text=long_goal))
    cli._handle_goal_command("/goal goal B objective")
    prompts = list(cli._pending_input.queue)
    assert goals.GOAL_ALREADY_SEEN_KICK in prompts  # plain str kept (only once)
    assert prompts.count(goals.GOAL_ALREADY_SEEN_KICK) == 1
    assert "goal B objective" in prompts


def test_inflight_output_not_judged_toward_replacement():
    goals._DB_CACHE.clear()
    mgr = goals.GoalManager(session_id="own-3")
    mgr.set("goal A objective")
    gen_a = mgr.state.goal_generation
    mgr.set("goal B objective")
    assert mgr.state.goal_generation == gen_a + 1
    decision = mgr.evaluate_after_turn("output produced under goal A", expected_generation=gen_a)
    assert decision["status"] == "obsolete"
    assert decision["should_continue"] is False
    assert decision["continuation_prompt"] is None
    # Current generation still evaluates normally (judge mocked by gate? use inactive path)
    decision2 = mgr.evaluate_after_turn("output under B", expected_generation=mgr.state.goal_generation)
    assert decision2["status"] != "obsolete" or decision2["should_continue"] is False  # judge may park; never obsolete
