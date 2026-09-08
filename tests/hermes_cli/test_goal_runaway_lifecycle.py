"""Root-cause regression tests for the Goal runaway-completion lifecycle.

Two defects this file guards against (both observed on a live desktop session):

1. **Stale-pause resurrection.** ``GoalManager.evaluate_after_turn`` loads the goal once and
   mutates that in-memory snapshot; a slow judge call (10-40s) runs while the user pauses the
   goal from the Desktop control surface. The turn's final ``_save()`` is a whole-object write
   that overwrites the freshly-persisted ``paused`` back to ``active`` — the goal "resumes"
   itself and keeps dispatching continuations even though the user asked for it to stop.

2. **Judge sees only the last response, not cumulative evidence.** The strict subgoal judge is
   told to require concrete evidence (a file excerpt, an output line, a command result) for a
   criterion to count, but ``evaluate_after_turn``/``judge_goal`` only forward the last
   assistant text. When an already-complete goal (e.g. count reached 25, ``count_state.json``
   written) is re-judged, the judge can't see the tool result and keeps returning CONTINUE, so
   the loop re-dispatches forever ("Done — goal complete." repeated 20x). The judge must be
   handed the turn's concrete evidence, not just the final prose.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hermes_cli.goals import GoalManager, GoalState, load_goal, save_goal


@pytest.fixture(autouse=True)
def _isolate_goal_db(monkeypatch):
    """Give each test a fresh goal DB so a persistent SessionDB cache can't leak a paused/done
    row across tests. Mirrors tests/tui_gateway/test_goal_command.py."""
    import tempfile

    from hermes_cli import goals

    home = tempfile.mkdtemp(prefix="goal-runaway-")
    goals._DB_CACHE.clear()
    monkeypatch.setenv("HERMES_HOME", home)
    yield
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# Defect 1 — stale-pause resurrection
# ──────────────────────────────────────────────────────────────────────


def test_evaluate_after_turn_preserves_concurrent_user_pause():
    """A user pause that lands while the (slow) judge runs must survive the turn's state write.

    Regression: the turn's whole-object ``_save`` overwrote the concurrent ``paused`` back to
    ``active``, so a goal the user stopped kept dispatching continuation turns.
    """
    sid = "runaway-pause-1"
    mgr = GoalManager(sid)
    mgr.set("count to 25", max_turns=20)

    def _judge_pauses_mid_call(goal, last_response, **kwargs):
        # Simulate the user pausing from the Desktop while the judge LLM is in flight.
        GoalManager(sid).pause(reason="user-paused")
        return ("continue", "keep going", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge_pauses_mid_call):
        decision = mgr.evaluate_after_turn("worked toward it")

    persisted = load_goal(sid)
    assert persisted.status == "paused", (
        "a concurrent user pause must survive the turn write; got active"
    )
    assert persisted.paused_reason == "user-paused"
    assert decision["should_continue"] is False


def test_evaluate_after_turn_preserves_concurrent_clear():
    """A clear (goal.clear) racing the judge must not be resurrected by the turn write either."""
    sid = "runaway-clear-1"
    mgr = GoalManager(sid)
    mgr.set("do the thing", max_turns=10)

    def _judge_clears_mid_call(goal, last_response, **kwargs):
        GoalManager(sid).clear()
        return ("done", "achieved", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge_clears_mid_call):
        decision = mgr.evaluate_after_turn("finished it")

    persisted = load_goal(sid)
    assert persisted is None or persisted.status == "cleared", (
        "a concurrent clear must not be resurrected to done/active"
    )
    # The decision must not drive a continuation turn onto a cleared goal.
    assert decision["should_continue"] is False
    assert decision["status"] not in ("active", "done")


def test_done_verdict_is_not_resurrected_by_a_later_continue_judge():
    """Once a goal reaches done, a follow-up evaluation with a sloppy CONTINUE judge must not
    flip it back to active (the runaway loop's terminal-state overwrite)."""
    sid = "runaway-done-1"
    mgr = GoalManager(sid)
    mgr.set("count to 25", max_turns=20)

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "verified 25 reached", False, None, False),
    ):
        done = mgr.evaluate_after_turn("count reached 25 and file written")
    assert done["status"] == "done"

    # A fresh manager re-evaluates with a bogus CONTINUE judge: it must NOT resurrect.
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "keep going", False, None, False),
    ):
        second = mgr.evaluate_after_turn("the goal is already done")
    assert second["status"] == "done"
    assert second["should_continue"] is False


# ──────────────────────────────────────────────────────────────────────
# Defect 2 — judge must receive cumulative evidence, not just last prose
# ──────────────────────────────────────────────────────────────────────


def test_evaluate_after_turn_forwards_evidence_to_judge():
    """The cumulative evidence for a completed goal (the tool result proving count=25) must be
    passed through to the judge so it can verify completion instead of returning CONTINUE."""
    sid = "runaway-evidence-1"
    mgr = GoalManager(sid)
    mgr.set("count to 25", max_turns=20)

    captured = {}

    def _judge(goal, last_response, **kwargs):
        captured["evidence"] = kwargs.get("evidence")
        return ("done", "verified from evidence", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge):
        mgr.evaluate_after_turn(
            "Done — goal complete.",
            evidence=[
                'write_file: wrote {"count": 25} to count_state.json (verified: true)',
            ],
        )

    assert captured["evidence"] and '"count": 25' in captured["evidence"][0]


def test_judge_goal_renders_evidence_into_the_prompt():
    """judge_goal must put the evidence into the judge's user prompt so the strict subgoal
    judge can see concrete proof rather than only the assistant's assertion of completion."""
    import hermes_cli.goals as goals_mod

    seen = {}

    def _call_llm(call_llm, system_prompt, user_prompt, timeout):
        seen["user_prompt"] = user_prompt
        return '{"verdict": "done", "reason": "verified"}'

    with patch("hermes_cli.goals._call_goal_judge_llm", side_effect=_call_llm):
        verdict, reason, parse_failed, wait, transport = goals_mod.judge_goal(
            "count to 25",
            "Done — goal complete.",
            subgoals=["25 is reached"],
            evidence=['write_file: {"count": 25} written to count_state.json'],
        )

    assert verdict == "done"
    assert "count_state.json" in seen["user_prompt"]
    assert '"count": 25' in seen["user_prompt"]
    assert "Agent's most recent response" in seen["user_prompt"]


def test_evidence_is_optional_and_prompt_unchanged_without_it():
    """No evidence → the judge prompt must be byte-identical to today's (no new block, so prompt
    caching and existing prompt-shape tests are unaffected)."""
    import hermes_cli.goals as goals_mod

    seen = {}

    def _call_llm(call_llm, system_prompt, user_prompt, timeout):
        seen["user_prompt"] = user_prompt
        return '{"verdict": "continue", "reason": "more work"}'

    with patch("hermes_cli.goals._call_goal_judge_llm", side_effect=_call_llm):
        goals_mod.judge_goal("count to 25", "did some work", subgoals=["25 is reached"])

    assert "Evidence" not in seen["user_prompt"]
    assert "Agent's most recent response" in seen["user_prompt"]


def test_extract_turn_evidence_pulls_tool_results_for_the_judge():
    """The run_conversation result's tool messages (file writes, command outputs) are the concrete
    proof of completion. extract_turn_evidence must surface them (plus a trailing assistant
    summary) so the strict judge can verify instead of re-judging bare prose as CONTINUE."""
    from hermes_cli.goals import extract_turn_evidence

    result = {
        "messages": [
            {"role": "user", "content": "[Continuing toward your standing goal]"},
            {"role": "assistant", "content": "Advancing by 3: 22 -> 25.", "tool_calls": []},
            {"role": "tool", "tool_name": "write_file",
             "content": '{"bytes_written": 14, "verified": true, "resolved_path": "count_state.json"}'},
            {"role": "assistant", "content": "**Done — goal complete.** The count reached 25."},
            {"role": "user", "content": "unrelated user ping"},  # never evidence
        ],
    }
    ev = extract_turn_evidence(result)
    assert any("write_file" in e and "count_state.json" in e for e in ev)
    # User pings are never evidence; assistant prose is capped behind tool proof.
    assert not any("unrelated user ping" in e for e in ev)


def test_extract_turn_evidence_handles_non_dict_results_and_empty():
    from hermes_cli.goals import extract_turn_evidence

    assert extract_turn_evidence(None) == []
    assert extract_turn_evidence({"messages": []}) == []
    assert extract_turn_evidence({"messages": [{"role": "user", "content": "hi"}]}) == []