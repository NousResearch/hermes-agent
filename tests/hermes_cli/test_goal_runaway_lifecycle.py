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


@pytest.mark.parametrize("field,value", [
    ("subgoals", ["new unmet criterion"]),
    ("max_turns", 40),
])
def test_stale_judge_cannot_complete_a_rescoped_goal(field, value):
    sid = "rescoped-during-judge"
    mgr = GoalManager(sid)
    mgr.set("original objective", max_turns=20)

    def judge_changes_scope(*args, **kwargs):
        fresh = load_goal(sid)
        setattr(fresh, field, value)
        save_goal(sid, fresh)
        return "done", "old scope complete", False, None, False

    with patch("hermes_cli.goals.judge_goal", side_effect=judge_changes_scope):
        decision = mgr.evaluate_after_turn("old work complete")
    assert load_goal(sid).status == "active"
    assert getattr(load_goal(sid), field) == value
    assert decision["should_continue"] is False


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


def test_extract_turn_evidence_pairs_tool_call_args_with_result():
    """The judge needs the write_file CONTENT (count reached 25), not the claim receipt.

    For write_file the concrete proof lives in the tool CALL arguments (path + content), while
    the tool result is only a receipt (``bytes_written`` / ``verified``). The extractor must pair
    each tool call with its result and surface the args, or a complete goal is re-judged CONTINUE
    forever because the "25 is reached" evidence never reaches the judge.
    """
    from hermes_cli.goals import extract_turn_evidence

    result = {
        "messages": [
            {"role": "user", "content": "[Continuing toward your standing goal]"},
            {
                "role": "assistant",
                "content": "Writing count_state.json.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {"path": "count_state.json", "content": '{"count": 25}'}
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "tool_name": "write_file",
                "content": '{"bytes_written": 14, "verified": true, "resolved_path": "count_state.json"}',
            },
            {"role": "assistant", "content": "Done — goal complete."},
        ],
    }
    ev = extract_turn_evidence(result)
    joined = "\n".join(ev)
    # The claim-receipt alone is "bytes_written": 14 — it does NOT contain the count. The proof
    # is the write_file ARGUMENT. Assert both the path and the concrete content surface.
    assert '"count": 25' in joined, (
        "judge must see the write_file CONTENT proving count=25, not just the receipt"
    )
    assert "count_state.json" in joined
    assert '"bytes_written": 14' in joined, "the paired tool result must accompany its call args"


def test_evidence_prompt_is_bounded_after_pairing_tool_args_and_results():
    """Paired evidence cannot consume an unbounded judge context on a tool-heavy turn."""
    from hermes_cli.goals import _JUDGE_EVIDENCE_MAX_CHARS, _JUDGE_EVIDENCE_MAX_ITEMS, _append_evidence_to_judge_prompt

    evidence = [f"tool {i}: {'x' * (_JUDGE_EVIDENCE_MAX_CHARS + 50)}" for i in range(10)]
    prompt = _append_evidence_to_judge_prompt("base\n", evidence)

    lines = [line for line in prompt.splitlines() if line.startswith("- tool")]
    assert len(lines) == _JUDGE_EVIDENCE_MAX_ITEMS
    assert all(len(line.removeprefix("- ")) <= _JUDGE_EVIDENCE_MAX_CHARS for line in lines)


# ──────────────────────────────────────────────────────────────────────
# Defect 3 — fail-closed turn persistence.
#
# The atomic write protecting done/continue explicitly resurrects a *deleted* goal, falls back to
# a whole-object _save on storage failure, and wholly overwrites the active row so a concurrent
# edit/replacement is clobbered. The blocked / wait / parse / transport / budget branches bypass
# the atomic write entirely and still write the stale in-memory snapshot whole-object — so a user
# pause/clear/done landing during the judge is overwritten (resurrected) by those branches too.
#
# Every branch that mutates state after a turn must re-read the FRESH persisted row inside the
# transaction, refuse to write when the row is gone / stopped / re-scoped, never fall back to a
# whole-object save on storage failure, and skip the continuation.
# ──────────────────────────────────────────────────────────────────────


def _delete_goal_row(session_id: str) -> None:
    """Physically remove the goal row (as a cleanup/migration would) while a judge is in flight."""
    from hermes_cli import goals

    db = goals._get_session_db()
    assert db is not None, "session DB unavailable in test"
    key = goals._meta_key(session_id)
    db._execute_write(lambda conn: conn.execute("DELETE FROM state_meta WHERE key = ?", (key,)))


def _concurrent_action(session_id: str, which: str) -> None:
    mgr = GoalManager(session_id)
    if which == "pause":
        mgr.pause(reason="user-paused")
    elif which == "clear":
        mgr.clear()
    elif which == "done":
        mgr.mark_done("user done")
    else:  # pragma: no cover
        raise AssertionError(which)


def test_delete_mid_judge_is_not_resurrected():
    """A goal whose row is deleted mid-judge must not be re-inserted by the turn write."""
    sid = "runaway-delete-1"
    mgr = GoalManager(sid)
    mgr.set("do the thing", max_turns=10)

    def _judge_deletes(goal, last_response, **kwargs):
        _delete_goal_row(sid)
        return ("continue", "keep going", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge_deletes):
        decision = mgr.evaluate_after_turn("worked")

    assert load_goal(sid) is None, "a goal deleted mid-judge must not be resurrected"
    assert decision["should_continue"] is False


def test_concurrent_replacement_mid_judge_is_preserved():
    """A goal replaced while the judge runs must not be clobbered by the stale turn write."""
    sid = "runaway-replace-1"
    mgr = GoalManager(sid)
    mgr.set("count to 25", max_turns=20)

    def _judge_replaces(goal, last_response, **kwargs):
        GoalManager(sid).set("pivot to 50", max_turns=20)
        return ("continue", "keep going", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge_replaces):
        decision = mgr.evaluate_after_turn("did 25")

    persisted = load_goal(sid)
    assert persisted.goal == "pivot to 50", "a concurrent replacement must survive the turn write"
    assert persisted.last_verdict is None, "a stale judge verdict must not be written onto the new goal"
    assert decision["should_continue"] is False, "a stale continue judge must not outrank a fresh replacement"


def test_concurrent_edit_mid_judge_is_preserved():
    """A goal edited while the judge runs must keep its edited text, not the stale snapshot."""
    sid = "runaway-edit-1"
    mgr = GoalManager(sid)
    mgr.set("count to 25", max_turns=20)

    def _judge_edits(goal, last_response, **kwargs):
        GoalManager(sid).update("count to 100")
        return ("continue", "keep going", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge_edits):
        decision = mgr.evaluate_after_turn("did 25")

    persisted = load_goal(sid)
    assert persisted.goal == "count to 100", "a concurrent edit must survive the turn write"
    assert decision["should_continue"] is False


def test_storage_error_fails_closed_no_unsafe_fallback():
    """On a storage error the turn must fail closed: no whole-object _save fallback (which would
    clobber concurrent user state), no continuation, and no stale turn accounting persisted."""
    from hermes_cli import goals

    sid = "runaway-storage-1"
    mgr = GoalManager(sid)
    mgr.set("do it", max_turns=10)
    db = goals._get_session_db()
    calls = {"save": 0}
    orig_save_goal = goals.save_goal

    def _spy_save(*a, **k):
        calls["save"] += 1
        return orig_save_goal(*a, **k)

    def _boom(*a, **k):
        raise RuntimeError("disk full")

    with (
        patch("hermes_cli.goals.save_goal", side_effect=_spy_save),
        patch.object(db, "mutate_meta", side_effect=_boom),
        patch(
            "hermes_cli.goals.judge_goal",
            return_value=("continue", "keep going", False, None, False),
        ),
    ):
        calls["save"] = 0
        decision = mgr.evaluate_after_turn("worked")

    assert calls["save"] == 0, "a storage failure must not fall back to a whole-object save"
    assert decision["should_continue"] is False
    persisted = load_goal(sid)
    assert persisted.status == "active"
    assert persisted.turns_used == 0, "stale turn accounting must not be written on storage failure"


@pytest.mark.parametrize(
    "judge_return",
    [
        ("blocked", "impossible", False, None, False),
        ("wait", "needs server", False, {"seconds": 30}, False),
        ("continue", "keep going", False, None, False),
        ("done", "verified", False, None, False),
    ],
    ids=["blocked", "wait", "continue", "done"],
)
@pytest.mark.parametrize("concurrent", ["pause", "clear", "done"], ids=["pause", "clear", "done"])
def test_judge_outcome_preserves_concurrent_user_stop(judge_return, concurrent):
    """Every judge outcome must preserve a concurrent user pause/clear/done that lands while the
    judge is in flight — none may resurrect the user stop or dispatch a continuation."""
    sid = f"runaway-{judge_return[0]}-{concurrent}-1"
    mgr = GoalManager(sid)
    mgr.set("do the thing", max_turns=8)

    def _judge_stops(goal, last_response, **kwargs):
        _concurrent_action(sid, concurrent)
        return judge_return

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge_stops):
        decision = mgr.evaluate_after_turn("worked")

    persisted = load_goal(sid)
    if concurrent == "pause":
        assert persisted.status == "paused"
        assert persisted.paused_reason == "user-paused", (
            "the branch's own pause must not overwrite the user's concurrent pause reason"
        )
    elif concurrent == "clear":
        assert persisted.status == "cleared", "concurrent clear must survive every judge outcome"
    else:
        assert persisted.status == "done", "concurrent done must survive every judge outcome"
    assert decision["should_continue"] is False, (
        f"{judge_return[0]} verdict must not dispatch a continuation onto a stopped goal"
    )


@pytest.mark.parametrize(
    "kind,seed_field,pre,judge_return",
    [
        ("parse", "consecutive_parse_failures", 2, ("continue", "bad output", True, None, False)),
        ("transport", "consecutive_transport_failures", 4, ("continue", "api down", False, None, True)),
    ],
    ids=["parse-budget", "transport-budget"],
)
def test_budget_pause_preserves_concurrent_done(kind, seed_field, pre, judge_return):
    """The parse/transport auto-pause branches must not resurrect a concurrent done either."""
    sid = f"runaway-budget-{kind}-1"
    mgr = GoalManager(sid)
    mgr.set("do it", max_turns=8)
    setattr(mgr._state, seed_field, pre)

    def _judge(goal, last_response, **kwargs):
        _concurrent_action(sid, "done")
        return judge_return

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge):
        decision = mgr.evaluate_after_turn("worked")

    persisted = load_goal(sid)
    assert persisted.status == "done", f"{kind}-budget pause must not resurrect concurrent done"
    assert decision["should_continue"] is False


def test_max_turns_budget_pause_preserves_concurrent_done():
    """The turn-budget auto-pause branch must not resurrect a concurrent done either."""
    sid = "runaway-budget-maxturns-1"
    mgr = GoalManager(sid)
    mgr.set("do it", max_turns=1)  # turns_used 0 -> 1 after the increment == budget

    def _judge(goal, last_response, **kwargs):
        _concurrent_action(sid, "done")
        return ("continue", "keep going", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge):
        decision = mgr.evaluate_after_turn("worked")

    persisted = load_goal(sid)
    assert persisted.status == "done", "turn-budget pause must not resurrect concurrent done"
    assert decision["should_continue"] is False


def test_gate_failure_preserves_concurrent_pause_and_never_continues(monkeypatch):
    """Gate bookkeeping is also turn persistence: a pause during a slow gate must win.

    This is deliberately outside the judge matrix.  Gates run before the judge, but used the
    same stale in-memory state and a whole-object save, so they could revive a Desktop pause.
    """
    sid = "runaway-gate-pause-1"
    mgr = GoalManager(sid)
    mgr.set("do the thing", max_turns=8)
    mgr.add_gate("false")

    def _gate_pauses(_gate):
        GoalManager(sid).pause(reason="user-paused")
        return False, 1, "still failing"

    monkeypatch.setattr("hermes_cli.goals.run_gate", _gate_pauses)
    decision = mgr.evaluate_after_turn("worked")

    persisted = load_goal(sid)
    assert persisted.status == "paused"
    assert persisted.paused_reason == "user-paused"
    assert decision["should_continue"] is False
