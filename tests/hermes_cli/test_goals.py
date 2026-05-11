"""Tests for hermes_cli/goals.py — persistent cross-turn goals."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module's DB cache for each test so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# _parse_judge_response
# ──────────────────────────────────────────────────────────────────────


class TestParseJudgeResponse:
    def test_clean_json_done(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response('{"done": true, "reason": "all good"}')
        assert done is True
        assert reason == "all good"

    def test_clean_json_continue(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response('{"done": false, "reason": "more work needed"}')
        assert done is False
        assert reason == "more work needed"

    def test_json_in_markdown_fence(self):
        from hermes_cli.goals import _parse_judge_response

        raw = '```json\n{"done": true, "reason": "done"}\n```'
        done, reason, _ = _parse_judge_response(raw)
        assert done is True
        assert "done" in reason

    def test_json_embedded_in_prose(self):
        """Some models prefix reasoning before emitting JSON — we extract it."""
        from hermes_cli.goals import _parse_judge_response

        raw = 'Looking at this... the agent says X. Verdict: {"done": false, "reason": "partial"}'
        done, reason, _ = _parse_judge_response(raw)
        assert done is False
        assert reason == "partial"

    def test_string_done_values(self):
        from hermes_cli.goals import _parse_judge_response

        for s in ("true", "yes", "done", "1"):
            done, _, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert done is True
        for s in ("false", "no", "not yet"):
            done, _, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert done is False

    def test_malformed_json_fails_open(self):
        """Non-JSON → not done, with error-ish reason (so judge_goal can map to continue)."""
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response("this is not json at all")
        assert done is False
        assert reason  # non-empty

    def test_empty_response(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response("")
        assert done is False
        assert reason


# ──────────────────────────────────────────────────────────────────────
# judge_goal — fail-open semantics
# ──────────────────────────────────────────────────────────────────────


class TestJudgeGoal:
    def test_empty_goal_skipped(self):
        from hermes_cli.goals import judge_goal

        verdict, _, _ = judge_goal("", "some response")
        assert verdict == "skipped"

    def test_empty_response_continues(self):
        from hermes_cli.goals import judge_goal

        verdict, _, _ = judge_goal("ship the thing", "")
        assert verdict == "continue"

    def test_no_aux_client_continues(self):
        """Fail-open: if no aux client, we must return continue, not skipped/done."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(None, None),
        ):
            verdict, _, _ = goals.judge_goal("my goal", "my response")
        assert verdict == "continue"

    def test_api_error_continues(self):
        """Judge exception → fail-open continue (don't wedge progress on judge bugs)."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, _ = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()

    def test_judge_says_done(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"done": true, "reason": "achieved"}')
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, _ = goals.judge_goal("goal", "agent response")
        assert verdict == "done"
        assert reason == "achieved"

    def test_judge_says_continue(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"done": false, "reason": "not yet"}')
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, _ = goals.judge_goal("goal", "agent response")
        assert verdict == "continue"
        assert reason == "not yet"


# ──────────────────────────────────────────────────────────────────────
# GoalManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestGoalManager:
    def test_no_goal_initial(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-1")
        assert mgr.state is None
        assert not mgr.is_active()
        assert not mgr.has_goal()
        assert "No active goal" in mgr.status_line()

    def test_set_then_status(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-2", default_max_turns=5)
        state = mgr.set("port the thing")
        assert state.goal == "port the thing"
        assert state.status == "active"
        assert state.max_turns == 5
        assert state.turns_used == 0
        assert mgr.is_active()
        assert "active" in mgr.status_line().lower()
        assert "port the thing" in mgr.status_line()

    def test_set_rejects_empty(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-3")
        with pytest.raises(ValueError):
            mgr.set("")
        with pytest.raises(ValueError):
            mgr.set("   ")

    def test_pause_and_resume(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-4")
        mgr.set("goal text")
        mgr.pause(reason="user-paused")
        assert mgr.state.status == "paused"
        assert not mgr.is_active()
        assert mgr.has_goal()

        mgr.resume()
        assert mgr.state.status == "active"
        assert mgr.is_active()

    def test_clear(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-5")
        mgr.set("goal")
        mgr.clear()
        assert mgr.state is None
        assert not mgr.is_active()

    def test_persistence_across_managers(self, hermes_home):
        """Key invariant: a second manager on the same session sees the goal.

        This is what makes /resume work — each session rebinds its
        GoalManager and picks up the saved state.
        """
        from hermes_cli.goals import GoalManager

        mgr1 = GoalManager(session_id="persist-sid")
        mgr1.set("do the thing")

        mgr2 = GoalManager(session_id="persist-sid")
        assert mgr2.state is not None
        assert mgr2.state.goal == "do the thing"
        assert mgr2.is_active()

    def test_evaluate_after_turn_done(self, hermes_home):
        """Judge says done → status=done, no continuation.

        Skips Phase-A decompose by patching ``decompose_goal`` to return
        an empty checklist so the manager falls through to the freeform
        judge path (legacy behavior preserved when decompose is unavailable).
        """
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-1")
        mgr.set("ship it")

        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(goals, "judge_goal_freeform", return_value=("done", "shipped", False)):
            decision = mgr.evaluate_after_turn("I shipped the feature.")

        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert mgr.state.status == "done"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_continue_under_budget(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-2", default_max_turns=5)
        mgr.set("a long goal")

        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(goals, "judge_goal_freeform", return_value=("continue", "more work", False)):
            decision = mgr.evaluate_after_turn("made some progress")

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert decision["continuation_prompt"] is not None
        assert "a long goal" in decision["continuation_prompt"]
        assert mgr.state.status == "active"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_budget_exhausted(self, hermes_home):
        """When turn budget hits ceiling, auto-pause instead of continuing."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-3", default_max_turns=2)
        mgr.set("hard goal")

        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(goals, "judge_goal_freeform", return_value=("continue", "not yet", False)):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.turns_used == 1
            assert mgr.state.status == "active"

            d2 = mgr.evaluate_after_turn("step 2")
            # turns_used is now 2 which equals max_turns → paused
            assert d2["should_continue"] is False
            assert mgr.state.status == "paused"
            assert mgr.state.turns_used == 2
            assert "budget" in (mgr.state.paused_reason or "").lower()

    def test_evaluate_after_turn_inactive(self, hermes_home):
        """evaluate_after_turn is a no-op when goal isn't active."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-4")
        d = mgr.evaluate_after_turn("anything")
        assert d["verdict"] == "inactive"
        assert d["should_continue"] is False

        mgr.set("a goal")
        mgr.pause()
        d2 = mgr.evaluate_after_turn("anything")
        assert d2["verdict"] == "inactive"
        assert d2["should_continue"] is False

    def test_continuation_prompt_shape(self, hermes_home):
        """The continuation prompt must include the goal text verbatim —
        and must be safe to inject as a user-role message (prompt-cache
        invariants: no system-prompt mutation)."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="cont-sid")
        mgr.set("port goal command to hermes")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "port goal command to hermes" in prompt
        assert prompt.strip()  # non-empty


# ──────────────────────────────────────────────────────────────────────
# Smoke: CommandDef is wired
# ──────────────────────────────────────────────────────────────────────


def test_goal_command_in_registry():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("goal")
    assert cmd is not None
    assert cmd.name == "goal"


def test_goal_command_dispatches_in_cli_registry_helpers():
    """goal shows up in autocomplete / help categories alongside other Session cmds."""
    from hermes_cli.commands import COMMANDS, COMMANDS_BY_CATEGORY

    assert "/goal" in COMMANDS
    session_cmds = COMMANDS_BY_CATEGORY.get("Session", {})
    assert "/goal" in session_cmds


# ──────────────────────────────────────────────────────────────────────
# Auto-pause on consecutive judge parse failures
# ──────────────────────────────────────────────────────────────────────


class TestJudgeParseFailureAutoPause:
    """Regression: weak judge models (e.g. deepseek-v4-flash) that return
    empty strings or non-JSON prose must auto-pause the loop after N turns
    instead of burning the whole turn budget."""

    def test_parse_response_flags_empty_as_parse_failure(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, parse_failed = _parse_judge_response("")
        assert done is False
        assert parse_failed is True
        assert "empty" in reason.lower()

    def test_parse_response_flags_non_json_as_parse_failure(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, parse_failed = _parse_judge_response(
            "Let me analyze whether the goal is fully satisfied based on the agent's response..."
        )
        assert done is False
        assert parse_failed is True
        assert "not json" in reason.lower()

    def test_parse_response_clean_json_is_not_parse_failure(self):
        from hermes_cli.goals import _parse_judge_response

        done, _, parse_failed = _parse_judge_response(
            '{"done": false, "reason": "more work"}'
        )
        assert done is False
        assert parse_failed is False

    def test_api_error_does_not_count_as_parse_failure(self):
        """Transient network/API errors must not trip the auto-pause guard."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("connection reset")
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, _, parse_failed = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert parse_failed is False

    def test_empty_judge_reply_flagged_as_parse_failure(self):
        """End-to-end: judge returns empty content → parse_failed=True."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, _, parse_failed = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert parse_failed is True

    def test_auto_pause_after_three_consecutive_parse_failures(self, hermes_home):
        """N=3 consecutive parse failures → auto-pause with config pointer."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES

        assert DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES == 3
        mgr = GoalManager(session_id="parse-fail-sid-1", default_max_turns=20)
        mgr.set("do a thing")

        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(
                 goals, "judge_goal_freeform",
                 return_value=("continue", "judge returned empty response", True),
             ):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 1

            d2 = mgr.evaluate_after_turn("step 2")
            assert d2["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 2

            d3 = mgr.evaluate_after_turn("step 3")
            assert d3["should_continue"] is False
            assert d3["status"] == "paused"
            assert mgr.state.consecutive_parse_failures == 3
            # Message points at the config surface so the user can fix it.
            assert "auxiliary" in d3["message"]
            assert "goal_judge" in d3["message"]
            assert "config.yaml" in d3["message"]

    def test_parse_failure_counter_resets_on_good_reply(self, hermes_home):
        """A single good judge reply resets the counter — transient flakes don't pause."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="parse-fail-sid-2", default_max_turns=20)
        mgr.set("another goal")

        # Two parse failures…
        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(
                 goals, "judge_goal_freeform",
                 return_value=("continue", "not json", True),
             ):
            mgr.evaluate_after_turn("step 1")
            mgr.evaluate_after_turn("step 2")
            assert mgr.state.consecutive_parse_failures == 2

        # …then one clean reply resets the counter.
        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(
                 goals, "judge_goal_freeform",
                 return_value=("continue", "making progress", False),
             ):
            d = mgr.evaluate_after_turn("step 3")
            assert d["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 0

    def test_parse_failure_counter_not_incremented_by_api_errors(self, hermes_home):
        """API/transport errors must NOT count toward the auto-pause threshold."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="parse-fail-sid-3", default_max_turns=20)
        mgr.set("goal")

        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(
                 goals, "judge_goal_freeform",
                 return_value=("continue", "judge error: RuntimeError", False),
             ):
            for _ in range(5):
                d = mgr.evaluate_after_turn("still going")
                assert d["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 0
            assert mgr.state.status == "active"

    def test_consecutive_parse_failures_persists_across_goalmanager_reloads(
        self, hermes_home
    ):
        """The counter must be durable so cross-session resumes see it."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, load_goal

        mgr = GoalManager(session_id="parse-fail-sid-4", default_max_turns=20)
        mgr.set("persistent goal")

        with patch.object(goals, "decompose_goal", return_value=([], "stub")), \
             patch.object(
                 goals, "judge_goal_freeform",
                 return_value=("continue", "empty", True),
             ):
            mgr.evaluate_after_turn("r")
            mgr.evaluate_after_turn("r")

        reloaded = load_goal("parse-fail-sid-4")
        assert reloaded is not None
        assert reloaded.consecutive_parse_failures == 2


# ──────────────────────────────────────────────────────────────────────
# Checklist mode: GoalState backcompat + ChecklistItem
# ──────────────────────────────────────────────────────────────────────


class TestGoalStateBackcompat:
    def test_old_state_meta_row_loads_without_checklist_fields(self):
        """A goal serialized BEFORE the checklist fields existed must
        round-trip through GoalState.from_json with empty defaults."""
        from hermes_cli.goals import GoalState

        legacy_json = json.dumps({
            "goal": "do the thing",
            "status": "active",
            "turns_used": 3,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
            "last_verdict": "continue",
            "last_reason": "still working",
            "paused_reason": None,
            "consecutive_parse_failures": 1,
        })
        state = GoalState.from_json(legacy_json)
        assert state.goal == "do the thing"
        assert state.checklist == []
        assert state.decomposed is False

    def test_new_state_round_trip(self):
        from hermes_cli.goals import (
            ChecklistItem,
            GoalState,
            ITEM_COMPLETED,
            ITEM_PENDING,
            ADDED_BY_JUDGE,
            ADDED_BY_USER,
        )

        state = GoalState(
            goal="g",
            decomposed=True,
            checklist=[
                ChecklistItem(text="a", status=ITEM_COMPLETED,
                              added_by=ADDED_BY_JUDGE, evidence="done"),
                ChecklistItem(text="b", status=ITEM_PENDING,
                              added_by=ADDED_BY_USER),
            ],
        )
        round_tripped = GoalState.from_json(state.to_json())
        assert round_tripped.decomposed is True
        assert len(round_tripped.checklist) == 2
        assert round_tripped.checklist[0].text == "a"
        assert round_tripped.checklist[0].status == ITEM_COMPLETED
        assert round_tripped.checklist[0].evidence == "done"
        assert round_tripped.checklist[1].added_by == ADDED_BY_USER

    def test_checklist_counts_and_all_terminal(self):
        from hermes_cli.goals import (
            ChecklistItem, GoalState,
            ITEM_COMPLETED, ITEM_IMPOSSIBLE, ITEM_PENDING,
        )

        state = GoalState(
            goal="g",
            checklist=[
                ChecklistItem(text="a", status=ITEM_COMPLETED),
                ChecklistItem(text="b", status=ITEM_IMPOSSIBLE),
                ChecklistItem(text="c", status=ITEM_PENDING),
            ],
        )
        total, done, imp, pending = state.checklist_counts()
        assert (total, done, imp, pending) == (3, 1, 1, 1)
        assert state.all_terminal() is False

        state.checklist[2].status = ITEM_IMPOSSIBLE
        assert state.all_terminal() is True

    def test_empty_checklist_is_not_all_terminal(self):
        """Empty list must NOT be considered done."""
        from hermes_cli.goals import GoalState

        state = GoalState(goal="g")
        assert state.all_terminal() is False


# ──────────────────────────────────────────────────────────────────────
# Phase A: decompose
# ──────────────────────────────────────────────────────────────────────


class TestPhaseADecompose:
    def test_decompose_writes_checklist_and_marks_decomposed(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, ITEM_PENDING, ADDED_BY_JUDGE

        mgr = GoalManager(session_id="phase-a-sid-1")
        mgr.set("build a website")

        items = [{"text": "homepage exists"}, {"text": "is mobile-friendly"}]
        with patch.object(goals, "decompose_goal", return_value=(items, None)):
            d = mgr.evaluate_after_turn("(initial response)")

        assert d["verdict"] == "decompose"
        assert d["should_continue"] is True
        # Phase A produces a continuation prompt that includes the checklist.
        assert d["continuation_prompt"] is not None
        assert "Checklist progress" in d["continuation_prompt"]
        assert mgr.state.decomposed is True
        assert len(mgr.state.checklist) == 2
        assert mgr.state.checklist[0].text == "homepage exists"
        assert mgr.state.checklist[0].status == ITEM_PENDING
        assert mgr.state.checklist[0].added_by == ADDED_BY_JUDGE

    def test_decompose_only_runs_once(self, hermes_home):
        """Decomposed=True after first call. Subsequent calls go to Phase B."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="phase-a-sid-2")
        mgr.set("g")

        with patch.object(
            goals, "decompose_goal", return_value=([{"text": "x"}], None)
        ) as decompose_mock, patch.object(
            goals, "evaluate_checklist",
            return_value=({"updates": [], "new_items": [], "reason": "..."}, False),
        ) as eval_mock:
            mgr.evaluate_after_turn("turn 1")
            mgr.evaluate_after_turn("turn 2")
            mgr.evaluate_after_turn("turn 3")

        assert decompose_mock.call_count == 1
        assert eval_mock.call_count == 2

    def test_decompose_failure_falls_back_to_freeform(self, hermes_home):
        """If decompose returns no items, manager falls through to freeform judge."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="phase-a-sid-3")
        mgr.set("g")

        with patch.object(goals, "decompose_goal", return_value=([], "model error")), \
             patch.object(goals, "judge_goal_freeform",
                          return_value=("done", "shipped", False)):
            d = mgr.evaluate_after_turn("done!")

        assert d["verdict"] == "done"
        assert mgr.state.decomposed is True
        assert mgr.state.checklist == []


# ──────────────────────────────────────────────────────────────────────
# Phase B: evaluate (checklist mode)
# ──────────────────────────────────────────────────────────────────────


class TestPhaseBChecklist:
    def _make_decomposed_mgr(self, sid: str, items):
        """Helper: skip Phase A, install a decomposed checklist directly."""
        from hermes_cli.goals import (
            GoalManager, ChecklistItem, ITEM_PENDING, ADDED_BY_JUDGE,
        )
        from hermes_cli import goals as _g
        mgr = GoalManager(session_id=sid)
        mgr.set("a goal")
        mgr.state.decomposed = True
        mgr.state.checklist = [
            ChecklistItem(text=t, status=ITEM_PENDING, added_by=ADDED_BY_JUDGE)
            for t in items
        ]
        _g.save_goal(sid, mgr.state)
        return mgr

    def test_judge_flips_pending_to_completed(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import ITEM_COMPLETED, ITEM_PENDING

        mgr = self._make_decomposed_mgr("phase-b-1", ["a", "b", "c"])
        with patch.object(
            goals, "evaluate_checklist",
            return_value=(
                {
                    "updates": [
                        {"index": 0, "status": "completed", "evidence": "done"},
                        {"index": 1, "status": "completed", "evidence": "shipped"},
                    ],
                    "new_items": [],
                    "reason": "made progress",
                },
                False,
            ),
        ):
            d = mgr.evaluate_after_turn("agent did stuff")

        assert d["verdict"] == "continue"
        assert mgr.state.checklist[0].status == ITEM_COMPLETED
        assert mgr.state.checklist[0].evidence == "done"
        assert mgr.state.checklist[1].status == ITEM_COMPLETED
        assert mgr.state.checklist[2].status == ITEM_PENDING

    def test_goal_done_when_all_items_terminal(self, hermes_home):
        from hermes_cli import goals

        mgr = self._make_decomposed_mgr("phase-b-2", ["a", "b"])
        with patch.object(
            goals, "evaluate_checklist",
            return_value=(
                {
                    "updates": [
                        {"index": 0, "status": "completed", "evidence": "ok"},
                        {"index": 1, "status": "impossible", "evidence": "blocked"},
                    ],
                    "new_items": [],
                    "reason": "all done or blocked",
                },
                False,
            ),
        ):
            d = mgr.evaluate_after_turn("response")

        assert d["verdict"] == "done"
        assert d["should_continue"] is False
        assert mgr.state.status == "done"

    def test_stickiness_judge_cannot_regress_completed(self, hermes_home):
        """Once an item is completed, judge updates trying to flip it back are ignored."""
        from hermes_cli import goals
        from hermes_cli.goals import ITEM_COMPLETED

        mgr = self._make_decomposed_mgr("phase-b-stick", ["a"])
        # First turn completes item 0.
        with patch.object(
            goals, "evaluate_checklist",
            return_value=(
                {
                    "updates": [{"index": 0, "status": "completed", "evidence": "yes"}],
                    "new_items": [],
                    "reason": "done",
                },
                False,
            ),
        ):
            mgr.evaluate_after_turn("turn 1")
        assert mgr.state.checklist[0].status == ITEM_COMPLETED
        # Second turn: judge tries to send a non-terminal update.
        # _parse_evaluate_response already filters non-terminal, but at the
        # apply layer we also skip terminal items entirely. Smoke both.
        with patch.object(
            goals, "evaluate_checklist",
            return_value=(
                {
                    "updates": [{"index": 0, "status": "impossible", "evidence": "regress"}],
                    "new_items": [],
                    "reason": "trying to regress",
                },
                False,
            ),
        ):
            mgr.evaluate_after_turn("turn 2")
        # Sticky: status stays completed, evidence unchanged.
        assert mgr.state.checklist[0].status == ITEM_COMPLETED
        assert mgr.state.checklist[0].evidence == "yes"

    def test_judge_appends_new_items(self, hermes_home):
        from hermes_cli import goals

        mgr = self._make_decomposed_mgr("phase-b-new", ["a"])
        with patch.object(
            goals, "evaluate_checklist",
            return_value=(
                {
                    "updates": [],
                    "new_items": [{"text": "newly discovered"}, {"text": "also this"}],
                    "reason": "found more work",
                },
                False,
            ),
        ):
            mgr.evaluate_after_turn("response")
        assert len(mgr.state.checklist) == 3
        assert mgr.state.checklist[1].text == "newly discovered"
        assert mgr.state.checklist[1].added_by == "judge"


# ──────────────────────────────────────────────────────────────────────
# /subgoal user controls
# ──────────────────────────────────────────────────────────────────────


class TestSubgoalUserControls:
    def test_add_subgoal_appends_user_item(self, hermes_home):
        from hermes_cli.goals import GoalManager, ITEM_PENDING, ADDED_BY_USER

        mgr = GoalManager(session_id="user-sid-1")
        mgr.set("g")
        item = mgr.add_subgoal("user added")
        assert item.text == "user added"
        assert item.status == ITEM_PENDING
        assert item.added_by == ADDED_BY_USER
        assert len(mgr.state.checklist) == 1

    def test_add_subgoal_requires_active_goal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="user-sid-2")
        with pytest.raises(RuntimeError):
            mgr.add_subgoal("x")

    def test_add_subgoal_rejects_empty_text(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="user-sid-3")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.add_subgoal("   ")

    def test_mark_subgoal_uses_1_based_index(self, hermes_home):
        from hermes_cli.goals import GoalManager, ITEM_COMPLETED, ITEM_IMPOSSIBLE
        mgr = GoalManager(session_id="user-sid-4")
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        mgr.add_subgoal("c")
        mgr.mark_subgoal(2, "completed")
        mgr.mark_subgoal(3, "impossible")
        assert mgr.state.checklist[0].status == "pending"
        assert mgr.state.checklist[1].status == ITEM_COMPLETED
        assert mgr.state.checklist[2].status == ITEM_IMPOSSIBLE

    def test_mark_subgoal_rejects_invalid_index(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="user-sid-5")
        mgr.set("g")
        mgr.add_subgoal("a")
        with pytest.raises(IndexError):
            mgr.mark_subgoal(5, "completed")
        with pytest.raises(IndexError):
            mgr.mark_subgoal(0, "completed")

    def test_user_can_revert_terminal_item(self, hermes_home):
        """User mark_subgoal bypasses stickiness — only path to revert."""
        from hermes_cli.goals import GoalManager, ITEM_COMPLETED, ITEM_PENDING
        mgr = GoalManager(session_id="user-sid-6")
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.mark_subgoal(1, "completed")
        assert mgr.state.checklist[0].status == ITEM_COMPLETED
        mgr.mark_subgoal(1, "pending")
        assert mgr.state.checklist[0].status == ITEM_PENDING

    def test_remove_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="user-sid-7")
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        mgr.add_subgoal("c")
        removed = mgr.remove_subgoal(2)
        assert removed.text == "b"
        assert [it.text for it in mgr.state.checklist] == ["a", "c"]

    def test_clear_checklist_resets_decomposed(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="user-sid-8")
        mgr.set("g")
        mgr.state.decomposed = True
        mgr.add_subgoal("a")
        mgr.clear_checklist()
        assert mgr.state.checklist == []
        assert mgr.state.decomposed is False


# ──────────────────────────────────────────────────────────────────────
# Conversation dump
# ──────────────────────────────────────────────────────────────────────


class TestConversationDump:
    def test_dump_writes_messages_to_goals_dir(self, hermes_home):
        from hermes_cli.goals import dump_conversation, conversation_dump_path

        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        path = dump_conversation("dump-sid-1", msgs)
        assert path is not None
        assert path.exists()
        # Path is under <HERMES_HOME>/goals/<sid>.json
        assert path.parent.name == "goals"
        assert path.name == "dump-sid-1.json"

        loaded = json.loads(path.read_text())
        assert loaded == msgs

        # conversation_dump_path returns the same path
        assert conversation_dump_path("dump-sid-1") == path

    def test_dump_handles_unsafe_session_id(self, hermes_home):
        from hermes_cli.goals import dump_conversation

        path = dump_conversation("evil/../../sid", [{"role": "user", "content": "x"}])
        assert path is not None
        # No traversal — slashes are normalized to underscores. (Periods are
        # preserved because they're legitimate in filenames; the resulting
        # name still cannot escape <HERMES_HOME>/goals/ since path
        # separators are gone.)
        assert "/" not in path.name
        assert path.parent.name == "goals"
        # Verify the resolved path stays under the goals dir.
        from hermes_cli.goals import _goals_dump_dir
        goals_dir = _goals_dump_dir().resolve()
        assert str(path.resolve()).startswith(str(goals_dir))

    def test_dump_skips_when_messages_empty(self, hermes_home):
        from hermes_cli.goals import dump_conversation
        assert dump_conversation("sid", []) is None
        assert dump_conversation("", [{"role": "user", "content": "x"}]) is None


# ──────────────────────────────────────────────────────────────────────
# Judge read_file tool: path restriction
# ──────────────────────────────────────────────────────────────────────


class TestJudgeReadFile:
    def test_restricted_to_allowed_path(self, hermes_home, tmp_path):
        from hermes_cli.goals import _judge_read_file

        allowed = tmp_path / "allowed.json"
        allowed.write_text("hello\nworld\n")

        ok = _judge_read_file(str(allowed), allowed_path=allowed)
        loaded = json.loads(ok)
        assert loaded["content"].startswith("hello")

        # Try to read a different file.
        sneaky = tmp_path / "secret.txt"
        sneaky.write_text("nope\n")
        denied = _judge_read_file(str(sneaky), allowed_path=allowed)
        loaded = json.loads(denied)
        assert "error" in loaded
        assert "restricted" in loaded["error"]

    def test_pagination(self, hermes_home, tmp_path):
        from hermes_cli.goals import _judge_read_file
        f = tmp_path / "big.json"
        f.write_text("\n".join(f"line-{i}" for i in range(50)) + "\n")

        # offset=10, limit=5 should return lines 10..14.
        result = json.loads(_judge_read_file(str(f), offset=10, limit=5, allowed_path=f))
        assert result["returned"] == 5
        assert "line-9" in result["content"]   # 1-based: line 10 == zero-indexed 9
        assert result["next_offset"] == 15


# ──────────────────────────────────────────────────────────────────────
# Index conversion: judge emits 1-based, apply layer uses 0-based
# ──────────────────────────────────────────────────────────────────────


class TestJudgeIndexConversion:
    def test_parse_evaluate_converts_1based_to_0based(self):
        """The judge sees the checklist with 1-based indices (rendered as
        '1. [ ] foo, 2. [ ] bar'). It emits updates with those same indices.
        ``_parse_evaluate_response`` must convert them to 0-based so the
        apply layer can index ``state.checklist`` directly.
        """
        from hermes_cli.goals import _parse_evaluate_response

        raw = '''
        {"updates": [
            {"index": 1, "status": "completed", "evidence": "first item"},
            {"index": 3, "status": "impossible", "evidence": "third item"}
        ],
         "new_items": [],
         "reason": "evaluated"}
        '''
        parsed, parse_failed = _parse_evaluate_response(raw)
        assert parse_failed is False
        # 1 → 0, 3 → 2
        assert [u["index"] for u in parsed["updates"]] == [0, 2]
        assert parsed["updates"][0]["evidence"] == "first item"
        assert parsed["updates"][1]["status"] == "impossible"

    def test_full_round_trip_judge_index_to_state(self, hermes_home):
        """End-to-end: judge emits 1-based, parser converts, apply layer
        flips the right items in state.checklist."""
        from hermes_cli import goals
        from hermes_cli.goals import (
            GoalManager, ChecklistItem, ITEM_PENDING, ITEM_COMPLETED,
            ADDED_BY_JUDGE,
        )

        mgr = GoalManager(session_id="idx-round-trip")
        mgr.set("g")
        mgr.state.decomposed = True
        mgr.state.checklist = [
            ChecklistItem(text="first", status=ITEM_PENDING, added_by=ADDED_BY_JUDGE),
            ChecklistItem(text="second", status=ITEM_PENDING, added_by=ADDED_BY_JUDGE),
            ChecklistItem(text="third", status=ITEM_PENDING, added_by=ADDED_BY_JUDGE),
        ]
        goals.save_goal("idx-round-trip", mgr.state)

        # Simulate the judge returning a raw-JSON Phase-B reply via the
        # auxiliary client: the parser handles the 1-based → 0-based
        # conversion so the apply layer flips item 1 (text="first").
        class FakeMessage:
            content = '''
            {"updates": [{"index": 1, "status": "completed", "evidence": "first done"}],
             "new_items": [],
             "reason": "..."}
            '''
            tool_calls = None

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return FakeResponse()

        with patch.object(goals, "_get_judge_client", return_value=(FakeClient, "fake-model")):
            mgr.evaluate_after_turn("ran the script and item 1 is done")

        # Item 1 (text="first") should now be completed.
        assert mgr.state.checklist[0].text == "first"
        assert mgr.state.checklist[0].status == ITEM_COMPLETED
        assert mgr.state.checklist[0].evidence == "first done"
        # Other items still pending.
        assert mgr.state.checklist[1].status == ITEM_PENDING
        assert mgr.state.checklist[2].status == ITEM_PENDING


# ──────────────────────────────────────────────────────────────────────
# Compression session-rotation: goal must follow the new session_id
# ──────────────────────────────────────────────────────────────────────


class TestGoalSurvivesCompressionRotation:
    def test_load_goal_after_session_id_rotates(self, hermes_home):
        """When auto-compression rotates the session_id, the goal must be
        readable from the new session_id (forwarded by run_agent's
        _compress_context block).

        We don't run the full _compress_context method here — it has
        ~60 dependencies. Instead we mirror exactly what that block does
        with state_meta and assert the goal manager picks it up.
        """
        from hermes_cli.goals import GoalManager
        from hermes_state import SessionDB

        # Create a goal under a parent session_id.
        parent_sid = "parent-rotate-001"
        mgr = GoalManager(session_id=parent_sid)
        mgr.set("survive compression")
        assert mgr.is_active()

        # Simulate the run_agent._compress_context forwarding block:
        # read goal:<old>, write goal:<new> on the same SessionDB instance.
        db = SessionDB()
        new_sid = "child-rotate-001"
        blob = db.get_meta(f"goal:{parent_sid}")
        assert blob, "goal must be in state_meta"
        db.set_meta(f"goal:{new_sid}", blob)

        # New GoalManager for the rotated session_id should load the same goal.
        mgr2 = GoalManager(session_id=new_sid)
        assert mgr2.is_active()
        assert mgr2.state.goal == "survive compression"
        # Counters/checklist preserved verbatim.
        assert mgr2.state.turns_used == mgr.state.turns_used
        assert mgr2.state.checklist == mgr.state.checklist

    def test_no_forward_when_no_goal(self, hermes_home):
        """Forwarding is a no-op when the parent session has no goal."""
        from hermes_state import SessionDB
        from hermes_cli.goals import load_goal

        db = SessionDB()
        # Parent has no goal at all.
        assert db.get_meta("goal:parent-no-goal") is None
        blob = db.get_meta("goal:parent-no-goal")
        if blob:  # parity with production guard
            db.set_meta("goal:child-no-goal", blob)

        # Child should still have no goal.
        assert load_goal("child-no-goal") is None


# ──────────────────────────────────────────────────────────────────────
# Forced tool-call judge: submit_checklist (Phase A) + update_checklist (Phase B)
# ──────────────────────────────────────────────────────────────────────


class _FakeFn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args if isinstance(args, str) else json.dumps(args)


class _FakeToolCall:
    def __init__(self, tc_id, name, args):
        self.id = tc_id
        self.type = "function"
        self.function = _FakeFn(name, args)


class _FakeMessage:
    def __init__(self, *, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def _make_fake_client(scripted_messages):
    """Return a fake client whose .chat.completions.create() returns the
    next scripted message each call. Mutates the underlying list as a
    queue so repeat calls advance.
    """
    class FakeClient:
        class chat:
            class completions:
                _queue = list(scripted_messages)
                _calls = []

                @classmethod
                def create(cls, **kwargs):
                    cls._calls.append(kwargs)
                    if not cls._queue:
                        raise RuntimeError("scripted-message queue exhausted")
                    return _FakeResponse(cls._queue.pop(0))

    return FakeClient


class TestPhaseAToolCall:
    def test_decompose_via_submit_checklist_tool(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import decompose_goal

        msg = _FakeMessage(
            tool_calls=[_FakeToolCall(
                "tc-1", "submit_checklist",
                {"items": [{"text": "first criterion"}, {"text": "second criterion"}]},
            )],
        )
        client = _make_fake_client([msg])

        with patch.object(goals, "_get_judge_client", return_value=(client, "fake-model")):
            items, err = decompose_goal("build a website")

        assert err is None
        assert [it["text"] for it in items] == ["first criterion", "second criterion"]
        # Verify we forced the tool: tool_choice should target submit_checklist.
        call = client.chat.completions._calls[0]
        assert "tools" in call
        assert call["tools"][0]["function"]["name"] == "submit_checklist"
        # tool_choice should be either {"type":"function","function":{"name":"submit_checklist"}}
        # or "required" / "auto" if a fallback was used; primary attempt forces it.
        tc = call["tool_choice"]
        assert (
            (isinstance(tc, dict) and tc.get("function", {}).get("name") == "submit_checklist")
            or tc == "required"
            or tc == "auto"
        )

    def test_decompose_falls_back_to_json_content_when_no_tool_call(self, hermes_home):
        """If a broken provider returns content instead of a tool call, the
        backstop JSON parser still salvages a checklist."""
        from hermes_cli import goals
        from hermes_cli.goals import decompose_goal

        msg = _FakeMessage(
            content='{"checklist": [{"text": "salvaged"}]}',
            tool_calls=[],
        )
        client = _make_fake_client([msg])

        with patch.object(goals, "_get_judge_client", return_value=(client, "fake-model")):
            items, err = decompose_goal("g")

        assert err is None
        assert items == [{"text": "salvaged"}]

    def test_decompose_returns_error_when_no_tool_and_no_json(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import decompose_goal

        msg = _FakeMessage(content="I think this should be done in stages.", tool_calls=[])
        client = _make_fake_client([msg])

        with patch.object(goals, "_get_judge_client", return_value=(client, "fake-model")):
            items, err = decompose_goal("g")

        assert items == []
        assert err and "submit_checklist" in err

    def test_decompose_drops_empty_text_items(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import decompose_goal

        msg = _FakeMessage(
            tool_calls=[_FakeToolCall(
                "tc-1", "submit_checklist",
                {"items": [{"text": "ok"}, {"text": ""}, {"text": "  "}, {"text": "two"}]},
            )],
        )
        client = _make_fake_client([msg])

        with patch.object(goals, "_get_judge_client", return_value=(client, "fake-model")):
            items, err = decompose_goal("g")

        assert err is None
        assert [it["text"] for it in items] == ["ok", "two"]


class TestPhaseBToolCall:
    def test_evaluate_via_update_checklist_tool(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import evaluate_checklist, GoalState, ChecklistItem, ITEM_PENDING

        state = GoalState(
            goal="g",
            decomposed=True,
            checklist=[
                ChecklistItem(text="a", status=ITEM_PENDING),
                ChecklistItem(text="b", status=ITEM_PENDING),
            ],
        )

        msg = _FakeMessage(
            tool_calls=[_FakeToolCall(
                "tc-1", "update_checklist",
                {
                    # 1-based indices; layer converts to 0-based.
                    "updates": [{"index": 1, "status": "completed", "evidence": "did a"}],
                    "new_items": [{"text": "discovered c"}],
                    "reason": "ran a",
                },
            )],
        )
        client = _make_fake_client([msg])

        with patch.object(goals, "_get_judge_client", return_value=(client, "fake-model")):
            parsed, parse_failed = evaluate_checklist(
                state, "did the first thing", history_path=None,
            )

        assert parse_failed is False
        # Index converted 1 → 0
        assert parsed["updates"] == [{"index": 0, "status": "completed", "evidence": "did a"}]
        assert parsed["new_items"] == [{"text": "discovered c"}]
        assert parsed["reason"] == "ran a"

    def test_evaluate_does_read_file_then_update(self, hermes_home, tmp_path):
        """Phase-B tool loop: judge calls read_file once, then update_checklist."""
        from hermes_cli import goals
        from hermes_cli.goals import evaluate_checklist, GoalState, ChecklistItem, ITEM_PENDING

        # Make a real history file so the path-restriction check passes.
        hist = tmp_path / "hist.json"
        hist.write_text(json.dumps([{"role": "user", "content": "hi"}]))

        state = GoalState(
            goal="g",
            decomposed=True,
            checklist=[ChecklistItem(text="a", status=ITEM_PENDING)],
        )

        msg1 = _FakeMessage(tool_calls=[_FakeToolCall(
            "tc-1", "read_file", {"path": str(hist), "offset": 1, "limit": 100},
        )])
        msg2 = _FakeMessage(tool_calls=[_FakeToolCall(
            "tc-2", "update_checklist",
            {
                "updates": [{"index": 1, "status": "completed", "evidence": "saw it"}],
                "new_items": [],
                "reason": "verified via read_file",
            },
        )])
        client = _make_fake_client([msg1, msg2])

        with patch.object(goals, "_get_judge_client", return_value=(client, "fake-model")):
            parsed, parse_failed = evaluate_checklist(
                state, "did the thing", history_path=hist,
            )

        assert parse_failed is False
        assert parsed["updates"][0]["status"] == "completed"
        assert parsed["reason"] == "verified via read_file"
        # Two API calls — one for the read, one for the verdict.
        assert len(client.chat.completions._calls) == 2

    def test_evaluate_filters_non_terminal_status_in_tool_args(self, hermes_home):
        """update_checklist should only accept 'completed' or 'impossible' —
        any 'pending' updates are dropped at the normalize layer."""
        from hermes_cli import goals
        from hermes_cli.goals import evaluate_checklist, GoalState, ChecklistItem, ITEM_PENDING

        state = GoalState(
            goal="g",
            decomposed=True,
            checklist=[
                ChecklistItem(text="a", status=ITEM_PENDING),
                ChecklistItem(text="b", status=ITEM_PENDING),
            ],
        )
        msg = _FakeMessage(tool_calls=[_FakeToolCall(
            "tc-1", "update_checklist",
            {
                "updates": [
                    {"index": 1, "status": "completed", "evidence": "yes"},
                    {"index": 2, "status": "pending", "evidence": "skip me"},
                ],
                "new_items": [],
                "reason": "...",
            },
        )])
        client = _make_fake_client([msg])

        with patch.object(goals, "_get_judge_client", return_value=(client, "fake-model")):
            parsed, _pf = evaluate_checklist(state, "x", history_path=None)

        # Only the completed flip survives; pending update is dropped silently.
        assert len(parsed["updates"]) == 1
        assert parsed["updates"][0]["index"] == 0
