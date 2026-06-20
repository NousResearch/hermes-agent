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
        """Judge says done → status=done, no continuation."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-1")
        mgr.set("ship it")

        with patch.object(goals, "judge_goal", return_value=("done", "shipped", False)):
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

        with patch.object(goals, "judge_goal", return_value=("continue", "more work", False)):
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

        with patch.object(goals, "judge_goal", return_value=("continue", "not yet", False)):
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

        with patch.object(
            goals, "judge_goal", return_value=("continue", "judge returned empty response", True)
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
        with patch.object(
            goals, "judge_goal", return_value=("continue", "not json", True)
        ):
            mgr.evaluate_after_turn("step 1")
            mgr.evaluate_after_turn("step 2")
            assert mgr.state.consecutive_parse_failures == 2

        # …then one clean reply resets the counter.
        with patch.object(
            goals, "judge_goal", return_value=("continue", "making progress", False)
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

        with patch.object(
            goals, "judge_goal", return_value=("continue", "judge error: RuntimeError", False)
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

        with patch.object(
            goals, "judge_goal", return_value=("continue", "empty", True)
        ):
            mgr.evaluate_after_turn("r")
            mgr.evaluate_after_turn("r")

        reloaded = load_goal("parse-fail-sid-4")
        assert reloaded is not None
        assert reloaded.consecutive_parse_failures == 2


# ──────────────────────────────────────────────────────────────────────
# /subgoal — user-added criteria
# ──────────────────────────────────────────────────────────────────────


class TestGoalStateSubgoalsBackcompat:
    def test_old_state_meta_row_loads_without_subgoals(self):
        """A goal serialized BEFORE the subgoals field existed must
        round-trip with an empty list, not crash."""
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "do a thing",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
            "consecutive_parse_failures": 0,
        })
        state = GoalState.from_json(legacy)
        assert state.goal == "do a thing"
        assert state.subgoals == []

    def test_subgoals_round_trip(self):
        from hermes_cli.goals import GoalState
        state = GoalState(goal="g", subgoals=["a", "b", "c"])
        rt = GoalState.from_json(state.to_json())
        assert rt.subgoals == ["a", "b", "c"]


class TestGoalManagerSubgoals:
    def test_add_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-add")
        mgr.set("main goal")
        text = mgr.add_subgoal("  use bullet points  ")
        assert text == "use bullet points"
        assert mgr.state.subgoals == ["use bullet points"]

    def test_add_subgoal_requires_active_goal(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-noactive")
        with pytest.raises(RuntimeError):
            mgr.add_subgoal("oops")

    def test_add_empty_subgoal_rejected(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-empty")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.add_subgoal("   ")

    def test_remove_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-remove")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")
        mgr.add_subgoal("third")
        removed = mgr.remove_subgoal(2)
        assert removed == "second"
        assert mgr.state.subgoals == ["first", "third"]

    def test_remove_subgoal_out_of_range(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-oob")
        mgr.set("g")
        mgr.add_subgoal("only")
        with pytest.raises(IndexError):
            mgr.remove_subgoal(5)
        with pytest.raises(IndexError):
            mgr.remove_subgoal(0)

    def test_clear_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-clear")
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        prev = mgr.clear_subgoals()
        assert prev == 2
        assert mgr.state.subgoals == []

    def test_subgoals_persist_across_reloads(self, hermes_home):
        """Subgoals stored in SessionDB survive a fresh GoalManager."""
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-persist")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")

        mgr2 = GoalManager(session_id="sub-persist")
        assert mgr2.state.subgoals == ["first", "second"]


class TestContinuationPromptWithSubgoals:
    def test_empty_subgoals_uses_original_template(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-empty")
        mgr.set("ship the feature")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" not in prompt

    def test_with_subgoals_includes_them(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-with")
        mgr.set("ship the feature")
        mgr.add_subgoal("write tests")
        mgr.add_subgoal("update docs")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" in prompt
        assert "1. write tests" in prompt
        assert "2. update docs" in prompt


class TestJudgeGoalWithSubgoals:
    def test_judge_uses_subgoals_template_when_provided(self, hermes_home):
        """judge_goal switches templates when subgoals is non-empty.

        We don't actually call the model — we patch the aux client to
        capture the prompt that would be sent.
        """
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "all done"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return _FakeResp()

        with patch.object(goals, "get_text_auxiliary_client",
                          return_value=(_FakeClient, "fake-model"), create=True), \
             patch.object(goals, "get_auxiliary_extra_body",
                          return_value=None, create=True), \
             patch("agent.auxiliary_client.get_text_auxiliary_client",
                   return_value=(_FakeClient, "fake-model")), \
             patch("agent.auxiliary_client.get_auxiliary_extra_body",
                   return_value=None):
            verdict, reason, parse_failed = goals.judge_goal(
                "ship the feature",
                "ok shipped",
                subgoals=["write tests", "update docs"],
            )

        # The aux client was called with a prompt that includes the subgoals.
        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" in user_msg
        assert "1. write tests" in user_msg
        assert "2. update docs" in user_msg
        assert "every additional criterion" in user_msg
        assert verdict == "done"

    def test_judge_uses_original_template_when_no_subgoals(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "ok"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return _FakeResp()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                   return_value=(_FakeClient, "fake-model")), \
             patch("agent.auxiliary_client.get_auxiliary_extra_body",
                   return_value=None):
            goals.judge_goal("ship it", "done", subgoals=None)

        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" not in user_msg
        assert "ship it" in user_msg


class TestStatusLineSubgoalCount:
    def test_status_line_no_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-empty")
        mgr.set("ship it")
        line = mgr.status_line()
        assert "ship it" in line
        assert "subgoal" not in line.lower()

    def test_status_line_with_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-with")
        mgr.set("ship it")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        line = mgr.status_line()
        assert "2 subgoals" in line


# ──────────────────────────────────────────────────────────────────────
# Agent self-attestation stop sentinels (#29090)
# ──────────────────────────────────────────────────────────────────────


class TestStopSentinelDetection:
    """Unit-level coverage for ``_detect_goal_stop_sentinel``.

    The sentinel only fires when it's the LAST non-blank line of the
    response — anything else would false-positive on agents that echo
    the continuation prompt's instruction text back in their reasoning.
    """

    @pytest.mark.parametrize("response,kind,reason_fragment", [
        ("Work done.\n<<HERMES_GOAL_DONE: shipped feature X>>",
         "done", "shipped feature x"),
        ("Final reply.\n<<HERMES_GOAL_BLOCKED: need API key from user>>",
         "blocked", "need api key"),
        # Bare sentinel without reason — fallback reason supplied.
        ("All good.\n<<HERMES_GOAL_DONE>>", "done", "agent"),
        ("Cannot proceed.\n<<HERMES_GOAL_BLOCKED>>", "blocked", "blocked"),
        # Mixed casing — regex is case-insensitive.
        ("ok.\n<<hermes_goal_done: lowercase ok>>", "done", "lowercase"),
        # Extra whitespace around the reason gets stripped.
        ("ok.\n<<HERMES_GOAL_DONE:    spaced reason   >>",
         "done", "spaced reason"),
        # Whitespace after sentinel still treated as last line.
        ("ok.\n<<HERMES_GOAL_DONE: trailing newlines>>\n\n   \n",
         "done", "trailing newlines"),
    ])
    def test_detection_happy_paths(self, response, kind, reason_fragment):
        from hermes_cli.goals import _detect_goal_stop_sentinel

        result = _detect_goal_stop_sentinel(response)
        assert result is not None
        out_kind, out_reason = result
        assert out_kind == kind
        assert reason_fragment.lower() in out_reason.lower()

    @pytest.mark.parametrize("response", [
        "",
        None,
        "Just a normal response with no sentinel.",
        # Bare prose mentioning the sentinel string but not as a marker.
        "Per the instructions, I would emit HERMES_GOAL_DONE here.",
        # Sentinel in the MIDDLE of the response — agent kept talking,
        # so they didn't actually want to stop.  This is the prompt-echo
        # false-positive guard.
        ("I see the instructions: <<HERMES_GOAL_DONE: example>>\n\n"
         "Now, here is my actual work for this turn: step 1, step 2, "
         "step 3.\nI will continue next turn."),
        # Same shape with BLOCKED.
        ("<<HERMES_GOAL_BLOCKED: example>>\n\n"
         "Actually I'm not blocked — pushing on."),
    ])
    def test_detection_rejects_non_terminal_or_absent(self, response):
        from hermes_cli.goals import _detect_goal_stop_sentinel
        assert _detect_goal_stop_sentinel(response) is None

    def test_continuation_prompt_teaches_both_sentinels(self, hermes_home):
        """The continuation prompt the agent receives every loop turn
        must teach the sentinel contract — otherwise the agent can't
        emit it and the fix is dead on arrival."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sentinel-teach")
        mgr.set("some long goal")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "<<HERMES_GOAL_DONE" in prompt
        assert "<<HERMES_GOAL_BLOCKED" in prompt
        assert "LAST non-blank line" in prompt

    def test_continuation_prompt_with_subgoals_teaches_sentinels(self, hermes_home):
        """Subgoals path must keep the sentinel instruction in lockstep."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sentinel-teach-sub")
        mgr.set("some long goal")
        mgr.add_subgoal("criterion A")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "<<HERMES_GOAL_DONE" in prompt
        assert "<<HERMES_GOAL_BLOCKED" in prompt
        assert "criterion A" in prompt


class TestEvaluateAfterTurnSentinel:
    """End-to-end behaviour: when the agent emits a stop sentinel,
    ``evaluate_after_turn`` must short-circuit BEFORE the judge call
    and persist ``status="done"``.  Reproduces the #29090 fix path."""

    def test_done_sentinel_short_circuits_judge(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sent-done")
        mgr.set("gibberish goal")

        with patch.object(goals, "judge_goal") as judge_mock:
            decision = mgr.evaluate_after_turn(
                "I'm finished with the task.\n"
                "<<HERMES_GOAL_DONE: gibberish goal is unverifiable, "
                "treating as completed>>"
            )
        judge_mock.assert_not_called(), (
            "judge must NOT run when the agent emits a terminal sentinel "
            "— that's the whole point of #29090"
        )
        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert "unverifiable" in decision["reason"]
        assert mgr.state.status == "done"

    def test_blocked_sentinel_short_circuits_judge(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sent-blocked")
        mgr.set("need-user-input goal")

        with patch.object(goals, "judge_goal") as judge_mock:
            decision = mgr.evaluate_after_turn(
                "I need credentials to proceed.\n"
                "<<HERMES_GOAL_BLOCKED: missing AWS_PROFILE — please set it>>"
            )
        judge_mock.assert_not_called()
        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False
        assert "blocked" in decision["message"].lower()
        assert "AWS_PROFILE" in decision["reason"]
        assert mgr.state.status == "done"

    def test_sentinel_resets_parse_failure_counter(self, hermes_home):
        """If a flaky judge had built up consecutive parse failures, the
        sentinel landing must clear the counter so the *next* goal
        (after /goal resume + a real prompt) doesn't immediately
        auto-pause on stale state."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sent-reset")
        mgr.set("g")
        mgr.state.consecutive_parse_failures = 2  # one short of the cap
        mgr.evaluate_after_turn("ok\n<<HERMES_GOAL_DONE: yes>>")
        assert mgr.state.consecutive_parse_failures == 0

    def test_non_sentinel_response_still_calls_judge(self, hermes_home):
        """Regression guard for the existing judge path — make sure the
        new sentinel branch only triggers when the sentinel is present."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sent-passthrough")
        mgr.set("a goal")

        with patch.object(
            goals, "judge_goal", return_value=("continue", "more", False)
        ) as judge_mock:
            decision = mgr.evaluate_after_turn("Plain prose with no sentinel.")
        judge_mock.assert_called_once()
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True

    def test_sentinel_inside_response_but_not_last_line_calls_judge(
        self, hermes_home
    ):
        """The prompt-echo guard, end-to-end: an agent that quotes the
        sentinel mid-response but keeps talking afterward must NOT
        short-circuit — the judge still has the final say."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sent-echo")
        mgr.set("a goal")

        echo_response = (
            "Per the instructions, I should emit "
            "<<HERMES_GOAL_DONE: example>> when I finish.  "
            "Right now I'm still working — running tests next."
        )
        with patch.object(
            goals, "judge_goal", return_value=("continue", "more", False)
        ) as judge_mock:
            decision = mgr.evaluate_after_turn(echo_response)
        judge_mock.assert_called_once()
        assert decision["verdict"] == "continue"

    def test_sentinel_with_subgoals_active_still_short_circuits(
        self, hermes_home
    ):
        """If the user added /subgoal criteria mid-loop, the sentinel
        path still honors the agent's stop — the agent has full
        context including subgoals in the continuation prompt and is
        in the best position to attest completion."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sent-sub")
        mgr.set("base goal")
        mgr.add_subgoal("extra criterion")

        with patch.object(goals, "judge_goal") as judge_mock:
            decision = mgr.evaluate_after_turn(
                "Wrapping up.\n<<HERMES_GOAL_DONE: both criteria satisfied>>"
            )
        judge_mock.assert_not_called()
        assert decision["verdict"] == "done"
        assert mgr.state.status == "done"

    def test_sentinel_on_inactive_goal_is_inert(self, hermes_home):
        """If no goal is active, sentinel detection must not invent one
        — the early return on ``state.status != 'active'`` runs first."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="sent-inactive")
        # No goal set — state is None.
        decision = mgr.evaluate_after_turn(
            "<<HERMES_GOAL_DONE: spurious>>"
        )
        assert decision["verdict"] == "inactive"
        assert mgr.state is None


class TestRepro29090GibberishGoal:
    """End-to-end reproduction of the exact reporter scenario:
    ``/goal lsdjflasjdf;ljasdlfja;sldjfalsdjf`` (gibberish).

    Pre-fix: weak judges return ``continue`` for every turn and the
    loop spam-fires continuation prompts until the 20-turn budget
    runs out.  Post-fix: turn 2's continuation prompt teaches the
    sentinel, the agent emits ``<<HERMES_GOAL_BLOCKED: …>>`` on its
    next reply, and the loop halts in one extra turn instead of
    nineteen.  This test models that two-turn shape with a fake
    weak judge and a sentinel-emitting agent."""

    def test_full_loop_halts_within_two_turns_post_fix(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(
            session_id="repro-29090",
            default_max_turns=20,  # the reporter's "maximum limit"
        )
        mgr.set("lsdjflasjdf;ljasdlfja;sldjfalsdjf")

        # Turn 1: agent replies in normal prose, weak judge keeps
        # hedging on "continue" — pre-fix this is where the spam
        # started.  Post-fix the judge still runs (no sentinel yet) and
        # queues turn 2.
        with patch.object(
            goals, "judge_goal", return_value=("continue", "unclear", False)
        ):
            d1 = mgr.evaluate_after_turn(
                "I don't understand the goal text. Could you clarify?"
            )
        assert d1["should_continue"] is True
        assert mgr.state.turns_used == 1
        # The continuation prompt MUST teach the sentinel so turn 2 can
        # actually emit it.
        assert "<<HERMES_GOAL_DONE" in d1["continuation_prompt"]

        # Turn 2: the agent, now taught by the continuation prompt,
        # emits the sentinel and the loop halts BEFORE the 20-turn
        # budget runs out.  Crucially, the judge is NOT consulted.
        with patch.object(goals, "judge_goal") as judge_mock:
            d2 = mgr.evaluate_after_turn(
                "The goal text is gibberish — no work to do.\n"
                "<<HERMES_GOAL_BLOCKED: goal text is unintelligible, "
                "please re-send>>"
            )
        judge_mock.assert_not_called()
        assert d2["should_continue"] is False
        assert d2["verdict"] == "done"
        assert mgr.state.status == "done"
        assert mgr.state.turns_used == 2
        # Burned 2 of 20 turns, not all 20 — the regression is sealed.
        assert mgr.state.turns_used < mgr.state.max_turns
