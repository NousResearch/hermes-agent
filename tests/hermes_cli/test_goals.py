"""Tests for hermes_cli/goals.py — SOTA 10/10 persistent cross-turn goals."""

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

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# _parse_judge_response (now in goal_judge.py)
# ──────────────────────────────────────────────────────────────────────


class TestParseJudgeResponse:
    def test_clean_json_done(self):
        from hermes_cli.goal_judge import _parse_judge_response

        verdict = _parse_judge_response('{"action":"done","completion":0.95,"progress_signal":"forward","quality_score":0.85,"reasoning":"all good"}')
        assert verdict.action == "done"
        assert verdict.completion == 0.95
        assert verdict.reasoning == "all good"

    def test_clean_json_continue(self):
        from hermes_cli.goal_judge import _parse_judge_response

        verdict = _parse_judge_response('{"action":"continue_as_is","completion":0.3,"progress_signal":"forward","quality_score":0.6,"reasoning":"more work needed"}')
        assert verdict.action == "continue_as_is"
        assert verdict.completion == 0.3

    def test_json_in_markdown_fence(self):
        from hermes_cli.goal_judge import _parse_judge_response

        raw = '```json\n{"action":"done","completion":0.95,"progress_signal":"forward","quality_score":0.9,"reasoning":"done"}\n```'
        verdict = _parse_judge_response(raw)
        assert verdict.action == "done"

    def test_json_embedded_in_prose(self):
        """Some models prefix reasoning before emitting JSON — we extract it."""
        from hermes_cli.goal_judge import _parse_judge_response

        raw = 'Looking at this... Verdict: {"action":"continue_as_is","completion":0.4,"progress_signal":"forward","quality_score":0.55,"reasoning":"partial"}'
        verdict = _parse_judge_response(raw)
        assert verdict.action == "continue_as_is"
        assert verdict.completion == 0.4

    def test_string_values(self):
        from hermes_cli.goal_judge import _parse_judge_response

        verdict = _parse_judge_response('{"action":"done","completion":"0.8","progress_signal":"forward","quality_score":"0.7","reasoning":"r"}')
        assert verdict.action == "done"
        assert verdict.completion == 0.8
        assert verdict.quality_score == 0.7

    def test_malformed_json_fails_open(self):
        """Non-JSON → default_continue with action=continue_as_is."""
        from hermes_cli.goal_judge import _parse_judge_response

        verdict = _parse_judge_response("this is not json at all")
        assert verdict.action == "continue_as_is"
        assert verdict.completion == 0.0

    def test_empty_response(self):
        from hermes_cli.goal_judge import _parse_judge_response

        verdict = _parse_judge_response("")
        assert verdict.action == "continue_as_is"
        assert verdict.completion == 0.0


# ──────────────────────────────────────────────────────────────────────
# evaluate_turn — fail-open semantics
# ──────────────────────────────────────────────────────────────────────


class TestEvaluateTurn:
    def test_empty_goal_continues(self):
        from hermes_cli.goal_judge import evaluate_turn
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad.empty()
        verdict = evaluate_turn("", "some response", pad)
        assert verdict.action == "continue_as_is"

    def test_empty_response_stalled(self):
        from hermes_cli.goal_judge import evaluate_turn
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad.empty()
        verdict = evaluate_turn("ship the thing", "", pad)
        assert verdict.action == "continue_as_is"
        assert verdict.progress_signal == "stalled"

    def test_no_aux_client_continues(self):
        """Fail-open: if no aux client, we must return continue, not done/failed."""
        from hermes_cli.goal_judge import evaluate_turn
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad.empty()
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(None, None),
        ):
            verdict = evaluate_turn("my goal", "my response", pad)
        assert verdict.action == "continue_as_is"

    def test_api_error_continues(self):
        """Judge exception → fail-open continue (don't wedge progress on judge bugs)."""
        from hermes_cli.goal_judge import evaluate_turn
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad.empty()
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict = evaluate_turn("goal", "response", pad)
        assert verdict.action == "continue_as_is"

    def test_judge_says_done(self):
        from hermes_cli.goal_judge import evaluate_turn
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad.empty()
        pad.add_artifact("/tmp/result.txt", description="final output", verified=True)
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"action":"done","completion":0.95,"progress_signal":"forward","quality_score":0.85,"reasoning":"achieved"}'
                    )
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict = evaluate_turn("goal", "agent response", pad)
        assert verdict.action == "done"
        assert verdict.completion == 0.95
        assert verdict.reasoning == "achieved"

    def test_judge_says_continue(self):
        from hermes_cli.goal_judge import evaluate_turn
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad.empty()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"action":"continue_as_is","completion":0.3,"progress_signal":"forward","quality_score":0.65,"reasoning":"not yet"}'
                    )
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict = evaluate_turn("goal", "agent response", pad)
        assert verdict.action == "continue_as_is"
        assert verdict.reasoning == "not yet"


# ──────────────────────────────────────────────────────────────────────
# Semantic loop detection
# ──────────────────────────────────────────────────────────────────────


class TestSemanticLoopDetection:
    def test_repeated_tool_intent_detected(self):
        from hermes_cli.goal_judge import _detect_semantic_loop

        calls = [
            {"function": {"name": "terminal", "arguments": '{"command": "pip install foo"}'}},
            {"function": {"name": "terminal", "arguments": '{"command": "pip install bar"}'}},
            {"function": {"name": "terminal", "arguments": '{"command": "pip install baz"}'}},
        ]
        is_loop, desc = _detect_semantic_loop(calls)
        assert is_loop is True
        assert "install" in desc

    def test_diverse_intents_not_looping(self):
        from hermes_cli.goal_judge import _detect_semantic_loop

        calls = [
            {"function": {"name": "read_file", "arguments": '{"path": "/tmp/a"}'}},
            {"function": {"name": "write_file", "arguments": '{"path": "/tmp/b"}'}},
            {"function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
        ]
        is_loop, _ = _detect_semantic_loop(calls)
        assert is_loop is False

    def test_exact_loop_detected(self):
        from hermes_cli.goal_judge import _detect_semantic_loop

        calls = [
            {"function": {"name": "terminal", "arguments": '{"command": "curl http://localhost"}'}},
            {"function": {"name": "terminal", "arguments": '{"command": "curl http://localhost"}'}},
        ]
        is_loop, desc = _detect_semantic_loop(calls)
        assert is_loop is True


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

        mgr = GoalManager(session_id="test-sid-2")
        state = mgr.set("port the thing", max_turns=5)
        assert state.goal == "port the thing"
        assert state.status == "active"
        assert state.max_turns == 5
        assert state.turns_used == 0
        assert mgr.is_active()
        assert "⊙" in mgr.status_line()
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
        """Key invariant: a second manager on the same session sees the goal."""
        from hermes_cli.goals import GoalManager

        mgr1 = GoalManager(session_id="persist-sid")
        mgr1.set("do the thing")

        mgr2 = GoalManager(session_id="persist-sid")
        assert mgr2.state is not None
        assert mgr2.state.goal == "do the thing"
        assert mgr2.is_active()

    def test_evaluate_after_turn_done(self, hermes_home):
        """Judge says done → status=done, no continuation."""
        from hermes_cli.goal_judge import evaluate_turn as real_evaluate
        from hermes_cli.goals import GoalManager
        from hermes_cli.goal_judge import JudgeVerdict

        mgr = GoalManager(session_id="eval-sid-1")
        mgr.set("ship it", max_turns=5)
        # Add a verified artifact so the verification gate allows "done"
        mgr.scratchpad.add_artifact("/tmp/shipped", description="final build", verified=True)

        fake_verdict = JudgeVerdict(
            action="done", completion=0.95, progress_signal="forward",
            quality_score=0.85, reasoning="shipped",
        )
        with patch("hermes_cli.goals.evaluate_turn", return_value=fake_verdict):
            decision = mgr.evaluate_after_turn("I shipped the feature.")

        assert decision["verdict_action"] == "done"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert decision["status"] == "done"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_continue_under_budget(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from hermes_cli.goal_judge import JudgeVerdict

        mgr = GoalManager(session_id="eval-sid-2")
        mgr.set("a long goal", max_turns=5)

        fake_verdict = JudgeVerdict(
            action="continue_as_is", completion=0.3, progress_signal="forward",
            quality_score=0.65, reasoning="more work",
        )
        with patch("hermes_cli.goals.evaluate_turn", return_value=fake_verdict):
            decision = mgr.evaluate_after_turn("made some progress")

        assert decision["verdict_action"] == "continue_as_is"
        assert decision["should_continue"] is True
        assert decision["continuation_prompt"] is not None
        assert "a long goal" in decision["continuation_prompt"]
        assert decision["status"] == "active"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_budget_exhausted(self, hermes_home):
        """When turn budget hits ceiling and progress is poor, auto-pause."""
        from hermes_cli.goals import GoalManager
        from hermes_cli.goal_judge import JudgeVerdict

        mgr = GoalManager(session_id="eval-sid-3")
        mgr.set("hard goal", max_turns=2)

        fake_verdict = JudgeVerdict(
            action="continue_as_is", completion=0.2, progress_signal="stalled",
            quality_score=0.5, reasoning="not yet",
        )
        with patch("hermes_cli.goals.evaluate_turn", return_value=fake_verdict):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.turns_used == 1
            assert mgr.state.status == "active"

            d2 = mgr.evaluate_after_turn("step 2")
            assert d2["should_continue"] is False
            assert mgr.state.status == "paused"
            assert mgr.state.turns_used == 2
            assert "budget" in (mgr.state.paused_reason or "").lower()

    def test_evaluate_after_turn_inactive(self, hermes_home):
        """evaluate_after_turn is a no-op when goal isn't active."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-4")
        d = mgr.evaluate_after_turn("anything")
        assert d["verdict_action"] == "inactive"
        assert d["should_continue"] is False

        mgr.set("a goal")
        mgr.pause()
        d2 = mgr.evaluate_after_turn("anything")
        assert d2["verdict_action"] == "inactive"
        assert d2["should_continue"] is False

    def test_continuation_prompt_shape(self, hermes_home):
        """Continuation prompt must include goal text and be safe as user-role message."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="cont-sid")
        mgr.set("port goal command to hermes")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "port goal command to hermes" in prompt
        assert prompt.strip()

    def test_evaluate_after_turn_auto_extend_budget(self, hermes_home):
        """Budget auto-extends when making good forward progress."""
        from hermes_cli.goals import GoalManager
        from hermes_cli.goal_judge import JudgeVerdict

        mgr = GoalManager(session_id="extend-sid")
        mgr.set("ship a thing", max_turns=2)

        original_max = mgr.state.max_turns
        fake_verdict = JudgeVerdict(
            action="continue_as_is", completion=0.6, progress_signal="forward",
            quality_score=0.7, reasoning="good progress",
        )
        with patch("hermes_cli.goals.evaluate_turn", return_value=fake_verdict):
            d1 = mgr.evaluate_after_turn("good progress")
            d2 = mgr.evaluate_after_turn("more progress — should trigger extend")

        # Budget should have extended since completion>=0.5 and progress=forward
        assert mgr.state.max_turns > original_max

    def test_negative_constraint_persistence(self, hermes_home):
        """Negative constraints from pivots persist across turns."""
        from hermes_cli.goals import GoalManager
        from hermes_cli.goal_judge import JudgeVerdict

        mgr = GoalManager(session_id="nc-sid")
        mgr.set("build something")

        fake_verdict = JudgeVerdict(
            action="pivot_strategy", completion=0.2, progress_signal="looping",
            quality_score=0.4, reasoning="looping",
            suggested_pivot="try X instead", negative_constraint="do NOT use Y",
        )
        with patch("hermes_cli.goals.evaluate_turn", return_value=fake_verdict):
            decision = mgr.evaluate_after_turn("tried Y, it looped")
            assert "do NOT use Y" in decision["continuation_prompt"]
            assert len(mgr.scratchpad.negative_constraints) >= 1


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
# Scratchpad DAG features
# ──────────────────────────────────────────────────────────────────────


class TestScratchpadDAG:
    def test_infer_dependencies_linear_chain(self):
        from hermes_cli.goal_scratchpad import GoalScratchpad, SubTask

        pad = GoalScratchpad(
            sub_tasks=[
                SubTask(id="st01", description="task 1"),
                SubTask(id="st02", description="task 2"),
                SubTask(id="st03", description="task 3"),
            ]
        )
        pad.infer_dependencies()
        assert pad.sub_tasks[0].depends_on == []
        assert pad.sub_tasks[1].depends_on == ["st01"]
        assert pad.sub_tasks[2].depends_on == ["st02"]

    def test_ready_tasks(self):
        from hermes_cli.goal_scratchpad import GoalScratchpad, SubTask

        pad = GoalScratchpad(
            sub_tasks=[
                SubTask(id="st01", description="t1"),
                SubTask(id="st02", description="t2", depends_on=["st01"]),
                SubTask(id="st03", description="t3"),
            ]
        )
        ready = pad.get_ready_tasks()
        assert len(ready) == 2  # st01 and st03 (no deps)
        assert {st.id for st in ready} == {"st01", "st03"}

    def test_parallel_batches(self):
        from hermes_cli.goal_scratchpad import GoalScratchpad, SubTask

        pad = GoalScratchpad(
            sub_tasks=[
                SubTask(id="st01", description="setup"),
                SubTask(id="st02", description="backend", depends_on=["st01"]),
                SubTask(id="st03", description="frontend", depends_on=["st01"]),
            ]
        )
        batches = pad.get_parallel_batches()
        assert len(batches) == 2
        # First batch: st01 (no deps)
        assert len(batches[0]) == 1
        assert batches[0][0].id == "st01"
        # Second batch: st02, st03 (both depend on st01 — parallelizable)
        assert len(batches[1]) == 2

    def test_history_recording(self):
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad()
        pad.record_verdict({"turn": 1, "completion": 0.3, "action": "continue_as_is"})
        pad.record_verdict({"turn": 2, "completion": 0.6, "action": "continue_as_is"})
        assert len(pad.history) == 2
        assert pad.history[-1]["completion"] == 0.6

    def test_error_tracking(self):
        from hermes_cli.goal_scratchpad import GoalScratchpad

        pad = GoalScratchpad()
        pad.track_error("connection refused")
        pad.track_error("connection refused")
        pad.track_error("timeout")
        assert pad.error_patterns["connection refused"] == 2
        assert pad.error_patterns["timeout"] == 1


# ──────────────────────────────────────────────────────────────────────
# Budget estimation
# ──────────────────────────────────────────────────────────────────────


class TestBudgetEstimation:
    def test_simple_goal_min_budget(self):
        from hermes_cli.goals import estimate_budget

        budget = estimate_budget("fix typo")
        assert budget >= 5  # MIN_TURNS

    def test_complex_goal_higher_budget(self):
        from hermes_cli.goals import estimate_budget

        budget = estimate_budget("build a full production web application with auth and database")
        assert budget > 20

    def test_subtask_count_affects_budget(self):
        from hermes_cli.goals import estimate_budget

        budget_no = estimate_budget("simple task", sub_task_count=0)
        budget_yes = estimate_budget("simple task", sub_task_count=10)
        assert budget_yes >= budget_no
        assert budget_yes >= 50  # 10 tasks × 5 turns each
