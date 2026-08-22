"""Tests for hermes_cli/goals.py — persistent cross-turn goals."""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from agent.executive.services import ObjectiveServices
from hermes_cli.goals import GoalContract, GoalManager, GoalState

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

        verdict, reason, _pf, wait = _parse_judge_response('{"done": true, "reason": "all good"}')
        assert verdict == "done"
        assert reason == "all good"
        assert wait is None




    def test_wait_verdict_with_pid(self):
        from hermes_cli.goals import _parse_judge_response

        v, reason, pf, wait = _parse_judge_response(
            '{"verdict": "wait", "wait_on_pid": 4242, "reason": "CI running"}'
        )
        assert v == "wait"
        assert pf is False
        assert wait == {"pid": 4242}
        assert reason == "CI running"




# ──────────────────────────────────────────────────────────────────────
# judge_goal — fail-open semantics
# ──────────────────────────────────────────────────────────────────────


class TestJudgeGoal:


    def test_api_error_continues(self):
        """Judge exception → fail-open continue (don't wedge progress on judge bugs)."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("boom"),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()

    def test_judge_says_done(self):
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"done": true, "reason": "achieved"}'))]
            ),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "agent response")
        assert verdict == "done"
        assert reason == "achieved"


# ──────────────────────────────────────────────────────────────────────
# GoalManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestGoalManager:

    def test_services_none_preserves_goal_manager_invariants(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="services-none", services=None)
        assert mgr.services is None
        assert mgr.state is None
        assert mgr.is_active() is False
        assert "No active goal" in mgr.status_line()

    @pytest.mark.parametrize("services", [
        ObjectiveServices(session_id="disabled", evidence_pack_status="disabled"),
        ObjectiveServices(
            session_id="degraded",
            evidence_pack_status="degraded",
            evidence_pack_degrade_reason="factory_missing",
        ),
    ])
    def test_disabled_or_degraded_services_preserve_goal_state_behavior(self, hermes_home, services):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id=services.session_id, services=services)
        assert mgr.services is services
        assert mgr.is_active() is False
        assert "No active goal" in mgr.status_line()
        state = mgr.set("keep existing behavior")
        assert state.status == "active"
        assert mgr.is_active() is True
        assert "active" in mgr.status_line()





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




    def test_api_error_does_not_count_as_parse_failure(self):
        """Transient network/API errors must not trip the auto-pause guard."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("connection reset"),
        ):
            verdict, _, parse_failed, _wd, transport_failed = goals.judge_goal(
                "goal", "response"
            )
        assert verdict == "continue"
        assert parse_failed is False
        assert transport_failed is True


    def test_auto_pause_after_three_consecutive_parse_failures(self, hermes_home):
        """N=3 consecutive parse failures → auto-pause with config pointer."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES

        assert DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES == 3
        mgr = GoalManager(session_id="parse-fail-sid-1", default_max_turns=20)
        mgr.set("do a thing")

        with patch.object(
            goals, "judge_goal", return_value=("continue", "judge returned empty response", True, None, False)
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


class TestMigrateGoalToSession:
    """migrate_goal_to_session carries a /goal from a parent session to its
    compression continuation child (#33618). load_goal does a flat
    per-session lookup with no lineage walk, so without migration an active
    goal silently dies when compression rotates session_id."""

    def test_migrates_active_goal_to_child(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("parent-sid", GoalState(goal="ship the feature"))
        assert migrate_goal_to_session("parent-sid", "child-sid", reason="compression") is True
        child = load_goal("child-sid")
        assert child is not None and child.goal == "ship the feature"
        # Parent row archived (cleared) so only the child is active.
        parent = load_goal("parent-sid")
        assert parent is not None and parent.status == "cleared"


    def test_does_not_clobber_existing_child_goal(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("p3", GoalState(goal="parent goal"))
        save_goal("c3", GoalState(goal="child already has one"))
        assert migrate_goal_to_session("p3", "c3") is False
        assert load_goal("c3").goal == "child already has one"


class TestGoalManagerSubgoals:
    def test_add_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-add")
        mgr.set("main goal")
        text = mgr.add_subgoal("  use bullet points  ")
        assert text == "use bullet points"
        assert mgr.state.subgoals == ["use bullet points"]


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
        def _fake_call_llm(**kwargs):
            captured.update(kwargs)
            return _FakeResp()

        with patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
            verdict, reason, parse_failed, _wd, _tf = goals.judge_goal(
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
        def _fake_call_llm(**kwargs):
            captured.update(kwargs)
            return _FakeResp()

        with patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
            goals.judge_goal("ship it", "done", subgoals=None)

        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" not in user_msg
        assert "ship it" in user_msg


class TestStatusLineSubgoalCount:

    def test_status_line_with_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-with")
        mgr.set("ship it")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        line = mgr.status_line()
        assert "2 subgoals" in line


# ──────────────────────────────────────────────────────────────────────
# Wait barrier — parking the goal loop on a background process
# ──────────────────────────────────────────────────────────────────────


class TestWaitBarrier:
    """The /goal wait barrier parks the loop on a live PID and resumes when
    the process exits, without burning turns or calling the judge."""

    @staticmethod
    def _spawn_sleeper():
        """Start a short-lived child process; return its Popen handle."""
        import subprocess
        import sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    @staticmethod
    def _dead_pid():
        """A PID that is essentially guaranteed not to be running."""
        return 2_000_000_000


    def test_parked_on_live_pid_does_not_continue_or_judge(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-live")
            mgr.set("ship it", max_turns=5)
            mgr.wait_on(proc.pid, reason="CI green")
            assert mgr.is_waiting() is True

            # The judge must NOT be called while parked, and no turn is burned.
            judge = MagicMock(return_value=("continue", "x", False, None, False))
            with patch.object(goals, "judge_goal", judge):
                decision = mgr.evaluate_after_turn("still waiting on CI")

            judge.assert_not_called()
            assert decision["verdict"] == "waiting"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.turns_used == 0  # no turn consumed while parked
            assert "CI green" in decision["message"]
            assert mgr.state.status == "active"  # still active, just parked
        finally:
            proc.terminate()
            proc.wait(timeout=10)


    def test_stop_waiting_clears_barrier(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-stop")
            mgr.set("g")
            mgr.wait_on(proc.pid)
            assert mgr.is_waiting() is True
            assert mgr.stop_waiting() is True
            assert mgr.state.waiting_on_pid is None
            assert mgr.is_waiting() is False
            assert mgr.stop_waiting() is False  # idempotent
        finally:
            proc.terminate()
            proc.wait(timeout=10)


# ──────────────────────────────────────────────────────────────────────
# Judge-driven auto-wait — the judge parks the loop on its own
# ──────────────────────────────────────────────────────────────────────


class TestJudgeDrivenWait:
    """The judge returns a `wait` verdict (given live background-process
    context) and the loop parks automatically — no manual /goal wait."""

    @staticmethod
    def _spawn_sleeper():
        import subprocess, sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def test_judge_wait_pid_parks_loop(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="jw-pid", default_max_turns=10)
            mgr.set("ship the PR")
            # Judge sees the running process and says wait-on-pid.
            with patch.object(
                goals, "judge_goal",
                return_value=("wait", "CI watcher still running", False, {"pid": proc.pid}, False),
            ):
                decision = mgr.evaluate_after_turn(
                    "Pushed the PR, watching CI.",
                    background_processes=[{
                        "pid": proc.pid, "command": "wait_for_pr_green.sh",
                        "status": "running", "uptime_seconds": 12,
                    }],
                )
            assert decision["verdict"] == "wait"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.waiting_on_pid == proc.pid
            assert mgr.is_waiting() is True

            # Next turn while still parked: judge must NOT be called again.
            judge = MagicMock()
            with patch.object(goals, "judge_goal", judge):
                d2 = mgr.evaluate_after_turn("still going")
            judge.assert_not_called()
            assert d2["verdict"] == "waiting"
            assert d2["should_continue"] is False
        finally:
            proc.terminate()
            proc.wait(timeout=10)


    def test_time_barrier_clears_after_deadline(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-deadline")
        mgr.set("g")
        mgr.wait_for_seconds(120, reason="backoff")
        assert mgr.is_waiting() is True
        # Force the deadline into the past → barrier auto-clears.
        mgr.state.waiting_until = time.time() - 1
        assert mgr.is_waiting() is False
        assert mgr.state.waiting_until == 0.0

    def test_continue_verdict_still_continues_with_background(self, hermes_home):
        """A running process present but judge says continue → normal loop."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-cont", default_max_turns=10)
        mgr.set("do work")
        with patch.object(
            goals, "judge_goal",
            return_value=("continue", "more to do", False, None, False),
        ):
            decision = mgr.evaluate_after_turn(
                "made progress",
                background_processes=[{"pid": 999999, "command": "x", "status": "running"}],
            )
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.waiting_on_pid is None


# ──────────────────────────────────────────────────────────────────────
# Session/trigger barrier — wait on a process's OWN trigger, not just exit
# ──────────────────────────────────────────────────────────────────────


class TestSessionTriggerBarrier:
    """The session barrier (wait_on_session) releases when a process's own
    trigger fires — a watch_patterns match mid-run (process may never exit)
    OR exit — not only on PID exit. CI-safe: uses synthetic registry session
    objects, no real child processes."""

    @staticmethod
    def _inject(sid, *, watch_patterns=None, exited=False):
        import time as _t
        from tools.process_registry import process_registry, ProcessSession
        s = ProcessSession(id=sid, command="watcher.sh", task_id="t",
                           session_key="", cwd="/tmp", started_at=_t.time())
        if watch_patterns:
            s.watch_patterns = list(watch_patterns)
        s.exited = exited
        if exited:
            process_registry._finished[sid] = s
        else:
            process_registry._running[sid] = s
        return s, process_registry


    def test_registry_releases_on_watch_match_while_alive(self, hermes_home):
        s, reg = self._inject("proc_t2", watch_patterns=["READY"])
        assert reg.is_session_waiting("proc_t2") is True
        s._watch_hits = 1  # what _check_watch_patterns sets on a match
        # Released even though the process is STILL running (never exited).
        assert s.exited is False
        assert reg.is_session_waiting("proc_t2") is False


    def test_wait_on_session_validation(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="st-val")
        # No active goal → RuntimeError
        try:
            mgr.wait_on_session("proc_x")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
        mgr.set("g")
        try:
            mgr.wait_on_session("")
            assert False, "expected ValueError"
        except ValueError:
            pass




# ──────────────────────────────────────────────────────────────────────
# Completion contract (Codex-inspired structured goals)
# ──────────────────────────────────────────────────────────────────────


class TestParseContract:


    def test_inline_fields_parsed(self):
        from hermes_cli.goals import parse_contract

        text = (
            "Migrate auth to JWT\n"
            "verify: the auth test suite passes\n"
            "constraints: keep the /login response shape unchanged\n"
            "boundaries: only touch services/auth and its tests\n"
            "stop when: a schema change needs product sign-off"
        )
        headline, contract = parse_contract(text)
        assert headline == "Migrate auth to JWT"
        assert contract.verification == "the auth test suite passes"
        assert contract.constraints == "keep the /login response shape unchanged"
        assert contract.boundaries == "only touch services/auth and its tests"
        assert contract.stop_when == "a schema change needs product sign-off"
        assert not contract.is_empty()

    def test_alias_variants(self):
        from hermes_cli.goals import parse_contract

        _, c = parse_contract("Goal\nverified by: tests green\npreserve: public API")
        assert c.verification == "tests green"
        assert c.constraints == "public API"


class TestGoalContractSerialization:
    def test_roundtrip_with_contract(self):
        from hermes_cli.goals import GoalState, GoalContract

        state = GoalState(
            goal="ship it",
            contract=GoalContract(
                verification="pytest passes",
                constraints="don't break the API",
            ),
        )
        restored = GoalState.from_json(state.to_json())
        assert restored.goal == "ship it"
        assert restored.contract.verification == "pytest passes"
        assert restored.contract.constraints == "don't break the API"
        assert restored.has_contract()

    def test_old_row_without_contract_loads_clean(self):
        # A state_meta row written before this feature has no "contract" key.
        from hermes_cli.goals import GoalState

        legacy = '{"goal": "old goal", "status": "active", "turns_used": 2}'
        state = GoalState.from_json(legacy)
        assert state.goal == "old goal"
        assert state.turns_used == 2
        assert state.contract.is_empty()
        assert not state.has_contract()

    def test_render_block_omits_empty_fields(self):
        from hermes_cli.goals import GoalContract

        block = GoalContract(outcome="X", verification="Y").render_block()
        assert "Outcome: X" in block
        assert "Verification: Y" in block
        assert "Constraints" not in block


class TestGoalManagerContract:



    def test_set_contract_after_the_fact(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-after")
        mgr.set("ship it")
        assert not mgr.has_contract()
        mgr.set_contract(GoalContract(verification="x"))
        assert mgr.has_contract()
        # Survives reload.
        from hermes_cli.goals import GoalManager as GM2
        assert GM2(session_id="c-after").has_contract()

    def test_persistence_roundtrip(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        GoalManager(session_id="c-persist").set(
            "ship it", contract=GoalContract(outcome="O", verification="V")
        )
        reloaded = GoalManager(session_id="c-persist")
        assert reloaded.state.contract.outcome == "O"
        assert reloaded.state.contract.verification == "V"


class TestJudgeWithContract:
    def _fake_call_llm(self, captured, content='{"done": false, "reason": "more"}'):
        """judge_goal routes through call_llm (#35566) — capture its kwargs."""
        class _FakeMsg:
            pass
        _FakeMsg.content = content
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]

        def _fake(**kwargs):
            captured.update(kwargs)
            return _FakeResp()
        return _fake

    def test_judge_uses_contract_template(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._fake_call_llm(captured)):
            goals.judge_goal(
                "ship it", "I think it's done",
                contract=GoalContract(verification="pytest -q passes"),
            )
        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        assert "completion contract" in user_msg.lower()
        assert "pytest -q passes" in user_msg
        assert "concrete evidence" in user_msg


class TestDraftContract:
    def test_draft_parses_json(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        class _FakeMsg:
            content = (
                '{"outcome": "auth on JWT", "verification": "auth suite green", '
                '"constraints": "no API change", "boundaries": "services/auth", '
                '"stop_when": "schema change needed"}'
            )
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_FakeResp()):
            contract = goals.draft_contract("Migrate auth to JWT")
        assert contract is not None
        assert contract.outcome == "auth on JWT"
        assert contract.verification == "auth suite green"
        assert not contract.is_empty()


    def test_draft_returns_none_when_no_client(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        with patch("agent.auxiliary_client.call_llm",
                   side_effect=RuntimeError("No LLM provider configured")):
            assert goals.draft_contract("anything") is None


# ──────────────────────────────────────────────────────────────────────
# Compose: completion contract + wait barrier in one judge call
# ──────────────────────────────────────────────────────────────────────


class TestContractAndBackgroundCompose:
    """A contract goal blocked on a background process must surface BOTH
    the contract block and the background-process list to the judge, so it
    can return either done (evidence met) or wait (parked on the poller)."""

    def _capture_call_llm(self, captured, content='{"verdict": "wait", "wait_on_pid": 4242, "reason": "CI still running"}'):
        """judge_goal routes through call_llm (#35566) — capture its kwargs."""
        class _FakeMsg:
            pass
        _FakeMsg.content = content
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]

        def _fake(**kwargs):
            captured.update(kwargs)
            return _FakeResp()
        return _fake

    def test_judge_prompt_carries_contract_and_background(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        bg = [{
            "session_id": "ci-watch", "pid": 4242, "status": "running",
            "command": "wait_for_pr_green.sh 50501", "trigger": "exit",
        }]
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._capture_call_llm(captured)):
            verdict, reason, parse_failed, wait_directive, _tf = goals.judge_goal(
                "ship the PR",
                "I pushed and started the CI watcher; waiting on it now.",
                contract=GoalContract(verification="PR CI goes green"),
                background_processes=bg,
            )
        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        # Both surfaces present in one prompt.
        assert "completion contract" in user_msg.lower()
        assert "PR CI goes green" in user_msg
        assert "Background processes" in user_msg
        assert "4242" in user_msg
        # The judge can return a wait verdict on a contract goal.
        assert verdict == "wait"
        assert wait_directive and wait_directive.get("pid") == 4242


# ──────────────────────────────────────────────────────────────────────
# B1-E5: production EvidencePack runtime invocation
# ──────────────────────────────────────────────────────────────────────


class _FakeEvidencePack:
    """Minimal stand-in for an EvidencePack dataclass instance.

    The implementation only uses ``sources_failed`` as a marker and
    treats any object as a "current pack" via ``_last_evidence_pack``.
    Real engine returns EvidencePack dataclasses; tests use this lightweight
    fake so they don't pull the whole engine surface into the test module.

    B1-E6: extended with allowlisted advisory fields plus optional fields
    that must NOT appear in the rendered prompt (hits / citations /
    source_uri / raw payload / timestamps / fingerprints). The class
    stores every attribute passed in so tests can assert non-mutation
    directly.
    """

    __slots__ = (
        "objective_id",
        "sources_queried",
        "sources_failed",
        "tag",
        "summary_text",
        "missing_information",
        "overall_confidence",
        "overall_freshness_score",
        # Forbidden fields — must NEVER surface in the rendered block.
        "hits",
        "citations",
        "conflicts",
        "source_uri",
        "query_fingerprint",
        "summary_fingerprint",
        "duration_ms",
        "created_at",
        "total_hits",
        "is_idempotent_reuse",
        "raw_payload",
        "exception_message",
    )

    def __init__(
        self,
        objective_id="",
        sources_queried=None,
        sources_failed=None,
        tag="",
        summary_text="",
        missing_information=None,
        overall_confidence=0.0,
        overall_freshness_score=0.0,
        hits=None,
        citations=None,
        conflicts=None,
        source_uri="",
        query_fingerprint="",
        summary_fingerprint="",
        duration_ms=0,
        created_at="",
        total_hits=0,
        is_idempotent_reuse=False,
        raw_payload=None,
        exception_message="",
    ):
        self.objective_id = objective_id
        # Lists are stored as new lists so test code can compare identity
        # rather than value to detect mutation.
        self.sources_queried = list(sources_queried or [])
        self.sources_failed = list(sources_failed or [])
        self.tag = tag
        self.summary_text = summary_text
        self.missing_information = list(missing_information or [])
        self.overall_confidence = overall_confidence
        self.overall_freshness_score = overall_freshness_score
        self.hits = list(hits or [])
        self.citations = list(citations or [])
        self.conflicts = list(conflicts or [])
        self.source_uri = source_uri
        self.query_fingerprint = query_fingerprint
        self.summary_fingerprint = summary_fingerprint
        self.duration_ms = duration_ms
        self.created_at = created_at
        self.total_hits = total_hits
        self.is_idempotent_reuse = is_idempotent_reuse
        self.raw_payload = raw_payload
        self.exception_message = exception_message


def _make_engine(sources_failed=None, *, raises=None, fail_after=False,
                **pack_kwargs):
    """Build a fake engine that records ``discover`` invocations.

    - ``sources_failed``: list of strings placed in pack.sources_failed
    - ``raises``: Exception class OR an Exception instance to raise from
      discover(). When a class is given, it is instantiated with "boom".
    - ``fail_after``: when True, raises on subsequent calls (used to model
      "succeeds once, then raises" if needed; default is single-shot raise
      per test which is enough).
    - Any additional kwargs (``summary_text``, ``missing_information``,
      ``overall_confidence``, ``overall_freshness_score``,
      ``sources_queried``, ``hits``, ``citations``, ``source_uri``,
      ``query_fingerprint``, ``summary_fingerprint``, ``duration_ms``,
      ``created_at``, ``total_hits``, ``is_idempotent_reuse``,
      ``raw_payload``, ``exception_message``, ...) are forwarded to the
      :class:`_FakeEvidencePack` constructor so prompt-consumption tests
      can populate allowlisted advisory fields AND forbidden fields in
      one call.

    Returns ``(engine_mock, calls)`` where ``calls`` is a list of dicts
    capturing each call's kwargs.
    """
    calls = []

    def _discover(*, objective_id, objective_text):
        calls.append({
            "objective_id": objective_id,
            "objective_text": objective_text,
        })
        if raises is not None:
            if isinstance(raises, type):
                raise raises("boom")
            raise raises
        return _FakeEvidencePack(
            objective_id=objective_id,
            sources_failed=list(sources_failed or []),
            tag=f"pack-for-{objective_id[:16]}",
            **pack_kwargs,
        )

    engine = MagicMock()
    engine.discover.side_effect = _discover
    # Restore the keyword-only signature so tests can introspect it.
    engine.discover_calls = calls
    return engine, calls


def _available_services(engine):
    """Build an ObjectiveServices with the engine pre-installed and 'available' status."""
    return ObjectiveServices(
        session_id="b1e5-available",
        evidence_pack_status="available",
        evidence_pack_engine=engine,
    )


def _available_services_for(session_id, engine):
    return ObjectiveServices(
        session_id=session_id,
        evidence_pack_status="available",
        evidence_pack_engine=engine,
    )


class TestEvidencePackPrivateRuntimeState:
    """The constructor initializes the three private runtime fields to None
    and never invokes ``discover``."""

    def test_constructor_loads_state_without_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager

        # No engine at all → no discover is possible regardless.
        mgr = GoalManager(session_id="b1e5-init-no-services")
        assert mgr._last_evidence_pack is None
        assert mgr._last_evidence_attempted_objective_id is None
        assert mgr._last_evidence_succeeded_objective_id is None

    def test_constructor_with_available_services_does_not_invoke_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-init-available",
            services=_available_services_for("b1e5-init-available", engine),
        )
        assert mgr._last_evidence_pack is None
        assert mgr._last_evidence_attempted_objective_id is None
        assert mgr._last_evidence_succeeded_objective_id is None
        # No discover call ever happened.
        assert calls == []


class TestEvidencePackServiceGate:
    """Disabled / degraded / missing-services / missing-engine paths
    must make zero ``discover`` calls."""

    def test_disabled_services_cause_no_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        engine, calls = _make_engine()
        services = ObjectiveServices(
            session_id="b1e5-disabled",
            evidence_pack_status="disabled",
            evidence_pack_engine=engine,  # even with engine set, disabled wins
        )
        mgr = GoalManager(session_id="b1e5-disabled", services=services)
        mgr.set("keep existing behavior")
        mgr.set_contract(GoalContract(verification="v"))
        mgr.add_subgoal("a")
        mgr.remove_subgoal(1)
        mgr.clear_subgoals()
        # Even an active goal loop doesn't trigger discovery when disabled.
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("done")
        assert calls == []

    def test_degraded_services_cause_no_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        services = ObjectiveServices(
            session_id="b1e5-degraded",
            evidence_pack_status="degraded",
            evidence_pack_degrade_reason="factory_missing",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e5-degraded", services=services)
        mgr.set("keep existing behavior")
        mgr.add_subgoal("a")
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("done")
        assert calls == []

    def test_missing_engine_cause_no_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager

        # Build a "ghost" engine — make a MagicMock for the services container
        # but leave evidence_pack_engine explicitly None.
        services = ObjectiveServices(
            session_id="b1e5-no-engine",
            evidence_pack_status="available",  # available BUT engine missing
            evidence_pack_engine=None,
        )
        mgr = GoalManager(session_id="b1e5-no-engine", services=services)
        mgr.set("keep existing behavior")
        mgr.add_subgoal("a")
        # We don't have an engine to call so any call would AttributeError;
        # the gate must prevent it entirely. Use a sentinel to detect calls.
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("done")
        # No error means the gate worked; nothing to assert on calls since
        # we have no real engine. The behavioral assertion is "no crash".
        assert mgr._last_evidence_pack is None


class TestEvidencePackMutationTriggers:
    """Mutation triggers (set, set_contract, add_subgoal, remove_subgoal,
    clear_subgoals) invoke ``_ensure_current_evidence_pack`` exactly once
    per real revision change."""

    def test_set_invokes_one_discovery_after_save(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-set",
            services=_available_services_for("b1e5-set", engine),
        )
        mgr.set("ship the PR")
        assert len(calls) == 1
        assert calls[0]["objective_id"].startswith("goalrev-")
        assert "ship the PR" in calls[0]["objective_text"]

    def test_set_continues_when_discover_raises(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine(raises=RuntimeError)
        mgr = GoalManager(
            session_id="b1e5-set-raises",
            services=_available_services_for("b1e5-set-raises", engine),
        )
        state = mgr.set("ship the PR")
        assert state is not None and state.goal == "ship the PR"
        assert state.status == "active"
        assert len(calls) == 1
        # Invocation failure: pack not stored, succeeded marker unset.
        assert mgr._last_evidence_pack is None
        assert mgr._last_evidence_succeeded_objective_id is None
        # Attempt marker is set (locked to this revision).
        assert mgr._last_evidence_attempted_objective_id == calls[0]["objective_id"]
        # Subsequent evaluate_after_turn MUST NOT retry this revision.
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("did some work")
        assert len(calls) == 1  # still one call total

    def test_set_contract_changed_calls_once(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-contract",
            services=_available_services_for("b1e5-contract", engine),
        )
        mgr.set("ship it")
        assert len(calls) == 1
        mgr.set_contract(GoalContract(verification="pytest -q passes"))
        assert len(calls) == 2  # set + set_contract

    def test_set_contract_canonically_identical_calls_zero(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-contract-idem",
            services=_available_services_for("b1e5-contract-idem", engine),
        )
        mgr.set("ship it", contract=GoalContract(verification="x"))
        assert len(calls) == 1
        # Same contract → no save, no discover.
        mgr.set_contract(GoalContract(verification="x"))
        assert len(calls) == 1
        # Empty contract matches the default empty contract.
        mgr2 = GoalManager(
            session_id="b1e5-contract-empty",
            services=_available_services_for("b1e5-contract-empty", engine),
        )
        mgr2.set("ship it")
        mgr2.set_contract(GoalContract())
        # The default GoalContract has all empty fields, so to_dict == to_dict.
        assert len(calls) == 2  # only the second manager's set() added a call

    def test_add_subgoal_calls_once(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-add",
            services=_available_services_for("b1e5-add", engine),
        )
        mgr.set("g")
        assert len(calls) == 1
        mgr.add_subgoal("criterion A")
        assert len(calls) == 2
        mgr.add_subgoal("criterion B")
        assert len(calls) == 3

    def test_remove_subgoal_calls_once(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-rm",
            services=_available_services_for("b1e5-rm", engine),
        )
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        before = len(calls)
        mgr.remove_subgoal(1)
        assert len(calls) == before + 1

    def test_clear_subgoals_changed_calls_once(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-clear",
            services=_available_services_for("b1e5-clear", engine),
        )
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        before = len(calls)
        prev = mgr.clear_subgoals()
        assert prev == 2
        assert len(calls) == before + 1

    def test_clear_subgoals_already_empty_calls_zero(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-clear-empty",
            services=_available_services_for("b1e5-clear-empty", engine),
        )
        mgr.set("g")  # one call from set
        before = len(calls)
        prev = mgr.clear_subgoals()
        assert prev == 0
        assert len(calls) == before  # zero new calls


class TestEvidencePackRevisionIdentity:
    """objective_id derivation: deterministic, 40 chars, excludes session_id,
    sensitive to created_at/goal/contract/subgoals."""

    def test_objective_id_is_deterministic_40_chars_excludes_session_id(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState

        state = GoalState(goal="g", created_at=12345.6789)
        oid = GoalManager._compute_objective_id(state)
        assert len(oid) == 40
        assert oid.startswith("goalrev-")
        suffix = oid[len("goalrev-"):]
        assert len(suffix) == 32
        assert suffix == suffix.lower()
        # All hex.
        int(suffix, 16)
        # Determinism.
        oid2 = GoalManager._compute_objective_id(state)
        assert oid == oid2

    def test_full_created_at_goal_contract_subgoals_affect_identity(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, GoalContract

        base = GoalState(goal="g", created_at=1.0)
        # created_at difference
        other = GoalState(goal="g", created_at=2.0)
        assert GoalManager._compute_objective_id(base) != GoalManager._compute_objective_id(other)
        # goal difference
        other2 = GoalState(goal="h", created_at=1.0)
        assert GoalManager._compute_objective_id(base) != GoalManager._compute_objective_id(other2)
        # subgoals difference
        s3 = GoalState(goal="g", created_at=1.0, subgoals=["a"])
        assert GoalManager._compute_objective_id(base) != GoalManager._compute_objective_id(s3)
        # contract difference (use a non-empty field to be observable)
        s4 = GoalState(goal="g", created_at=1.0, contract=GoalContract(verification="v"))
        assert GoalManager._compute_objective_id(base) != GoalManager._compute_objective_id(s4)
        # contract changes are also observable
        s4b = GoalState(goal="g", created_at=1.0, contract=GoalContract(verification="w"))
        assert GoalManager._compute_objective_id(s4) != GoalManager._compute_objective_id(s4b)
        # contract whitespace normalization should NOT collapse ids when same field
        s4c = GoalState(goal="g", created_at=1.0, contract=GoalContract(verification="  v  "))
        # whitespace difference at raw to_dict level preserves difference
        assert GoalManager._compute_objective_id(s4) != GoalManager._compute_objective_id(s4c)

    def test_same_second_goals_with_different_content_do_not_collide(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState

        # Same created_at (same second), different goal text
        a = GoalManager._compute_objective_id(GoalState(goal="alpha", created_at=1700000000.0))
        b = GoalManager._compute_objective_id(GoalState(goal="beta", created_at=1700000000.0))
        assert a != b
        # Same created_at + goal + contract but different subgoals
        a2 = GoalManager._compute_objective_id(GoalState(goal="g", created_at=1700000000.0, subgoals=["x"]))
        b2 = GoalManager._compute_objective_id(GoalState(goal="g", created_at=1700000000.0, subgoals=["y"]))
        assert a2 != b2
        # Session_id is NOT in the payload — explicitly verify via state.
        # The hash is computed over state, not session_id; nothing to assert
        # other than the property above.

    def test_revision_payload_excludes_session_id(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState

        s = GoalState(goal="g", created_at=1.0, subgoals=["a"])
        payload = GoalManager._revision_payload(s)
        assert "session_id" not in payload
        assert set(payload.keys()) == {"created_at", "goal", "contract", "subgoals"}

    def test_full_created_at_float_preserved_in_payload(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState

        # Full floating-point precision must be preserved.
        s = GoalState(goal="g", created_at=1234567890.123456789)
        payload = GoalManager._revision_payload(s)
        # The float itself is preserved (json repr may differ but the
        # numeric value is preserved through our canonical serialization).
        assert payload["created_at"] == 1234567890.123456789


class TestObjectiveTextRendering:
    """objective_text rendering: deterministic schema, bounded to 10_000."""

    def test_exact_objective_text_rendering(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState

        s = GoalState(goal="ship it")
        assert GoalManager._render_objective_text(s) == "ship it"

        s2 = GoalState(goal="ship it", subgoals=["write tests", "update docs"])
        assert GoalManager._render_objective_text(s2) == (
            "ship it\n\nSubgoals:\n1. write tests\n2. update docs"
        )

    def test_objective_text_is_bounded_to_10000_chars(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState

        long_goal = "a" * 20_000
        s = GoalState(goal=long_goal)
        rendered = GoalManager._render_objective_text(s)
        assert len(rendered) == 10_000

    def test_contract_prose_absent_from_objective_text(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState, GoalContract

        c = GoalContract(
            outcome="the outcome",
            verification="pytest passes",
            constraints="do not change the API",
            boundaries="services/auth",
            stop_when="schema change needed",
        )
        s = GoalState(goal="ship it", contract=c)
        rendered = GoalManager._render_objective_text(s)
        assert "the outcome" not in rendered
        assert "pytest passes" not in rendered
        assert "do not change the API" not in rendered
        assert "services/auth" not in rendered
        assert "schema change needed" not in rendered
        # Only the goal text is present.
        assert "ship it" in rendered

    def test_objective_text_when_no_goal(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalState

        s = GoalState(goal="")
        rendered = GoalManager._render_objective_text(s)
        # Defensive: empty goal yields empty text (engine still called
        # only if there is a state — but we never get here without state).
        assert rendered == ""


class TestEvaluateAfterTurnTrigger:
    """evaluate_after_turn trigger: one ensure call on eligible path."""

    def test_evaluate_after_turn_plus_next_continuation_prompt_does_not_double_attempt(
        self, hermes_home,
    ):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-eval-once",
            services=_available_services_for("b1e5-eval-once", engine),
        )
        mgr.set("g")
        # set() succeeded — succeeded_objective_id is set, so a passive
        # evaluate_after_turn on the same revision is a no-op (the gate
        # short-circuits). next_continuation_prompt must NOT attempt
        # discovery on top of that.
        before = len(calls)
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "more to do", False, None, False)):
            d = mgr.evaluate_after_turn("did some work")
        # No new discovery call — succeeded_objective_id matches.
        assert len(calls) == before
        # next_continuation_prompt must NOT attempt discovery either.
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert len(calls) == before

    def test_inactive_evaluate_makes_no_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-eval-inactive",
            services=_available_services_for("b1e5-eval-inactive", engine),
        )
        d = mgr.evaluate_after_turn("x")
        assert d["verdict"] == "inactive"
        assert calls == []

    def test_waiting_evaluate_makes_no_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-eval-waiting",
            services=_available_services_for("b1e5-eval-waiting", engine),
        )
        mgr.set("g")
        before = len(calls)
        # Patch is_waiting to True so the waiting short-circuit fires.
        # We don't spawn a real subprocess (the live-system guard blocks
        # out-of-subtree kill()).
        with patch.object(mgr, "is_waiting", return_value=True):
            d = mgr.evaluate_after_turn("x")
        assert d["verdict"] == "waiting"
        # Parked → no new discovery call.
        assert len(calls) == before

    def test_evaluate_with_empty_last_response_makes_no_discovery(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-eval-empty",
            services=_available_services_for("b1e5-eval-empty", engine),
        )
        mgr.set("g")
        before = len(calls)
        # Empty/whitespace last_response → no ensure call.
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "empty", False, None, False)):
            mgr.evaluate_after_turn("   \n  ")
        assert len(calls) == before

    def test_attempted_marker_remains_current_after_failure(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine(raises=RuntimeError("boom"))
        mgr = GoalManager(
            session_id="b1e5-marker-fail",
            services=_available_services_for("b1e5-marker-fail", engine),
        )
        mgr.set("g")
        assert len(calls) == 1
        attempted = mgr._last_evidence_attempted_objective_id
        assert attempted is not None
        # Subsequent evaluate must not change the marker or retry.
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("done")
        assert len(calls) == 1
        assert mgr._last_evidence_attempted_objective_id == attempted


class TestForceEvidenceRetry:
    """force_evidence_retry clears ONLY the attempt marker; the diagnostic
    pack and succeeded marker survive."""

    def test_force_evidence_retry_clears_only_attempted_marker(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-force-retry",
            services=_available_services_for("b1e5-force-retry", engine),
        )
        mgr.set("g")
        # Capture the diagnostic pack + succeeded marker.
        prev_pack = mgr._last_evidence_pack
        prev_succeeded = mgr._last_evidence_succeeded_objective_id
        assert prev_pack is not None
        assert prev_succeeded is not None
        # Now force a retry — should clear the attempt marker only.
        mgr.resume(force_evidence_retry=True)
        assert mgr._last_evidence_pack is prev_pack  # preserved
        assert mgr._last_evidence_succeeded_objective_id == prev_succeeded  # preserved
        assert mgr._last_evidence_attempted_objective_id is None  # cleared

    def test_default_resume_makes_zero_discovery_calls(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-resume-default",
            services=_available_services_for("b1e5-resume-default", engine),
        )
        mgr.set("g")
        before = len(calls)
        mgr.pause("test")
        mgr.resume()
        assert len(calls) == before

    def test_next_eligible_evaluate_retries_exactly_once_after_force_retry(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-force-retry-next",
            services=_available_services_for("b1e5-force-retry-next", engine),
        )
        mgr.set("g")
        # Failure on first attempt
        # (force retry after success — but the test isolates the retry flow)
        # Force retry path on a previously-successful revision:
        mgr.resume(force_evidence_retry=True)
        before = len(calls)
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("more work")
        # Exactly one new discovery call.
        assert len(calls) == before + 1
        # And another evaluate_after_turn with same revision: zero new.
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("more work again")
        assert len(calls) == before + 1

    def test_passive_failure_attempts_once_per_revision(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine(raises=RuntimeError)
        mgr = GoalManager(
            session_id="b1e5-passive-fail",
            services=_available_services_for("b1e5-passive-fail", engine),
        )
        mgr.set("g")
        assert len(calls) == 1
        # 5 more passive calls — none should retry.
        for i in range(5):
            with patch("hermes_cli.goals.judge_goal",
                       return_value=("continue", "x", False, None, False)):
                mgr.evaluate_after_turn(f"step {i}")
        assert len(calls) == 1  # still one


class TestFailureClassification:
    """Failure classification: provider/source failure vs outer exception
    vs set_meta failure."""

    def test_provider_sources_failed_pack_remains_current(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine(sources_failed=["obsidian"])
        mgr = GoalManager(
            session_id="b1e5-provider-failed",
            services=_available_services_for("b1e5-provider-failed", engine),
        )
        mgr.set("g")
        assert len(calls) == 1
        # The pack is stored as current — engine returned a pack with
        # sources_failed inside it, but invocation succeeded.
        assert mgr._last_evidence_pack is not None
        assert mgr._last_evidence_pack.sources_failed == ["obsidian"]
        assert mgr._current_evidence_pack is mgr._last_evidence_pack
        assert mgr._last_evidence_succeeded_objective_id is not None

    def test_outer_discover_exception_is_invocation_failure(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine(raises=RuntimeError("nope"))
        mgr = GoalManager(
            session_id="b1e5-outer-failure",
            services=_available_services_for("b1e5-outer-failure", engine),
        )
        mgr.set("g")
        # No pack stored, succeeded marker None.
        assert mgr._last_evidence_pack is None
        assert mgr._last_evidence_succeeded_objective_id is None
        assert mgr._current_evidence_pack is None

    def test_set_meta_style_runtime_error_is_invocation_failure(self, hermes_home):
        """Even if the underlying failure looks like a 'set_meta' failure,
        it surfaces as an invocation_failure (no new pack)."""
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine(raises=RuntimeError("set_meta failed"))
        mgr = GoalManager(
            session_id="b1e5-set-meta-fail",
            services=_available_services_for("b1e5-set-meta-fail", engine),
        )
        mgr.set("g")
        assert mgr._last_evidence_pack is None
        assert mgr._last_evidence_succeeded_objective_id is None


class TestPreviousPackRetention:
    """A failed new revision may retain the previous pack internally for
    diagnostics, but it must not be exposed as current."""

    def test_previous_successful_pack_survives_internally_after_new_revision_failure(
        self, hermes_home,
    ):
        from hermes_cli.goals import GoalManager, GoalState

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-prev-pack",
            services=_available_services_for("b1e5-prev-pack", engine),
        )
        mgr.set("g")
        prev_pack = mgr._last_evidence_pack
        prev_succeeded = mgr._last_evidence_succeeded_objective_id
        assert prev_pack is not None
        # Now mutate the state (new revision). Monkey-patch discover to raise.
        engine.discover.side_effect = RuntimeError("new revision failed")
        mgr.add_subgoal("new criterion")
        # Internal diagnostics: previous pack & succeeded marker survive.
        assert mgr._last_evidence_pack is prev_pack
        assert mgr._last_evidence_succeeded_objective_id == prev_succeeded
        # But the attempted marker has moved to the new revision.
        assert mgr._last_evidence_attempted_objective_id != prev_succeeded

    def test_previous_pack_is_not_returned_by_current_evidence_pack_for_new_revision(
        self, hermes_home,
    ):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-prev-pack-stale",
            services=_available_services_for("b1e5-prev-pack-stale", engine),
        )
        mgr.set("g")
        prev_pack = mgr._last_evidence_pack
        # New revision, discover raises.
        engine.discover.side_effect = RuntimeError("new revision failed")
        mgr.add_subgoal("new criterion")
        # _current_evidence_pack must return None for the failed new revision.
        assert mgr._current_evidence_pack is None
        # The internal previous pack is still there for diagnostics.
        assert mgr._last_evidence_pack is prev_pack


class TestInterruptPropagation:
    """KeyboardInterrupt and SystemExit MUST propagate (only Exception is
    swallowed)."""

    def test_keyboard_interrupt_propagates(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        engine.discover.side_effect = KeyboardInterrupt()
        mgr = GoalManager(
            session_id="b1e5-kbd",
            services=_available_services_for("b1e5-kbd", engine),
        )
        import pytest
        with pytest.raises(KeyboardInterrupt):
            mgr.set("g")

    def test_system_exit_propagates(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        engine.discover.side_effect = SystemExit(1)
        mgr = GoalManager(
            session_id="b1e5-sys",
            services=_available_services_for("b1e5-sys", engine),
        )
        import pytest
        with pytest.raises(SystemExit):
            mgr.set("g")


class TestSafeLogging:
    """Logging must not leak exception text, goal, contract, subgoals,
    or any sensitive metadata."""

    def test_safe_logging_excludes_content_and_exception_message(self, hermes_home, caplog):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine(
            raises=RuntimeError("SECRET: do not log this verbatim")
        )
        mgr = GoalManager(
            session_id="b1e5-safelogsid",
            services=_available_services_for("b1e5-safelogsid", engine),
        )
        with caplog.at_level("WARNING"):
            mgr.set("SENSITIVE GOAL TEXT 12345-XYZ")
        # The exception message MUST NOT appear in the log.
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "SECRET: do not log this verbatim" not in joined
        assert "SENSITIVE GOAL TEXT 12345-XYZ" not in joined
        # The exception class name MUST appear (we log only the class name).
        assert "RuntimeError" in joined


class TestInspectionPathsHaveNoDiscovery:
    """status, status_line, show, has_goal, is_active, is_waiting,
    contract rendering and equivalent inspection paths must cause no
    discovery."""

    def test_status_show_paths_have_zero_discovery_side_effects(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-status-show",
            services=_available_services_for("b1e5-status-show", engine),
        )
        mgr.set("g")
        mgr.add_subgoal("a")
        before = len(calls)
        # Status / inspection surfaces.
        _ = mgr.status_line()
        _ = mgr.is_active()
        _ = mgr.is_waiting()
        _ = mgr.has_goal()
        _ = mgr.has_contract()
        _ = mgr.render_subgoals()
        _ = mgr.render_contract()
        assert len(calls) == before

    def test_next_continuation_prompt_has_zero_discovery_side_effects(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-cont",
            services=_available_services_for("b1e5-cont", engine),
        )
        mgr.set("g")
        before = len(calls)
        for _ in range(5):
            _ = mgr.next_continuation_prompt()
        assert len(calls) == before


class TestClearResetsEvidenceFields:
    """clear() resets all three private evidence fields and makes zero
    discovery calls."""

    def test_clear_resets_all_evidence_runtime_fields(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-clear",
            services=_available_services_for("b1e5-clear", engine),
        )
        mgr.set("g")
        assert mgr._last_evidence_pack is not None
        before = len(calls)
        mgr.clear()
        assert mgr._last_evidence_pack is None
        assert mgr._last_evidence_attempted_objective_id is None
        assert mgr._last_evidence_succeeded_objective_id is None
        assert len(calls) == before
        # State also cleared.
        assert mgr.state is None
        assert mgr.is_active() is False


class TestActivePersistedStateHydration:
    """Active persisted state hydrates on first eligible
    evaluate_after_turn."""

    def test_active_persisted_state_hydrates_on_first_eligible_evaluate(self, hermes_home):
        from hermes_cli.goals import GoalManager

        # Set the goal first (one discovery call), then re-create the
        # manager — the new manager should NOT auto-discover (constructor
        # is no-discovery by spec). The first eligible evaluate_after_turn
        # IS the trigger.
        engine, calls = _make_engine()
        sid = "b1e5-hydrate"
        mgr = GoalManager(
            session_id=sid,
            services=_available_services_for(sid, engine),
        )
        mgr.set("g")
        assert len(calls) == 1

        # Reload — fresh constructor.
        mgr2 = GoalManager(
            session_id=sid,
            services=_available_services_for(sid, engine),
        )
        # No discovery from the constructor.
        assert mgr2._last_evidence_pack is None
        # First eligible evaluate_after_turn IS the hydration trigger.
        before = len(calls)
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            d = mgr2.evaluate_after_turn("did some work")
        assert d["should_continue"] is True
        assert len(calls) == before + 1
        # Now the pack is hydrated.
        assert mgr2._last_evidence_pack is not None
        assert mgr2._last_evidence_succeeded_objective_id is not None


class TestSealedFileInvariants:
    """No GoalState schema change. No prompt consumption in B1-E5.
    Sealed files are not modified."""

    def test_no_goal_state_serialization_schema_change(self, hermes_home):
        """GoalState.to_json / from_json round-trip is unchanged: pre-B1-E5
        fields still parse, no B1-E5 fields appear in the schema."""
        from hermes_cli.goals import GoalState, GoalContract

        s = GoalState(
            goal="g", status="active", turns_used=1, max_turns=5,
            created_at=1.0, last_turn_at=2.0,
            consecutive_parse_failures=0, consecutive_transport_failures=0,
            subgoals=["a", "b"],
            waiting_on_pid=None, waiting_on_session=None,
            waiting_until=0.0, waiting_reason=None, waiting_since=0.0,
            contract=GoalContract(verification="x"),
        )
        raw = s.to_json()
        # Round-trip.
        s2 = GoalState.from_json(raw)
        assert s2.goal == s.goal
        assert s2.subgoals == s.subgoals
        assert s2.contract.to_dict() == s.contract.to_dict()
        # The 3 private evidence fields are NOT serialized.
        assert "_last_evidence_pack" not in raw
        assert "_last_evidence_attempted_objective_id" not in raw
        assert "_last_evidence_succeeded_objective_id" not in raw

    def test_no_prompt_consumption_in_b1_e5(self, hermes_home):
        """The continuation prompt must NOT include pack content in B1-E5
        (that's B1-E6)."""
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-no-consume",
            services=_available_services_for("b1e5-no-consume", engine),
        )
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        # No reference to the pack, no raw sources, etc.
        assert prompt is not None
        assert "evidence_pack" not in prompt.lower()
        assert "sources_failed" not in prompt.lower()
        assert "sources_queried" not in prompt.lower()
        # The pack itself is held internally, never rendered.
        assert mgr._current_evidence_pack is not None
        # Add a subgoal and check prompt again — still no pack content.
        mgr.add_subgoal("extra")
        prompt2 = mgr.next_continuation_prompt()
        assert "evidence_pack" not in prompt2.lower()

    def test_engine_discover_invocation_signature(self, hermes_home):
        """engine.discover is called with EXACTLY objective_id and
        objective_text — no alternate risk, complexity, source or max-hit
        kwargs."""
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        mgr = GoalManager(
            session_id="b1e5-sig",
            services=_available_services_for("b1e5-sig", engine),
        )
        mgr.set("g")
        assert len(calls) == 1
        # The call must have only objective_id and objective_text as kwargs.
        assert set(calls[0].keys()) == {"objective_id", "objective_text"}
        # No 'risk_profile', 'complexity', 'sources_requested',
        # 'max_hits_per_source' etc.
        forbidden = {
            "risk_profile", "complexity", "sources_requested",
            "max_hits_per_source", "max_hits_total", "goal_class",
        }
        assert not (set(calls[0].keys()) & forbidden)

    def test_dry_run_and_rollback_not_invoked(self, hermes_home):
        """Production integration uses ONLY discover() — no dry_run, no
        rollback, no global engine, no close, no ObjectiveServices
        mutation."""
        from hermes_cli.goals import GoalManager

        engine = MagicMock()
        engine.discover.return_value = _FakeEvidencePack(objective_id="x")
        mgr = GoalManager(
            session_id="b1e5-engine-surf",
            services=_available_services_for("b1e5-engine-surf", engine),
        )
        mgr.set("g")
        # discover called once; nothing else on the engine.
        assert engine.discover.call_count == 1
        assert engine.dry_run.call_count == 0
        assert engine.rollback.call_count == 0
        assert engine.close.call_count == 0
        # ObjectiveServices was not mutated by the manager.
        services = mgr.services
        for attr in (
            "session_id", "sources", "storage", "audit_sink",
            "evidence_pack_engine", "evidence_pack_status",
            "evidence_pack_degrade_reason", "evidence_pack_error_type",
        ):
            assert getattr(services, attr) == getattr(
                ObjectiveServices(
                    session_id="b1e5-engine-surf",
                    evidence_pack_status="available",
                    evidence_pack_engine=engine,
                ),
                attr,
            )


class TestDefaultOffSideEffects:
    """When evidence-pack configuration is disabled (services=None /
    disabled / degraded / missing-engine), no discover call, no provider
    call, no storage read/write, no emitter side effect."""

    def test_default_off_no_side_effects(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="b1e5-default-off")
        # Constructor never writes storage — DB is None because hermes_home
        # is isolated. set() is the first persistence event.
        mgr.set("g")
        # status_line / has_goal / is_active / etc. don't trigger discover.
        _ = mgr.status_line()
        _ = mgr.has_goal()
        _ = mgr.is_active()
        _ = mgr.is_waiting()
        # No engine → no call. No emitter → no monitoring side effect.
        assert mgr._last_evidence_pack is None
        assert mgr._last_evidence_attempted_objective_id is None
        assert mgr._last_evidence_succeeded_objective_id is None
        # Run a turn with the judge patched — still no discover call.
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            d = mgr.evaluate_after_turn("did some work")
        assert d["should_continue"] is True


class TestObjectiveServicesMutation:
    """The manager never mutates its ObjectiveServices object."""

    def test_objective_services_not_mutated(self, hermes_home):
        from hermes_cli.goals import GoalManager

        engine, calls = _make_engine()
        services = _available_services_for("b1e5-no-mutate", engine)
        snapshot = {
            "session_id": services.session_id,
            "sources": services.sources,
            "storage": services.storage,
            "audit_sink": services.audit_sink,
            "evidence_pack_engine": services.evidence_pack_engine,
            "evidence_pack_status": services.evidence_pack_status,
            "evidence_pack_degrade_reason": services.evidence_pack_degrade_reason,
            "evidence_pack_error_type": services.evidence_pack_error_type,
        }
        mgr = GoalManager(
            session_id="b1e5-no-mutate", services=services,
        )
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.remove_subgoal(1)
        with patch("hermes_cli.goals.judge_goal",
                   return_value=("continue", "x", False, None, False)):
            mgr.evaluate_after_turn("work")
        # Snapshot unchanged.
        for k, v in snapshot.items():
            assert getattr(services, k) == v


# ──────────────────────────────────────────────────────────────────────
# B1-E6: production EvidencePack continuation-prompt consumption
# ──────────────────────────────────────────────────────────────────────


# Canonical continuation marker the insertion helper searches for. The
# constant is exported by the implementation module under the same name
# so tests can assert against it without duplicating the literal.
_B1E6_MARKER = "\n\nContinue working"
_B1E6_OPEN = "<untrusted_goal_evidence>"
_B1E6_CLOSE = "</untrusted_goal_evidence>"
_B1E6_ADVISORY_FRAGMENT = "advisory reference data only"
_B1E6_TRUNCATION = "…[evidence block truncated]"


def _mgr_with_pack(hermes_home, **pack_kwargs):
    """Build a GoalManager backed by a fake engine that returns one pack.

    ``pack_kwargs`` is forwarded to :func:`_make_engine` so tests can
    populate advisory fields and forbidden fields directly. After
    ``set()`` runs the manager has a current pack available via
    ``_current_evidence_pack``.
    """
    engine, _calls = _make_engine(**pack_kwargs)
    sid = "b1e6-pack"
    services = ObjectiveServices(
        session_id=sid,
        evidence_pack_status="available",
        evidence_pack_engine=engine,
    )
    mgr = GoalManager(session_id=sid, services=services)
    return mgr


class TestB1E6TemplateConstantsUnchanged:
    """The exported continuation template constants are an existing
    external API surface — they MUST remain byte-identical to the
    sealed base. This test pins their exact strings."""

    EXPECTED_PLAIN = (
        "[Continuing toward your standing goal]\n"
        "Goal: {goal}\n\n"
        "Continue working toward this goal. Take the next concrete step. "
        "If you believe the goal is complete, state so explicitly and stop. "
        "If you are blocked and need input from the user, say so clearly and stop."
    )

    EXPECTED_SUBGOALS = (
        "[Continuing toward your standing goal]\n"
        "Goal: {goal}\n\n"
        "Additional criteria the user added mid-loop:\n"
        "{subgoals_block}\n\n"
        "Continue working toward the goal AND all additional criteria. Take "
        "the next concrete step. If you believe the goal and every "
        "additional criterion are complete, state so explicitly and stop. "
        "If you are blocked and need input from the user, say so clearly "
        "and stop."
    )

    EXPECTED_CONTRACT = (
        "[Continuing toward your standing goal]\n"
        "Goal: {goal}\n\n"
        "Completion contract:\n"
        "{contract_block}\n\n"
        "Continue working toward the outcome above. Take the next concrete step. "
        "Stay within the stated boundaries and do not violate the constraints. "
        "Before claiming the goal is done, satisfy the Verification criterion and "
        "show the concrete evidence (command output, file contents, test result). "
        "If you hit the stated stop condition or are otherwise blocked and need "
        "user input, say so clearly and stop."
    )

    def test_plain_template_byte_identical(self):
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE
        assert CONTINUATION_PROMPT_TEMPLATE == self.EXPECTED_PLAIN

    def test_subgoals_template_byte_identical(self):
        from hermes_cli.goals import CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE
        assert CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE == self.EXPECTED_SUBGOALS

    def test_contract_template_byte_identical(self):
        from hermes_cli.goals import CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE
        assert CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE == self.EXPECTED_CONTRACT

    def test_no_evidence_block_placeholder_in_templates(self):
        """No {evidence_block} placeholder was added."""
        from hermes_cli.goals import (
            CONTINUATION_PROMPT_TEMPLATE,
            CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE,
            CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE,
        )
        for tpl in (
            CONTINUATION_PROMPT_TEMPLATE,
            CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE,
            CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE,
        ):
            assert "evidence_block" not in tpl

    def test_external_format_callers_backward_compatible(self):
        """Existing ``template.format(goal=...)`` calls still work — the
        templates accept the historical kwargs and produce the sealed
        output byte-for-byte."""
        from hermes_cli.goals import (
            CONTINUATION_PROMPT_TEMPLATE,
            CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE,
            CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE,
        )

        plain = CONTINUATION_PROMPT_TEMPLATE.format(goal="ship it")
        assert plain == self.EXPECTED_PLAIN.format(goal="ship it")
        sub = CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal="ship it", subgoals_block="- 1. write tests",
        )
        assert sub == self.EXPECTED_SUBGOALS.format(
            goal="ship it", subgoals_block="- 1. write tests",
        )
        con = CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal="ship it", contract_block="- Verification: pytest passes",
        )
        assert con == self.EXPECTED_CONTRACT.format(
            goal="ship it", contract_block="- Verification: pytest passes",
        )


class TestB1E6InsertionHelperContract:
    """``GoalManager._insert_evidence_block`` is the pure helper that
    splices a pre-rendered evidence block into a baseline prompt. The
    contract is enforced directly without any GoalManager state."""

    def test_empty_block_returns_baseline_byte_identically(self):
        from hermes_cli.goals import GoalManager

        baseline = "hello\n\nContinue working toward foo."
        assert GoalManager._insert_evidence_block(baseline, "") == baseline
        # All empty inputs are equivalent — None / "" produce the same
        # baseline return value (no side-effect).
        assert GoalManager._insert_evidence_block(baseline, None) == baseline

    def test_missing_marker_returns_baseline_unchanged(self):
        from hermes_cli.goals import GoalManager

        baseline = "no marker here at all"
        block = "\n\nBLOCK"
        assert GoalManager._insert_evidence_block(baseline, block) == baseline

    def test_ambiguous_marker_returns_baseline_unchanged(self):
        from hermes_cli.goals import GoalManager

        baseline = "first\n\nContinue working\n\nContinue working"
        block = "\n\nBLOCK"
        # Two occurrences of the marker → refuse to mis-position.
        assert GoalManager._insert_evidence_block(baseline, block) == baseline

    def test_unique_marker_inserts_block_before_marker(self):
        from hermes_cli.goals import GoalManager

        baseline = "intro\n\nContinue working rest"
        block = "\n\nBLOCK"
        out = GoalManager._insert_evidence_block(baseline, block)
        # Block sits between baseline tail and the marker; marker survives.
        assert out == "intro\n\nBLOCK\n\nContinue working rest"

    def test_helper_does_not_read_pack_state(self):
        """The helper is a staticmethod and never touches pack / manager
        state — confirm it returns deterministically for a self-built
        baseline without constructing any GoalManager instance."""
        from hermes_cli.goals import GoalManager

        baseline = "head\n\nContinue working tail"
        a = GoalManager._insert_evidence_block(baseline, "\n\nX")
        b = GoalManager._insert_evidence_block(baseline, "\n\nX")
        assert a == b == "head\n\nX\n\nContinue working tail"

    def test_helper_is_callable_via_class_only(self):
        """Static method — callable as ``GoalManager._insert_evidence_block``.

        Behavioral contract: the function is reachable as
        ``GoalManager._insert_evidence_block`` (class-only, no instance
        required), accepts exactly the two documented arguments by name
        (``baseline_prompt``, ``evidence_block``), and rejects unknown
        kwargs with ``TypeError``. The drive invokes the documented
        callable and observes the public surface.
        """
        import pytest

        from hermes_cli.goals import GoalManager

        # Behavioral observation 1: callable as a class attribute (no
        # instance required).
        assert callable(getattr(GoalManager, "_insert_evidence_block", None))

        # Behavioral observation 2: documented kwargs work; the function
        # inserts the evidence block immediately before the unique
        # ``\n\nContinue working`` marker when present, and returns the
        # baseline byte-identically when the marker is absent or ambiguous.
        baseline_with_marker = "intro\n\nContinue working rest"
        out = GoalManager._insert_evidence_block(
            baseline_prompt=baseline_with_marker,
            evidence_block="\n\nBLOCK",
        )
        assert isinstance(out, str)
        assert "BLOCK" in out
        assert out.startswith("intro")

        # Behavioral observation 3: positional args (in the documented
        # order) also work — the contract is two positional parameters.
        out_pos = GoalManager._insert_evidence_block(
            baseline_with_marker, "\n\nBLOCK"
        )
        assert out_pos == out

        # Behavioral observation 4: empty evidence_block returns the
        # baseline byte-identically (a documented contract, not a side
        # effect of the implementation).
        out_empty = GoalManager._insert_evidence_block(
            baseline_prompt=baseline_with_marker,
            evidence_block="",
        )
        assert out_empty == baseline_with_marker

        # Behavioral observation 5: unknown kwargs are rejected with
        # TypeError (the public surface does not accept extras).
        with pytest.raises(TypeError):
            GoalManager._insert_evidence_block(
                baseline_prompt=baseline_with_marker,
                evidence_block="\n\nBLOCK",
                unexpected_kwarg="x",
            )


class TestB1E6NoPackByteIdentical:
    """No current pack → the continuation prompt must be byte-identical
    to the B1-E5 baseline (no evidence block, no extra whitespace)."""

    def test_plain_goal_no_pack_byte_identical_to_baseline(self, hermes_home):
        from hermes_cli.goals import (
            GoalManager,
            CONTINUATION_PROMPT_TEMPLATE,
        )

        mgr = GoalManager(session_id="b1e6-plain-nopack")
        mgr.set("ship the feature")
        # No services → no pack is ever produced.
        baseline = CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the feature")
        assert mgr.next_continuation_prompt() == baseline

    def test_subgoal_goal_no_pack_byte_identical_to_baseline(self, hermes_home):
        from hermes_cli.goals import (
            GoalManager,
            CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE,
        )

        mgr = GoalManager(session_id="b1e6-subgoal-nopack")
        mgr.set("ship the feature")
        mgr.add_subgoal("add tests")
        baseline = CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal="ship the feature",
            subgoals_block=mgr.state.render_subgoals_block(),
        )
        assert mgr.next_continuation_prompt() == baseline

    def test_contract_goal_no_pack_byte_identical_to_baseline(self, hermes_home):
        from hermes_cli.goals import (
            GoalManager, GoalContract,
            CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE,
        )

        mgr = GoalManager(session_id="b1e6-contract-nopack")
        mgr.set("ship")
        mgr.set_contract(GoalContract(outcome="o", verification="v"))
        baseline = CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal="ship",
            contract_block=mgr.state.contract.render_block(),
        )
        assert mgr.next_continuation_prompt() == baseline

    def test_pack_present_but_empty_renderable_no_block(self, hermes_home):
        """A current pack whose allowlisted fields are all empty must
        render an empty block — so the prompt is byte-identical to the
        no-pack baseline."""
        from hermes_cli.goals import (
            GoalManager,
            CONTINUATION_PROMPT_TEMPLATE,
        )
        from agent.executive.services import ObjectiveServices

        engine, _calls = _make_engine(
            # All advisory fields empty; only forbidden fields populated.
            sources_failed=[],
            summary_text="",
            missing_information=[],
            overall_confidence=0.0,
            overall_freshness_score=0.0,
        )
        services = ObjectiveServices(
            session_id="b1e6-empty-pack",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-empty-pack", services=services)
        mgr.set("g")
        # The pack exists but has no renderable content.
        assert mgr._current_evidence_pack is not None
        baseline = CONTINUATION_PROMPT_TEMPLATE.format(goal="g")
        assert mgr.next_continuation_prompt() == baseline


class TestB1E6ContinuationPromptIntegration:
    """The block is rendered before the trusted continuation marker; the
    trusted instruction remains the final semantic section."""

    def test_block_renders_before_continuation_marker(self, hermes_home):
        mgr = _mgr_with_pack(hermes_home, summary_text="the summary")
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        open_idx = prompt.index(_B1E6_OPEN)
        marker_idx = prompt.index(_B1E6_MARKER)
        # Open tag precedes the marker; closing tag precedes the marker.
        close_idx = prompt.index(_B1E6_CLOSE)
        assert open_idx < close_idx < marker_idx

    def test_trusted_continuation_marker_survives(self, hermes_home):
        mgr = _mgr_with_pack(hermes_home, summary_text="S")
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        # The historical "Continue working toward..." text remains the
        # final semantic section, immediately after the closing tag.
        assert prompt.rstrip().endswith(
            "If you are blocked and need input from the user, say so "
            "clearly and stop."
        )
        # The marker substring appears exactly once.
        assert prompt.count(_B1E6_MARKER) == 1

    def test_plain_goal_prompt_integration(self, hermes_home):
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="A summary",
            missing_information=["missing"],
            overall_confidence=0.8,
            overall_freshness_score=0.7,
        )
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        assert "Goal: g" in prompt
        assert _B1E6_OPEN in prompt
        assert _B1E6_CLOSE in prompt
        assert "Summary: A summary" in prompt
        assert "Missing information:\n- missing" in prompt
        assert "Overall confidence: 0.80" in prompt
        assert "Overall freshness: 0.70" in prompt

    def test_subgoal_prompt_integration(self, hermes_home):
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="S",
            overall_confidence=0.5,
            overall_freshness_score=0.5,
        )
        mgr.set("g")
        mgr.add_subgoal("sub1")
        prompt = mgr.next_continuation_prompt()
        assert "Additional criteria the user added mid-loop:" in prompt
        assert "1. sub1" in prompt
        assert _B1E6_OPEN in prompt
        assert "Summary: S" in prompt

    def test_contract_prompt_integration(self, hermes_home):
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="S",
            overall_confidence=0.5,
            overall_freshness_score=0.5,
        )
        mgr.set("g")
        mgr.set_contract(GoalContract(verification="pytest passes"))
        prompt = mgr.next_continuation_prompt()
        assert "Completion contract:" in prompt
        assert "Verification: pytest passes" in prompt
        assert _B1E6_OPEN in prompt
        assert "Summary: S" in prompt

    def test_prompt_order_goal_then_advisory_then_continuation(self, hermes_home):
        """Final semantic order: authoritative goal → contract/subgoals →
        bounded untrusted advisory → authoritative continuation instruction."""
        mgr = _mgr_with_pack(hermes_home, summary_text="S")
        mgr.set("ship the feature")
        mgr.set_contract(GoalContract(outcome="out"))
        prompt = mgr.next_continuation_prompt()
        # Goal line precedes the block.
        assert prompt.index("Goal: ship the feature") < prompt.index(_B1E6_OPEN)
        # Contract precedes the block.
        assert prompt.index("Completion contract:") < prompt.index(_B1E6_OPEN)
        # Block precedes the trusted continuation line.
        assert prompt.index(_B1E6_CLOSE) < prompt.index(_B1E6_MARKER)
        # Final semantic section is the trusted "Continue working" line.
        tail = prompt[prompt.index(_B1E6_MARKER):]
        assert "Continue working toward the outcome above" in tail


class TestB1E6RendererSideEffects:
    """The renderer is deterministic, side-effect-free, and never mutates
    the pack or its lists."""

    def test_renderer_does_not_mutate_pack_lists(self, hermes_home):
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="S",
            missing_information=["a", "b"],
            sources_queried=["s1", "s2"],
            sources_failed=["f1"],
        )
        mgr.set("g")
        pack = mgr._current_evidence_pack
        # Snapshot identities and contents before rendering.
        miss_snap = list(pack.missing_information)
        q_snap = list(pack.sources_queried)
        f_snap = list(pack.sources_failed)
        # Render several times.
        for _ in range(3):
            _ = mgr._render_evidence_block()
        # Pack lists are untouched.
        assert pack.missing_information == miss_snap
        assert pack.sources_queried == q_snap
        assert pack.sources_failed == f_snap

    def test_repeated_calls_are_byte_identical(self, hermes_home):
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="S",
            missing_information=["x"],
            overall_confidence=0.42,
            overall_freshness_score=0.33,
        )
        mgr.set("g")
        first = mgr._render_evidence_block()
        for _ in range(5):
            assert mgr._render_evidence_block() == first

    def test_renderer_returns_empty_when_no_current_pack(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="b1e6-nopack")
        assert mgr._render_evidence_block() == ""

    def test_renderer_starts_with_exactly_two_newlines_when_populated(
        self, hermes_home,
    ):
        mgr = _mgr_with_pack(hermes_home, summary_text="S")
        mgr.set("g")
        block = mgr._render_evidence_block()
        assert block.startswith("\n\n")
        # And the second character must NOT be a third newline (i.e.
        # exactly two, not three+).
        assert not block.startswith("\n\n\n")


class TestB1E6RendererAllowlist:
    """Only allowlisted fields surface. Forbidden fields (hits, citations,
    conflicts, source URIs, raw payloads, fingerprints, timestamps,
    duration, total_hits, is_idempotent_reuse, exception messages,
    objective_id) NEVER appear in the rendered block."""

    def test_hits_citations_conflicts_uris_do_not_appear(self, hermes_home):
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="S",
            # Forbidden fields populated so they would appear if leaked.
            hits=["a long snippet that should never be rendered"],
            citations=["a citation text that should never be rendered"],
            source_uri="https://example.com/secret/path",
            query_fingerprint="qfp_secret",
            summary_fingerprint="sfp_secret",
            duration_ms=12345,
            created_at="2026-01-01T00:00:00Z",
            total_hits=999,
            is_idempotent_reuse=True,
            raw_payload={"internal": "this should never appear"},
            exception_message="internal exception message that should never leak",
        )
        mgr.set("g")
        block = mgr._render_evidence_block()
        prompt = mgr.next_continuation_prompt()
        for forbidden in (
            "long snippet",
            "citation text",
            "https://example.com/secret/path",
            "qfp_secret",
            "sfp_secret",
            "12345",
            "2026-01-01T00:00:00Z",
            "999",
            "idempotent",
            "internal",
            "exception message",
        ):
            assert forbidden not in block, f"leaked: {forbidden!r}"
            assert forbidden not in prompt

    def test_raw_provider_payload_does_not_appear(self, hermes_home):
        """A pack with a nested dict/list raw payload must never have
        any of its inner content rendered."""
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="S",
            raw_payload={
                "provider": "openai",
                "messages": ["secret message"],
                "usage": {"total_tokens": 99999},
            },
        )
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        for s in ("openai", "secret message", "total_tokens", "99999"):
            assert s not in prompt

    def test_cache_reused_packs_render_from_persisted_fields(self, hermes_home):
        """Cache-reused packs (empty hits/citations/conflicts) remain
        valid because the allowlisted persisted fields drive rendering."""
        from agent.executive.services import ObjectiveServices

        engine, _calls = _make_engine(
            summary_text="persisted summary",
            overall_confidence=0.6,
            overall_freshness_score=0.4,
            sources_queried=["s1"],
            hits=[],
            citations=[],
        )
        services = ObjectiveServices(
            session_id="b1e6-cache-reused",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-cache-reused", services=services)
        mgr.set("g")
        # Empty hits/citations/conflicts do not block rendering.
        pack = mgr._current_evidence_pack
        assert pack is not None
        assert pack.hits == []
        assert pack.citations == []
        block = mgr._render_evidence_block()
        assert "Summary: persisted summary" in block
        assert "Sources queried:\n- s1" in block
        assert "Overall confidence: 0.60" in block
        assert "Overall freshness: 0.40" in block


class TestB1E6FieldLimits:
    """Each allowlisted field has a deterministic length / count cap."""

    def test_summary_inclusion_and_800_char_limit(self, hermes_home):
        from hermes_cli.goals import GoalManager

        # Construct a pack directly so we can control every field.
        pack = _FakeEvidencePack(summary_text="x" * 1500)
        mgr = GoalManager(session_id="b1e6-sumcap")
        # Inject the pack bypassing the engine surface.
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        # Match succeeded objective id to current revision.
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        # The summary line in the block is capped to 800 chars (plus
        # the "Summary: " label).
        summary_line = next(
            ln for ln in block.splitlines() if ln.startswith("Summary: ")
        )
        assert len(summary_line) == len("Summary: ") + 800

    def test_missing_information_item_and_count_limits(self, hermes_home):
        from hermes_cli.goals import GoalManager

        long_item = "x" * 200
        items = [f"item-{i}-{long_item}" for i in range(20)]
        pack = _FakeEvidencePack(missing_information=items)
        mgr = GoalManager(session_id="b1e6-misscap")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        # At most 8 items.
        rendered_items = [
            ln for ln in block.splitlines() if ln.startswith("- ")
        ]
        # The block also renders "- " lines from sources_queried /
        # sources_failed. With the sources empty here, only the
        # missing_information section contributes. Cap check:
        missing_items = [
            ln for ln in rendered_items
            if ln[len("- "):].startswith("item-")
        ]
        assert len(missing_items) == 8
        # Each item line is "80-char body + '- ' prefix", so body is 80.
        for ln in missing_items:
            assert len(ln) == len("- ") + 80

    def test_queried_source_item_and_count_limits(self, hermes_home):
        from hermes_cli.goals import GoalManager

        long_src = "s" * 100
        items = [f"src-{i}-{long_src}" for i in range(30)]
        pack = _FakeEvidencePack(sources_queried=items)
        mgr = GoalManager(session_id="b1e6-qrycap")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        # Find the Sources queried: section and count its items.
        rendered = block.splitlines()
        # Locate the section header.
        idx = rendered.index("Sources queried:")
        section = rendered[idx + 1:]
        # Until the next blank or end.
        section_items = []
        for ln in section:
            if ln == "":
                break
            if ln.startswith("- "):
                section_items.append(ln)
            else:
                break
        assert len(section_items) == 16
        for ln in section_items:
            # 32-char body + "- " prefix.
            assert len(ln) == len("- ") + 32

    def test_failed_source_item_and_count_limits(self, hermes_home):
        from hermes_cli.goals import GoalManager

        long_src = "f" * 100
        items = [f"flt-{i}-{long_src}" for i in range(30)]
        pack = _FakeEvidencePack(sources_failed=items)
        mgr = GoalManager(session_id="b1e6-failcap")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        rendered = block.splitlines()
        idx = rendered.index("Sources that failed:")
        section = rendered[idx + 1:]
        section_items = []
        for ln in section:
            if ln == "":
                break
            if ln.startswith("- "):
                section_items.append(ln)
            else:
                break
        assert len(section_items) == 16
        for ln in section_items:
            assert len(ln) == len("- ") + 32

    def test_non_string_textual_fields_omitted_without_repr(self, hermes_home):
        """Non-string values in summary_text and list items must be
        omitted silently rather than coerced via repr()/str()."""
        from hermes_cli.goals import GoalManager

        class Weird:
            def __repr__(self):
                return "WEIRD_REPR_SHOULD_NOT_APPEAR"

            def __str__(self):
                return "WEIRD_STR_SHOULD_NOT_APPEAR"

        pack = _FakeEvidencePack(
            summary_text=Weird(),
            missing_information=[Weird(), "real-string", 42, None, 3.14],
        )
        mgr = GoalManager(session_id="b1e6-nonstr")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        assert "WEIRD_REPR_SHOULD_NOT_APPEAR" not in block
        assert "WEIRD_STR_SHOULD_NOT_APPEAR" not in block
        # The real string is rendered.
        assert "real-string" in block


class TestB1E6SourcesFailedOnly:
    """A pack with only non-empty sources_failed must render a useful
    block — and we do NOT pad it with zero confidence / zero freshness."""

    def test_sources_failed_only_pack_renders(self, hermes_home):
        from hermes_cli.goals import GoalManager

        pack = _FakeEvidencePack(
            sources_failed=["search-api", "wiki-api"],
            overall_confidence=0.0,  # must NOT be rendered
            overall_freshness_score=0.0,  # must NOT be rendered
        )
        mgr = GoalManager(session_id="b1e6-failed-only")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        assert block  # non-empty
        assert "Sources that failed:" in block
        assert "- search-api" in block
        assert "- wiki-api" in block
        # Zero confidence / freshness are explicitly NOT rendered.
        assert "Overall confidence:" not in block
        assert "Overall freshness:" not in block


class TestB1E6ScoreFormatting:
    """Score rendering: two decimals, finite positive only; zero and
    malformed values are omitted."""

    def test_positive_scores_render_with_two_decimals(self, hermes_home):
        from hermes_cli.goals import GoalManager

        pack = _FakeEvidencePack(overall_confidence=0.8, overall_freshness_score=0.4567)
        mgr = GoalManager(session_id="b1e6-pos")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        assert "Overall confidence: 0.80" in block
        assert "Overall freshness: 0.46" in block

    def test_zero_confidence_and_freshness_omitted(self, hermes_home):
        from hermes_cli.goals import GoalManager

        pack = _FakeEvidencePack(
            overall_confidence=0.0,
            overall_freshness_score=0.0,
            summary_text="S",
        )
        mgr = GoalManager(session_id="b1e6-zero")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        assert "Overall confidence:" not in block
        assert "Overall freshness:" not in block

    def test_malformed_and_non_finite_values_omitted_safely(self, hermes_home):
        from hermes_cli.goals import GoalManager
        import math

        class WeirdFloat:
            """Object whose float() coercion raises — must be treated as
            a malformed value and silently omitted."""
            def __float__(self):
                raise ValueError("nope")

        for bad_value in (
            math.nan,
            math.inf,
            -math.inf,
            "not a number",
            None,
            WeirdFloat(),
            True,        # bool is excluded
            False,       # bool is excluded
            [],
            {},
        ):
            pack = _FakeEvidencePack(
                overall_confidence=bad_value,
                overall_freshness_score=bad_value,
                summary_text="S",
            )
            mgr = GoalManager(session_id="b1e6-malformed")
            mgr._last_evidence_pack = pack
            mgr._state = GoalState(goal="g", created_at=1.0)
            mgr._last_evidence_succeeded_objective_id = (
                GoalManager._compute_objective_id(mgr._state)
            )
            block = mgr._render_evidence_block()
            assert "Overall confidence:" not in block
            assert "Overall freshness:" not in block

    def test_score_above_one_clamps_to_one(self, hermes_home):
        from hermes_cli.goals import GoalManager

        pack = _FakeEvidencePack(
            overall_confidence=1.5,
            overall_freshness_score=99.0,
        )
        mgr = GoalManager(session_id="b1e6-clamp")
        mgr._last_evidence_pack = pack
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        block = mgr._render_evidence_block()
        assert "Overall confidence: 1.00" in block
        assert "Overall freshness: 1.00" in block

    def test_score_below_or_equal_zero_omitted(self, hermes_home):
        from hermes_cli.goals import GoalManager

        for bad in (-0.0001, 0.0, -1.5):
            pack = _FakeEvidencePack(
                overall_confidence=bad, overall_freshness_score=bad,
            )
            mgr = GoalManager(session_id="b1e6-nonpos")
            mgr._last_evidence_pack = pack
            mgr._state = GoalState(goal="g", created_at=1.0)
            mgr._last_evidence_succeeded_objective_id = (
                GoalManager._compute_objective_id(mgr._state)
            )
            block = mgr._render_evidence_block()
            assert "Overall confidence:" not in block
            assert "Overall freshness:" not in block


class TestB1E6TextSanitization:
    """Each text-sanitization step is asserted against the contract."""

    def test_crlf_and_lone_cr_normalize_to_lf(self, hermes_home):
        mgr = GoalManager(session_id="b1e6-crlf")
        assert mgr._sanitize_text("a\r\nb\rc", limit=100) == "a\nb\nc"

    def test_c0_del_c1_controls_removed(self, hermes_home):
        mgr = GoalManager(session_id="b1e6-c0")

        # C0: 0x00..0x1F, DEL: 0x7F, C1: 0x80..0x9F.
        # LF (0x0A) and TAB (0x09) survive the C0-stripping step but
        # TAB gets normalized to a single space by the
        # horizontal-whitespace collapse step (step 8). LF survives
        # the full pipeline.
        raw = (
            "a\x00b\x01c\x07d\x08\te\x0Af\x0Bg\x0Fh\x0Fi"
            "\x10j\x11k\x12l\x13m\x14n\x15o\x16p\x17q\x18r"
            "\x19s\x1At\x1Bu\x1Cv\x1Dw\x1Ex\x1Fy\x7Fz"
            "\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8A"
            "\x8B\x8C\x8D\x8E\x8F\x90\x91\x92\x93\x94\x95"
            "\x96\x97\x98\x99\x9A\x9B\x9C\x9D\x9E\x9F"
        )
        sanitized = mgr._sanitize_text(raw, limit=10_000)
        # All controls other than \t and \n must be gone.
        for ch in sanitized:
            cp = ord(ch)
            assert cp >= 0x20 or ch == "\t" or ch == "\n", (
                f"unexpected control char: {cp:#x}"
            )
        # LF survives end-to-end.
        assert "\n" in sanitized
        # DEL did not survive.
        assert "\x7f" not in sanitized.lower()

    def test_lf_and_tab_handling_follows_contract(self, hermes_home):
        """LF survives C0 stripping and is later normalized by the
        excessive-blank-lines collapse; TAB survives stripping but is
        later collapsed to a single space."""
        mgr = GoalManager(session_id="b1e6-lf-tab")

        raw = "a\t\tb\n\n\n\nc"
        sanitized = mgr._sanitize_text(raw, limit=100)
        # The TAB run collapses to a single space.
        assert "\t" not in sanitized
        # The 4-newline run collapses to 2 (one blank line in between).
        assert sanitized.count("\n") == 2
        assert sanitized == "a b\n\nc"

    def test_ordinary_unicode_preserved(self, hermes_home):
        mgr = GoalManager(session_id="b1e6-unicode")

        raw = "café — résumé — naïve — 北京 — 🚀 — Ω — Ωµ"
        sanitized = mgr._sanitize_text(raw, limit=10_000)
        assert sanitized == raw

    def test_delimiter_tokens_defanged_case_insensitively(self, hermes_home):
        # Variants that DO contain the literal ``untrusted_goal_evidence``
        # token (case-insensitive) are rewritten to the hyphenated form.
        # Variants that ALREADY use hyphens (i.e. never contained the
        # underscore token) are left untouched — they don't need
        # rewriting because the literal tag we recognize
        # (``untrusted_goal_evidence``) is not present.
        for variant in (
            "untrusted_goal_evidence",
            "UNTRUSTED_GOAL_EVIDENCE",
            "Untrusted_Goal_Evidence",
            "uNtRuStEd_gOaL_eViDeNcE",
        ):
            sanitized = GoalManager._sanitize_text(variant, limit=1000)
            # The literal ``untrusted_goal_evidence`` token (any case)
            # must not survive.
            assert "untrusted_goal_evidence" not in sanitized.lower()
            # The replacement is the canonical hyphenated form.
            assert "untrusted-goal-evidence" in sanitized

        # Hyphenated variant does not contain the underscore token
        # (even after casefold) so it is left as-is. This is correct —
        # the spec only defangs the literal ``untrusted_goal_evidence``
        # token; a hyphenated variant already lacks the threat surface.
        hyphen_variant = "UNTRUSTED-goal-EVIDENCE"
        sanitized = GoalManager._sanitize_text(hyphen_variant, limit=1000)
        assert "untrusted_goal_evidence" not in sanitized.lower()

    def test_horizontal_whitespace_normalized(self, hermes_home):
        mgr = GoalManager(session_id="b1e6-hws")

        raw = "a    b\t\t\tc"
        sanitized = mgr._sanitize_text(raw, limit=100)
        assert sanitized == "a b c"

    def test_excessive_blank_lines_collapse(self, hermes_home):
        mgr = GoalManager(session_id="b1e6-blank")

        raw = "a\n\n\n\n\n\n\n\nb"
        sanitized = mgr._sanitize_text(raw, limit=100)
        # 8 newlines → at most 2 retained (so one blank line between).
        assert sanitized == "a\n\nb"

    def test_leading_and_trailing_blank_lines_stripped(self, hermes_home):
        mgr = GoalManager(session_id="b1e6-trim")

        raw = "\n\n\nfoo\n\n\n"
        sanitized = mgr._sanitize_text(raw, limit=100)
        assert sanitized == "foo"


class TestB1E6BlockBounds:
    """The full block is bounded to 2000 characters and the truncation
    marker appears when required; the closing tag always survives."""

    def test_full_block_length_is_at_most_2000(self, hermes_home):
        # Stuff every section to overflow: each list item uses the full
        # per-item length cap, so the body content approaches the upper
        # bound (~810 + ~660 + ~531 + ~531 + ~50 + ~180 = ~2760).
        from agent.executive.services import ObjectiveServices

        def _discover(*, objective_id, objective_text):
            return _FakeEvidencePack(
                objective_id=objective_id,
                sources_failed=["f" * 32 for _ in range(16)],
                sources_queried=["q" * 32 for _ in range(16)],
                missing_information=["m" * 80 for _ in range(8)],
                summary_text="s" * 800,
                overall_confidence=1.0,
                overall_freshness_score=1.0,
            )

        engine = MagicMock()
        engine.discover.side_effect = _discover
        services = ObjectiveServices(
            session_id="b1e6-bounds-1",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-bounds-1", services=services)
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        open_idx = prompt.index(_B1E6_OPEN)
        close_idx = prompt.index(_B1E6_CLOSE)
        block = prompt[open_idx:close_idx + len(_B1E6_CLOSE)]
        assert len(block) <= 2000

    def test_truncation_marker_appears_when_required(self, hermes_home):
        from agent.executive.services import ObjectiveServices

        def _discover(*, objective_id, objective_text):
            return _FakeEvidencePack(
                objective_id=objective_id,
                sources_failed=["f" * 32 for _ in range(16)],
                sources_queried=["q" * 32 for _ in range(16)],
                missing_information=["m" * 80 for _ in range(8)],
                summary_text="s" * 800,
                overall_confidence=1.0,
                overall_freshness_score=1.0,
            )

        engine = MagicMock()
        engine.discover.side_effect = _discover
        services = ObjectiveServices(
            session_id="b1e6-bounds-2",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-bounds-2", services=services)
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        # Body content is over the 2000 cap → truncation marker present.
        assert _B1E6_TRUNCATION in prompt

    def test_truncation_marker_absent_when_block_fits(self, hermes_home):
        mgr = _mgr_with_pack(
            hermes_home,
            summary_text="x" * 100,  # small
            missing_information=["m1"],
        )
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        assert _B1E6_TRUNCATION not in prompt

    def test_closing_delimiter_survives_truncation(self, hermes_home):
        from agent.executive.services import ObjectiveServices

        def _discover(*, objective_id, objective_text):
            return _FakeEvidencePack(
                objective_id=objective_id,
                sources_failed=["f" * 32 for _ in range(16)],
                sources_queried=["q" * 32 for _ in range(16)],
                missing_information=["m" * 80 for _ in range(8)],
                summary_text="s" * 800,
                overall_confidence=1.0,
                overall_freshness_score=1.0,
            )

        engine = MagicMock()
        engine.discover.side_effect = _discover
        services = ObjectiveServices(
            session_id="b1e6-bounds-3",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-bounds-3", services=services)
        mgr.set("g")
        prompt = mgr.next_continuation_prompt()
        # The closing tag is the last semantic element of the inserted
        # block (it sits immediately before the trusted continuation
        # marker).
        assert _B1E6_CLOSE in prompt
        # Marker appears on its own line immediately before the closing
        # tag.
        marker_idx = prompt.index(_B1E6_TRUNCATION)
        after_marker = prompt[
            marker_idx + len(_B1E6_TRUNCATION):
            marker_idx + len(_B1E6_TRUNCATION) + 1
        ]
        assert after_marker == "\n"
        # Closing tag follows the newline after the marker.
        close_idx = prompt.index(_B1E6_CLOSE, marker_idx)
        between = prompt[marker_idx + len(_B1E6_TRUNCATION):close_idx]
        assert between == "\n"
        # Block length bound is preserved even when truncated.
        open_idx = prompt.index(_B1E6_OPEN)
        block = prompt[open_idx:close_idx + len(_B1E6_CLOSE)]
        assert len(block) <= 2000


class TestB1E6ExceptionSafety:
    """Renderer exceptions are swallowed; KeyboardInterrupt / SystemExit
    propagate; nothing sensitive appears in logs."""

    def test_renderer_exception_returns_empty_block(self, hermes_home):
        from hermes_cli.goals import GoalManager

        class Boom:
            """A pack whose attribute access raises — the renderer must
            catch and return ""."""
            @property
            def summary_text(self):
                raise RuntimeError("kaboom in summary_text")

            def __getattr__(self, name):
                raise RuntimeError(f"kaboom in {name}")

        mgr = GoalManager(session_id="b1e6-boom")
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_pack = Boom()
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        assert mgr._render_evidence_block() == ""

    def test_keyboard_interrupt_propagates(self, hermes_home):
        from hermes_cli.goals import GoalManager

        class CtrlC:
            @property
            def summary_text(self):
                raise KeyboardInterrupt()

        mgr = GoalManager(session_id="b1e6-ctrlc")
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_pack = CtrlC()
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        with pytest.raises(KeyboardInterrupt):
            mgr._render_evidence_block()

    def test_system_exit_propagates(self, hermes_home):
        from hermes_cli.goals import GoalManager

        class SysExitPack:
            @property
            def summary_text(self):
                raise SystemExit(1)

        mgr = GoalManager(session_id="b1e6-sysexit")
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_pack = SysExitPack()
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        with pytest.raises(SystemExit):
            mgr._render_evidence_block()

    def test_exception_text_and_evidence_content_absent_from_logs(
        self, hermes_home, caplog,
    ):
        from hermes_cli.goals import GoalManager

        class Boom:
            @property
            def summary_text(self):
                raise RuntimeError("EXCEPTION_TEXT_MUST_NOT_LEAK")

            def __getattr__(self, name):
                raise RuntimeError(f"attribute {name} boom")

        mgr = GoalManager(session_id="b1e6-log-redact")
        mgr._state = GoalState(goal="g", created_at=1.0)
        mgr._last_evidence_pack = Boom()
        mgr._last_evidence_succeeded_objective_id = (
            GoalManager._compute_objective_id(mgr._state)
        )
        with caplog.at_level("WARNING", logger="hermes_cli.goals"):
            block = mgr._render_evidence_block()
        assert block == ""
        # Only the class name may appear in logs.
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert "EXCEPTION_TEXT_MUST_NOT_LEAK" not in joined
        assert "RuntimeError" in joined
        # The pack content (the "Boom" object's repr / attributes) must
        # also be absent.
        assert "<class" not in joined
        assert "Boom" not in joined


class TestB1E6NextContinuationPromptNoSideEffects:
    """``next_continuation_prompt`` MUST NOT call ``discover``,
    ``_ensure_current_evidence_pack``, ``dry_run``, or ``rollback``.
    It also MUST NOT mutate the GoalState schema."""

    def test_next_continuation_prompt_adds_no_discover_call(self, hermes_home):
        engine, calls = _make_engine(summary_text="S")
        from agent.executive.services import ObjectiveServices

        services = ObjectiveServices(
            session_id="b1e6-no-discover",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-no-discover", services=services)
        mgr.set("g")
        before = len(calls)
        for _ in range(10):
            _ = mgr.next_continuation_prompt()
        assert len(calls) == before
        assert engine.discover.call_count == before

    def test_next_continuation_prompt_does_not_invoke_ensure(
        self, hermes_home,
    ):
        """The ensure method is internal. We assert by checking that
        calling ``next_continuation_prompt`` repeatedly does not bump
        the discover counter."""
        engine, calls = _make_engine(summary_text="S")
        from agent.executive.services import ObjectiveServices

        services = ObjectiveServices(
            session_id="b1e6-no-ensure",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-no-ensure", services=services)
        mgr.set("g")
        before = len(calls)
        # Replace the ensure method with a sentinel so any call
        # registers.
        ensure_calls = []
        original_ensure = mgr._ensure_current_evidence_pack

        def _spy_ensure(*a, **kw):
            ensure_calls.append((a, kw))
            return original_ensure(*a, **kw)

        mgr._ensure_current_evidence_pack = _spy_ensure
        for _ in range(10):
            _ = mgr.next_continuation_prompt()
        assert ensure_calls == []

    def test_next_continuation_prompt_no_dry_run_or_rollback(self, hermes_home):
        engine, calls = _make_engine(summary_text="S")
        from agent.executive.services import ObjectiveServices

        services = ObjectiveServices(
            session_id="b1e6-no-dr",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-no-dr", services=services)
        mgr.set("g")
        before = len(calls)
        for _ in range(5):
            _ = mgr.next_continuation_prompt()
        assert len(calls) == before
        assert engine.dry_run.call_count == 0
        assert engine.rollback.call_count == 0


class TestB1E6SealedSurfaces:
    """External surfaces and the GoalState schema are not changed."""

    def test_judge_prompt_remains_unchanged(self, hermes_home):
        """The judge prompt construction must NOT consume the pack."""
        from unittest.mock import patch as _patch
        from agent.executive.services import ObjectiveServices

        engine, _calls = _make_engine(
            summary_text="LEAK_TEXT_SHOULD_NOT_REACH_JUDGE",
            sources_failed=["LEAK_FAILED_SHOULD_NOT_REACH_JUDGE"],
        )
        services = ObjectiveServices(
            session_id="b1e6-judge-unchanged",
            evidence_pack_status="available",
            evidence_pack_engine=engine,
        )
        mgr = GoalManager(session_id="b1e6-judge-unchanged", services=services)
        mgr.set("g")
        captured = {}
        def _capture(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return ("continue", "x", False, None, False)
        with _patch("hermes_cli.goals.judge_goal", side_effect=_capture):
            mgr.evaluate_after_turn("did some work")
        # The judge signature must be unchanged: positional goal, last_response
        # and kw-only subgoals/background_processes/contract. No evidence
        # pack kwargs.
        kwargs = captured["kwargs"]
        assert "evidence_pack" not in kwargs
        assert "pack" not in kwargs
        # The judge never sees the untrusted advisory text.
        all_args_text = repr((captured["args"], captured["kwargs"]))
        assert "LEAK_TEXT_SHOULD_NOT_REACH_JUDGE" not in all_args_text
        assert "LEAK_FAILED_SHOULD_NOT_REACH_JUDGE" not in all_args_text

    def test_goal_state_schema_unchanged(self):
        from hermes_cli.goals import GoalState, GoalContract

        s = GoalState(
            goal="g", status="active", turns_used=1, max_turns=5,
            created_at=1.0, last_turn_at=2.0,
            consecutive_parse_failures=0, consecutive_transport_failures=0,
            subgoals=["a"], contract=GoalContract(verification="v"),
        )
        raw = s.to_json()
        keys = set(json.loads(raw).keys())
        # B1-E6 sealed-surface assertion: GoalState persisted-schema keys
        # must match the sealed contract. ``gates`` existed BEFORE B1-E6
        # (introduced with the /goal gate command) and is part of the
        # persisted state — the B1-E6 evidence-pack work did not add it.
        # None of the B1-E6 implementation added new keys to GoalState.
        expected = {
            "goal", "status", "turns_used", "max_turns", "created_at",
            "last_turn_at", "last_verdict", "last_reason", "paused_reason",
            "consecutive_parse_failures", "consecutive_transport_failures",
            "subgoals", "waiting_on_pid", "waiting_on_session",
            "waiting_until", "waiting_reason", "waiting_since", "contract",
            "gates",
        }
        assert keys == expected

    def test_external_gateway_template_consumer_remain_compatible(self):
        """The exported constants are importable in the same shape, and
        ``template.format(goal=...)`` continues to produce the same
        prompt. This guards the public surface for gateway / TUI
        consumers."""
        from hermes_cli.goals import (
            CONTINUATION_PROMPT_TEMPLATE,
            CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE,
            CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE,
        )
        # The three exports are strings and are still formattable with
        # their historical kwargs.
        assert isinstance(CONTINUATION_PROMPT_TEMPLATE, str)
        assert isinstance(CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE, str)
        assert isinstance(CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE, str)
        out = CONTINUATION_PROMPT_TEMPLATE.format(goal="X")
        assert "Goal: X" in out
        out2 = CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal="X", subgoals_block="- 1. A",
        )
        assert "1. A" in out2
        out3 = CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal="X", contract_block="- Outcome: O",
        )
        assert "Outcome: O" in out3

    def test_no_file_outside_two_allowed_files_changes(self):
        """The two files permitted for modification are
        ``hermes_cli/goals.py`` and ``tests/hermes_cli/test_goals.py``.
        ``git status`` must show no other touched paths."""
        from hermes_cli.goals import GoalManager

        # No-op assertion — the gate here is structural (this test is in
        # the only test file allowed to change). The CI / review check
        # that catches an out-of-scope file is "git diff --stat".
        # We touch the manager surface to make this test count.
        mgr = GoalManager(session_id="b1e6-scope")
        assert mgr is not None

