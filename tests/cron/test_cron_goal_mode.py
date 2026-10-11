"""Cron ``/goal`` prompts run a bounded, judge-terminated goal loop."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cron.scheduler_goal import (
    cron_goal_session_id,
    goal_prompt_from_job,
    run_goal_turns,
)
from cron.scheduler_prompt import _CRON_HINT, _GOAL_CRON_HINT, _build_job_prompt


def test_goal_prompt_from_job_keeps_plain_cron_prompts_unchanged():
    assert goal_prompt_from_job({"prompt": "summarize the inbox"}) is None
    assert goal_prompt_from_job({"prompt": "/goal ship the release"}) == "ship the release"
    assert goal_prompt_from_job({"prompt": "/goal\tship the release"}) == "ship the release"
    assert goal_prompt_from_job({"prompt": "/goal"}) is None


def test_cron_goal_session_is_stable_per_job():
    assert cron_goal_session_id("nightly") == "cron-goal:nightly"


def test_goal_first_turn_uses_goal_adapted_cron_hint_without_losing_assembly_context(monkeypatch):
    """Goal fires keep ordinary assembly inputs but not its one-shot delivery contract."""
    monkeypatch.setattr(
        "cron.scheduler_prompt._load_cron_skill_parts",
        lambda _job, _names: ["SKILL INSTRUCTIONS"],
    )
    monkeypatch.setattr(
        "cron.notepad.render_notepad_section",
        lambda _job_id: "## Cron Notepad\nremember this\n\n",
    )

    prompt = _build_job_prompt(
        {
            "id": "goal-job",
            "prompt": "ship the release",
            "skills": ["release-skill"],
            "script": "collect-context.sh",
        },
        prerun_script=(True, "SCRIPT OUTPUT"),
        extra_prompt="RUN CONTEXT",
        runtime_data_prompt="MONITOR CONTEXT",
        cron_hint=_GOAL_CRON_HINT,
    )

    assert "SKILL INSTRUCTIONS" in prompt
    assert "SCRIPT OUTPUT" in prompt
    assert "RUN CONTEXT" in prompt
    assert "MONITOR CONTEXT" in prompt
    assert "remember this" in prompt
    assert _GOAL_CRON_HINT in prompt
    assert _CRON_HINT not in prompt
    assert "bounded goal loop" in prompt
    assert "without inventing earlier progress" in prompt
    assert "imply a new goal or a reset turn budget" in prompt
    assert 'respond with exactly "[SILENT]"' not in _GOAL_CRON_HINT
    assert "final response will be automatically delivered" not in _GOAL_CRON_HINT


def test_ordinary_cron_prompt_keeps_the_original_hint_bytes():
    assert _CRON_HINT in _build_job_prompt({"prompt": "check for updates"})


def test_prepare_goal_prompt_selects_goal_adapted_hint(monkeypatch):
    import cron.scheduler as scheduler

    captured = {}
    monkeypatch.setattr("hermes_cli.config.require_parseable_user_config", lambda: None)
    monkeypatch.setattr(
        scheduler,
        "_build_job_prompt",
        lambda job, **kwargs: captured.update(job=job, **kwargs) or "assembled goal prompt",
    )

    early, prompt = scheduler._prepare_job_prompt(
        {"id": "goal-job", "prompt": "/goal ship the release"},
        "goal-job",
        "goal job",
        None,
        None,
    )

    assert early is None
    assert prompt == "assembled goal prompt"
    assert captured["job"]["prompt"] == "ship the release"
    assert captured["cron_hint"] == _GOAL_CRON_HINT


def test_goal_watchdog_passes_prior_turn_history_to_the_agent(monkeypatch):
    import cron.scheduler as scheduler

    calls = []

    class Agent:
        def run_conversation(self, prompt, *, task_id, conversation_history=None):
            calls.append((prompt, task_id, conversation_history))
            return {"messages": [{"role": "assistant", "content": prompt}]}

    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    first = scheduler._run_agent_with_watchdog(
        Agent(), "assembled context", {}, "goal-job", "goal job", "cron:goal-job", None,
    )
    scheduler._run_agent_with_watchdog(
        Agent(), "continue", {}, "goal-job", "goal job", "cron:goal-job", None,
        conversation_history=first["messages"],
    )

    assert calls == [
        ("assembled context", "cron:goal-job", None),
        ("continue", "cron:goal-job", [{"role": "assistant", "content": "assembled context"}]),
    ]


def test_run_goal_turns_continues_until_the_judge_finishes(monkeypatch):
    decisions = iter((
        {"should_continue": True, "continuation_prompt": "continue", "message": "keep going"},
        {"should_continue": False, "continuation_prompt": None, "message": "goal achieved"},
    ))
    prompts = []

    class Manager:
        state = None

        def is_active(self):
            return False

        def has_goal(self):
            return False

        def set(self, goal):
            assert goal == "ship the release"

        def evaluate_after_turn(self, response, **_kwargs):
            return next(decisions)

    def run_turn(prompt):
        prompts.append(prompt)
        return {"final_response": f"response {len(prompts)}"}

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context", run_turn=run_turn,
        response_from_result=lambda result: result["final_response"],
    )

    assert prompts == ["assembled initial context", "continue"]
    assert result["final_response"] == "response 2"
    assert response == "response 2"
    assert status == "goal achieved"


def test_run_goal_turns_does_not_burn_a_turn_while_a_goal_is_parked():
    class Manager:
        state = type("State", (), {"goal": "ship the release"})()

        def is_active(self):
            return True

        def is_waiting(self):
            return True

        def has_goal(self):
            return True

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda _prompt: AssertionError(),
        response_from_result=lambda result: result["final_response"],
    )

    assert result == {}
    assert response == ""
    assert "parked" in status


def test_run_goal_turns_restores_context_after_a_wait_barrier_clears():
    prompts = []
    decisions = iter((
        {"should_continue": True, "continuation_prompt": "continuation only", "message": "keep going"},
        {"should_continue": False, "continuation_prompt": None, "message": "done"},
    ))

    class Manager:
        state = type("State", (), {"goal": "ship the release"})()

        def is_active(self):
            return True

        def is_waiting(self):
            # This goal was parked on fire A, but its barrier is now clear on
            # fire B. The fire must retain B's freshly assembled context.
            return False

        def has_goal(self):
            return True

        def evaluate_after_turn(self, _response, **_kwargs):
            return next(decisions)

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda prompt: prompts.append(prompt) or {"final_response": "response"},
        response_from_result=lambda result: result["final_response"],
    )

    assert prompts == ["assembled initial context", "continuation only"]
    assert result["final_response"] == response == "response"
    assert status == "done"


def test_run_goal_turns_preserves_a_paused_goal():
    class Manager:
        state = type("State", (), {"status": "paused"})()

        def is_active(self):
            return False

        def has_goal(self):
            return True

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda _prompt: AssertionError(),
        response_from_result=lambda result: result["final_response"],
    )

    assert result == {}
    assert response == ""
    assert "paused" in status


def test_run_goal_turns_preserves_a_completed_goal():
    class Manager:
        state = type("State", (), {"status": "done"})()

        def is_active(self):
            return False

        def has_goal(self):
            return False

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda _prompt: AssertionError(),
        response_from_result=lambda result: result["final_response"],
    )

    assert result == {}
    assert response == ""
    assert "complete" in status


def test_run_goal_turns_replaces_a_changed_active_goal_with_assembled_context():
    prompts = []

    class Manager:
        state = type("State", (), {"goal": "old goal"})()

        def is_active(self):
            return True

        def has_goal(self):
            return True

        def set(self, goal):
            assert goal == "ship the release"

        def evaluate_after_turn(self, _response, **_kwargs):
            return {"should_continue": False, "continuation_prompt": None, "message": "done"}

    result, response, status = run_goal_turns(
        Manager(), "ship the release", initial_prompt="assembled initial context",
        run_turn=lambda prompt: prompts.append(prompt) or {"final_response": "response"},
        response_from_result=lambda result: result["final_response"],
    )

    assert prompts == ["assembled initial context"]
    assert result["final_response"] == response == "response"
    assert status == "done"


def test_goal_execution_claim_blocks_overlap_until_the_running_claim_finishes(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    first = executions.create_execution("goal-job", source="builtin")
    second = executions.create_execution("goal-job", source="builtin")
    other = executions.create_execution("other-goal-job", source="builtin")

    assert executions.mark_execution_running(first["id"], exclusive_job=True) is not None
    assert executions.mark_execution_running(second["id"], exclusive_job=True) is None
    assert executions.mark_execution_running(other["id"], exclusive_job=True) is not None
    assert executions.finish_execution(first["id"], success=True) is not None
    assert executions.mark_execution_running(second["id"], exclusive_job=True) is not None


def test_goal_execution_adoption_blocks_overlap_until_the_running_claim_finishes(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    first = executions.create_execution("goal-job", source="builtin")
    second = executions.create_execution("goal-job", source="builtin")
    assert executions.mark_execution_handoff_pending(first["id"]) is not None
    assert executions.mark_execution_handoff_pending(second["id"]) is not None

    assert executions.adopt_claimed_execution(first["id"], exclusive_job=True) is not None
    assert executions.adopt_claimed_execution(second["id"], exclusive_job=True) is None
    assert executions.finish_execution(first["id"], success=True) is not None
    assert executions.adopt_claimed_execution(second["id"], exclusive_job=True) is not None



@pytest.fixture
def goal_runtime(monkeypatch, tmp_path):
    """Real isolated stores/GoalManager, with no model or production task execution."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    import hermes_cli.goals as goals
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "goal-state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    monkeypatch.setattr(goals, "judge_goal", lambda *_a, **_kw: ("continue", "artifact saved", False, None, False))
    monkeypatch.setattr(scheduler, "_prepare_job_prompt", lambda *_a: (None, "assembled context"))
    monkeypatch.setattr(scheduler, "_CronRunScope", lambda *_a: SimpleNamespace(
        enter=lambda: None, exit=lambda: None, workdir=None, task_id="test-goal"))
    monkeypatch.setattr(scheduler, "_reload_dotenv_and_publish_delivery_target", lambda *_a: None)
    monkeypatch.setattr(scheduler, "_load_cron_job_config", lambda *_a: SimpleNamespace(cfg={}, model="mock"))
    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", lambda *_a: SimpleNamespace(
        blocked=None, model="mock", fallback_notice=None))
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda *_a: None)
    monkeypatch.setattr(scheduler, "_FireAudit", lambda *_a: MagicMock())
    monkeypatch.setattr(scheduler, "_teardown_cron_agent", lambda *_a: None)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    with jobs.use_cron_store(tmp_path / "profile"):
        job = jobs.create_job("/goal ship the release", "every 1h")
        manager = goals.GoalManager(cron_goal_session_id(job["id"]), default_max_turns=5)
        manager.set("ship the release")
        manager.state.subgoals = ["preserve the built artifact"]
        goals.save_goal(manager.session_id, manager.state)
        calls = []
        def install_agent(action):
            class Agent:
                def run_conversation(self, prompt, *, task_id, conversation_history=None):
                    calls.append((prompt, conversation_history))
                    action()
                    return {"final_response": "artifact saved", "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "artifact saved"},
                    ]}
            monkeypatch.setattr(scheduler, "_construct_cron_agent", lambda *_a, **_kw: Agent())
        yield SimpleNamespace(jobs=jobs, goals=goals, scheduler=scheduler, job=job,
                              manager=manager, calls=calls, install_agent=install_agent)
    db.close()


def test_goal_cancelled_before_first_turn_preserves_existing_progress(goal_runtime):
    r = goal_runtime
    r.manager.state.turns_used = 2
    r.goals.save_goal(r.manager.session_id, r.manager.state)
    cancel = threading.Event()
    cancel.set()
    r.install_agent(lambda: pytest.fail("cancelled goal started an agent turn"))
    success, _, response, error = r.scheduler.run_job(r.job, cancel_event=cancel)
    state = r.goals.load_goal(r.manager.session_id)
    assert success and error is None and "interrupted" in response.lower()
    assert r.calls == []
    assert state.status == "paused" and state.turns_used == 2
    assert state.subgoals == ["preserve the built artifact"]


@pytest.mark.parametrize("stop", ["pause", "disabled", "paused_at", "state", "deleted", "unreadable", "cancel"])
def test_goal_stops_before_second_turn_from_current_persisted_state(goal_runtime, stop, monkeypatch):
    r = goal_runtime
    cancel = threading.Event()
    def persist_pause_marker():
        # UI/CLI half-paused records still gate execution even with enabled=True.
        records = r.jobs.load_jobs()
        records[0].update({stop: "paused" if stop == "state" else "2026-10-10T00:00:00Z"})
        r.jobs.save_jobs(records)

    actions = {
        "pause": lambda: r.jobs.pause_job(r.job["id"], reason="operator pause"),
        "disabled": lambda: r.jobs.update_job(r.job["id"], {"enabled": False}),
        "paused_at": persist_pause_marker,
        "state": persist_pause_marker,
        "deleted": lambda: r.jobs.remove_job(r.job["id"]),
        "unreadable": lambda: r.jobs._current_cron_store().jobs_file.write_text("{broken json"),
        "cancel": cancel.set,
    }
    stop_job = actions[stop]
    r.install_agent(stop_job)
    if stop == "cancel":
        # Model a completed turn whose cancel event arrives at the return boundary,
        # after the watchdog has handed back its result.
        monkeypatch.setattr(r.scheduler, "_run_agent_with_watchdog", lambda agent, prompt, *_a, **_kw:
                            agent.run_conversation(prompt, task_id="test-goal"))
    success, _, response, error = r.scheduler.run_job(r.job, cancel_event=cancel)
    state = r.goals.load_goal(r.manager.session_id)
    assert success and error is None
    assert len(r.calls) == 1
    assert "artifact saved" in response and "interrupted" in response.lower()
    assert state.status == "paused" and state.turns_used == 1 and state.max_turns == 5
    assert state.last_reason == "artifact saved"
    assert state.subgoals == ["preserve the built artifact"]
    assert r.job["enabled"] is True  # stale dispatch snapshot never mutated


def test_resume_job_requires_explicit_goal_resume_and_keeps_total_budget(goal_runtime):
    r = goal_runtime
    r.install_agent(lambda: r.jobs.pause_job(r.job["id"]))
    assert r.scheduler.run_job(r.job)[0]
    paused = r.goals.load_goal(r.manager.session_id)
    assert paused.status == "paused" and paused.turns_used == 1
    r.jobs.resume_job(r.job["id"])
    r.install_agent(lambda: None)
    assert r.scheduler.run_job(r.jobs.get_job(r.job["id"]))[0]
    assert len(r.calls) == 1
    assert r.goals.load_goal(r.manager.session_id).to_json() == paused.to_json()
    manager = r.goals.GoalManager(r.manager.session_id)
    manager.resume(reset_budget=False)
    r.install_agent(lambda: r.jobs.pause_job(r.job["id"]))
    assert r.scheduler.run_job(r.jobs.get_job(r.job["id"]))[0]
    resumed = r.goals.load_goal(r.manager.session_id)
    assert len(r.calls) == 2
    assert r.calls[1][1] is None  # a resumed fire has no in-fire history, but the goal is not new
    assert resumed.turns_used == 2 and resumed.max_turns == 5
    assert resumed.goal == paused.goal and resumed.subgoals == paused.subgoals


def test_non_goal_cron_remains_one_shot(goal_runtime):
    r = goal_runtime
    plain = r.jobs.update_job(r.job["id"], {"prompt": "summarize the inbox"})
    original = r.goals.load_goal(r.manager.session_id).to_json()
    r.install_agent(lambda: r.jobs.pause_job(r.job["id"]))
    success, _, response, error = r.scheduler.run_job(plain)
    assert success and error is None and response == "artifact saved"
    assert len(r.calls) == 1 and r.calls[0][1] is None
    assert r.goals.load_goal(r.manager.session_id).to_json() == original



def test_watchdog_cancellation_preserves_failure_and_pauses_goal(goal_runtime):
    r = goal_runtime
    cancel = threading.Event()
    r.manager.state.turns_used = 2
    r.goals.save_goal(r.manager.session_id, r.manager.state)
    r.install_agent(cancel.set)
    success, _, _, error = r.scheduler.run_job(r.job, cancel_event=cancel)
    state = r.goals.load_goal(r.manager.session_id)
    assert not success and error is not None
    assert len(r.calls) == 1
    assert state.status == "paused" and state.turns_used == 2
    assert state.subgoals == ["preserve the built artifact"]



@pytest.mark.parametrize("verdict", ["wait", "done"])
def test_job_pause_during_judge_preserves_wait_or_completed_work(goal_runtime, monkeypatch, verdict):
    r = goal_runtime
    def judge(*_a, **_kw):
        r.jobs.pause_job(r.job["id"])
        wait = {"seconds": 3600} if verdict == "wait" else None
        return verdict, "artifact saved", False, wait, False
    monkeypatch.setattr(r.goals, "judge_goal", judge)
    r.install_agent(lambda: None)
    assert r.scheduler.run_job(r.job)[0]
    state = r.goals.load_goal(r.manager.session_id)
    assert len(r.calls) == 1 and state.turns_used == 1
    assert state.status == ("paused" if verdict == "wait" else "done")
    assert state.last_verdict == verdict
    assert state.subgoals == ["preserve the built artifact"]
