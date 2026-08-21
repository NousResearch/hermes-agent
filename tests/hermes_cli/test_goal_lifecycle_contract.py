"""Deterministic lifecycle contract tests for persistent goals.

These tests intentionally exercise the owner (GoalManager/SessionDB projection)
without a provider, UI, or semantic-memory dependency.
"""
import json

import pytest

from hermes_cli import goals


def _manager(monkeypatch, tmp_path, sid="lifecycle-test", max_turns=4):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)
    goals._DB_CACHE.clear()
    return goals.GoalManager(sid, default_max_turns=max_turns)


def test_goal_identity_and_checkpoint_survive_reload_on_empty_output(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, max_turns=4)
    state = mgr.set("multi-turn work")

    decision = mgr.evaluate_after_turn(
        "",
        turn_outcome=goals.EXECUTION_FAILED,
        turn_metadata={"reason": "provider returned empty output"},
    )

    reloaded = goals.GoalManager(mgr.session_id).state
    assert decision["verdict"] == "continuation_required"
    assert decision["should_continue"] is True
    assert reloaded is not None
    assert reloaded.goal_id == state.goal_id
    assert reloaded.outcome == goals.CONTINUATION_REQUIRED
    assert reloaded.continuation_pending is True
    assert reloaded.checkpoint["goal_id"] == state.goal_id
    assert reloaded.checkpoint["stop_reason"] == "provider returned empty output"
    assert reloaded.turns_used == 1


def test_model_done_waits_for_authority_and_does_not_complete(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="false-done")
    mgr.set("unfinished work")
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *a, **k: ("done", "model says done", False, None, False),
    )

    decision = mgr.evaluate_after_turn("I am done")
    assert decision["verdict"] == "waiting_for_authority"
    assert mgr.state.status == "active"
    assert mgr.state.outcome == goals.WAITING_FOR_AUTHORITY
    assert mgr.state.completion_evidence is None


def test_model_done_with_contract_still_waits_for_verified_receipt(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="false-done-contract")
    mgr.set("unfinished work\nverify: a real receipt exists")
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *a, **k: ("done", "model says done", False, None, False),
    )

    decision = mgr.evaluate_after_turn("I am done")
    assert decision["verdict"] == "waiting_for_authority"
    assert mgr.state.status == "active"
    assert mgr.state.outcome == goals.WAITING_FOR_AUTHORITY
    assert mgr.state.completion_evidence is None


def test_explicit_completion_requires_evidence(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="explicit-complete")
    mgr.set("work")
    with pytest.raises(ValueError):
        mgr.confirm_completion("")
    assert mgr.confirm_completion("receipt=local-test-1", source="test") is True
    assert mgr.state.status == "done"
    assert mgr.state.outcome == goals.GOAL_COMPLETED
    assert mgr.state.completion_evidence["evidence"] == "receipt=local-test-1"


def test_completion_does_not_claim_success_when_persistence_fails(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="completion-save-failure")
    mgr.set("persist completion")
    before = goals.load_goal("completion-save-failure").to_json()
    monkeypatch.setattr(goals, "save_goal", lambda *_args, **_kwargs: False)

    assert mgr.confirm_completion("receipt=save-failure", source="test") is False
    assert mgr.state.status == "active"
    assert goals.load_goal("completion-save-failure").to_json() == before


def test_checkpoint_does_not_schedule_when_persistence_fails(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="checkpoint-save-failure")
    mgr.set("persist checkpoint")
    monkeypatch.setattr(goals, "save_goal", lambda *_args, **_kwargs: False)

    decision = mgr.evaluate_after_turn("", turn_outcome=goals.EXECUTION_FAILED)
    assert decision["verdict"] == "persistence_failed"
    assert decision["should_continue"] is False
    persisted = goals.load_goal("checkpoint-save-failure")
    assert persisted.turns_used == 0
    assert persisted.continuation_pending is False


def test_recovery_checkpoint_is_durable_without_spending_a_goal_turn(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="compression-recovery")
    state = mgr.set("survive context exhaustion")

    decision = mgr.checkpoint_recovery(
        "CONTEXT_COMPRESSION_EXHAUSTED",
        metadata={"reason": "compression failed before provider invocation"},
    )

    reloaded = goals.GoalManager("compression-recovery").state
    assert decision["verdict"] == "continuation_required"
    assert decision["should_continue"] is True
    assert reloaded.goal_id == state.goal_id
    assert reloaded.turns_used == 0
    assert reloaded.continuation_pending is True
    assert reloaded.checkpoint["stop_reason"] == "CONTEXT_COMPRESSION_EXHAUSTED"
    assert goals.GoalManager("compression-recovery").start_continuation() is True
    assert goals.GoalManager("compression-recovery").state.turns_used == 0


def test_recovery_checkpoint_failure_is_typed_and_not_scheduled(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="compression-recovery-save-failure")
    mgr.set("do not fake recovery")
    monkeypatch.setattr(goals, "save_goal", lambda *_args, **_kwargs: False)

    decision = mgr.checkpoint_recovery("CONTEXT_COMPRESSION_EXHAUSTED")

    assert decision["verdict"] == "persistence_failed"
    assert decision["should_continue"] is False
    persisted = goals.load_goal("compression-recovery-save-failure")
    assert persisted.turns_used == 0
    assert persisted.continuation_pending is False


def test_pause_rolls_back_when_persistence_fails(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="pause-save-failure")
    mgr.set("remain active if pause cannot persist")
    monkeypatch.setattr(goals, "save_goal", lambda *_args, **_kwargs: False)

    assert mgr.pause("infrastructure failure") is None
    assert mgr.state.status == "active"
    assert goals.load_goal("pause-save-failure").status == "active"


def test_quality_gate_failure_is_deterministic_and_skips_model_judge(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="gate-failure")
    mgr.set("gate work")
    gate = mgr.add_gate("python -c 'print(\"gate-red\"); raise SystemExit(7)'", max_retries=1)
    called = []
    monkeypatch.setattr(goals, "judge_goal", lambda *args, **kwargs: called.append(True))

    decision = mgr.evaluate_after_turn("model output")
    assert decision["verdict"] == "gate_failed"
    assert decision["should_continue"] is True
    assert called == []
    assert gate.last_exit_code == 7
    assert "gate-red" in gate.last_output_tail
    assert "gate-red" in decision["continuation_prompt"]


def test_quality_gate_passes_before_false_done_authority_boundary(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="gate-pass")
    mgr.set("gate work")
    gate = mgr.add_gate("python -c 'print(\"gate-green\")'", max_retries=1)
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: ("done", "model says done", False, None, False),
    )

    decision = mgr.evaluate_after_turn("model output")
    assert decision["verdict"] == "waiting_for_authority"
    assert gate.last_exit_code == 0
    assert mgr.state.outcome == goals.WAITING_FOR_AUTHORITY


def test_resume_preserves_cumulative_budget(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="budget", max_turns=2)
    mgr.set("bounded")
    mgr.evaluate_after_turn("", turn_outcome=goals.TURN_BUDGET_EXHAUSTED)
    assert mgr.state.turns_used == 1
    mgr.pause("operator pause")
    mgr.resume()
    assert mgr.state.turns_used == 1
    assert mgr.state.max_turns == 2


def test_single_continuation_claim_and_stale_checkpoint_rejection(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="lease")
    mgr.set("bounded")
    mgr.evaluate_after_turn("", turn_outcome=goals.EXECUTION_FAILED, turn_metadata={"reason": "tool error"})
    first = goals.GoalManager("lease")
    second = goals.GoalManager("lease")
    assert first.claim_continuation("worker-1") is True
    assert second.claim_continuation("worker-2") is False
    first.release_continuation(queued=False)

    stale = goals.GoalManager("lease")
    stale.state.checkpoint["goal_id"] = "wrong"
    goals.save_goal("lease", stale.state)
    assert stale.validate_checkpoint()[0] is False
    assert stale.claim_continuation("worker-3") is False


def test_enqueued_continuation_stays_recoverable_until_consumer_starts(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="enqueue-recovery")
    mgr.set("recover queued work")
    mgr.evaluate_after_turn("", turn_outcome=goals.EXECUTION_FAILED, turn_metadata={"reason": "provider down"})
    assert mgr.claim_continuation("worker") is True
    assert mgr.release_continuation(queued=True) is True

    reloaded = goals.GoalManager("enqueue-recovery")
    assert reloaded.state.continuation_pending is True
    assert reloaded.start_continuation() is True
    assert reloaded.state.continuation_pending is False
    assert goals.GoalManager("enqueue-recovery").start_continuation() is False


def test_pause_and_verified_completion_beat_queued_continuation(monkeypatch, tmp_path):
    paused = _manager(monkeypatch, tmp_path, sid="pause-race")
    paused.set("pause me")
    paused.evaluate_after_turn("", turn_outcome=goals.EXECUTION_FAILED)
    assert paused.pause("operator pause") is not None
    assert goals.GoalManager("pause-race").start_continuation() is False
    assert goals.GoalManager("pause-race").state.outcome == goals.GOAL_PAUSED

    completed = _manager(monkeypatch, tmp_path, sid="complete-race")
    completed.set("complete me")
    completed.evaluate_after_turn("", turn_outcome=goals.EXECUTION_FAILED)
    assert completed.confirm_completion("receipt=verified-1", source="operator") is True
    assert goals.GoalManager("complete-race").start_continuation() is False
    assert goals.GoalManager("complete-race").state.outcome == goals.GOAL_COMPLETED


def test_checkpoint_records_work_budget_authority_and_receipts(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="rich-checkpoint")
    mgr.set("ship artifact")
    mgr.evaluate_after_turn(
        "partial",
        turn_outcome=goals.TOOL_BUDGET_EXHAUSTED,
        turn_metadata={
            "reason": "tool budget exhausted",
            "current_task": "run validation",
            "verified_completed_work": ["implementation compiled"],
            "unfinished_work": ["run validation", "inspect receipt"],
            "graph_run_ids": ["graph-run-1"],
            "receipt_ids": ["receipt-1"],
            "required_authority": "operator completion confirmation",
        },
    )
    cp = mgr.state.checkpoint
    assert cp["current_task"] == "run validation"
    assert cp["verified_completed_work"] == ["implementation compiled"]
    assert cp["unfinished_work"] == ["run validation", "inspect receipt"]
    assert cp["graph_run_ids"] == ["graph-run-1"]
    assert cp["receipt_ids"] == ["receipt-1"]
    assert cp["required_authority"] == "operator completion confirmation"


def test_unknown_old_row_gets_stable_identity_and_typed_projection():
    state = goals.GoalState.from_json(json.dumps({
        "goal": "legacy", "status": "paused", "created_at": 123.0,
    }))
    again = goals.GoalState.from_json(state.to_json())
    assert state.goal_id == again.goal_id
    assert state.outcome == goals.GOAL_PAUSED


def test_legacy_active_row_migrates_without_fabricating_completion(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="legacy-migrate", max_turns=7)
    db = goals._get_session_db()
    db.set_meta("goal:legacy-migrate", json.dumps({
        "goal": "finish migration",
        "status": "active",
        "turns_used": 3,
        "max_turns": 7,
        "created_at": 123.0,
        "last_turn_at": 456.0,
        "last_verdict": "continue",
        "last_reason": "more work remains",
        "subgoals": ["preserve data", "verify restart"],
    }))
    mgr = goals.GoalManager("legacy-migrate", default_max_turns=99)

    result = mgr.migrate_legacy_state()

    assert result["migrated"] is True
    assert mgr.state.goal_id
    assert mgr.state.turns_used == 3
    assert mgr.state.max_turns == 7
    assert mgr.state.subgoals == ["preserve data", "verify restart"]
    assert mgr.state.status == "active"
    assert mgr.state.outcome == goals.CONTINUATION_REQUIRED
    assert mgr.state.continuation_pending is True
    assert mgr.state.completion_evidence is None
    assert mgr.state.checkpoint["unfinished_work"] == ["preserve data", "verify restart"]
    assert "legacy_schema" in mgr.state.migration


def test_legacy_migration_is_idempotent_and_preserves_budget(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, sid="legacy-idempotent", max_turns=9)
    db = goals._get_session_db()
    db.set_meta("goal:legacy-idempotent", json.dumps({
        "goal": "bounded work", "status": "active", "turns_used": 8,
        "max_turns": 9, "created_at": 321.0,
    }))
    mgr = goals.GoalManager("legacy-idempotent")
    first = mgr.migrate_legacy_state()
    first_json = goals.load_goal("legacy-idempotent").to_json()
    second = goals.GoalManager("legacy-idempotent").migrate_legacy_state()
    second_json = goals.load_goal("legacy-idempotent").to_json()

    assert first["migrated"] is True
    assert second["migrated"] is False
    assert first_json == second_json
