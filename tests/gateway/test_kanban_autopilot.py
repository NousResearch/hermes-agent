"""Regression tests for the Kanban-first /autopilot surface."""

from __future__ import annotations

import json


def test_autopilot_status_imports_and_reports_degraded_effective_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway_autopilot_state.json").write_text(
        json.dumps({"version": 1, "enabled": True, "updated_by": "test"}),
        encoding="utf-8",
    )

    from gateway.kanban_autopilot import handle_autopilot_command

    result = handle_autopilot_command("status", actor="tester")

    assert result.ok is True
    assert result.decision["desired_mode"] == "enabled"
    assert result.decision["effective_mode"] in {"blocked", "degraded"}
    assert result.decision["effective_mode"] != "enabled"
    assert result.decision["state_file_enabled_is_execution_proof"] is False
    assert "desired_mode=enabled" in result.message
    assert "effective_mode=" in result.message
    assert "State file enabled=true is not execution proof" in result.message


def test_autopilot_status_is_read_only_and_does_not_touch_dispatch_or_kanban(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("status path must not dispatch, claim, spawn, or mutate Kanban")

    monkeypatch.setattr(kanban_autopilot, "dispatch_once", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "claim_task", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "spawn_worker", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "mutate_kanban", forbidden, raising=False)

    result = kanban_autopilot.handle_autopilot_command("status", actor="tester")

    assert result.ok is True
    assert result.decision["read_only"] is True
    assert result.decision["mutations_attempted"] == []
    assert calls == []


def test_autopilot_control_actions_persist_desired_state_without_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    result = handle_autopilot_command("on", actor="tester")

    assert result.ok is True
    assert result.fail_closed is False
    assert result.decision["desired_mode"] == "on"
    assert result.decision["effective_mode"] == "blocked"
    assert result.decision["state_file_enabled_is_execution_proof"] is False
    assert result.decision["mutations_attempted"] == []
    state = json.loads((tmp_path / "gateway_autopilot_state.json").read_text(encoding="utf-8"))
    assert state["desired_mode"] == "on"
    assert state["enabled"] is True
    assert state["updated_by"] == "tester"


def test_autopilot_pause_and_focus_update_controller_state_without_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("controller state changes must not dispatch, claim, spawn, or mutate Kanban")

    monkeypatch.setattr(kanban_autopilot, "dispatch_once", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "claim_task", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "spawn_worker", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "mutate_kanban", forbidden, raising=False)

    pause = kanban_autopilot.handle_autopilot_command("pause waiting-for-review", actor="tester")
    focus = kanban_autopilot.handle_autopilot_command("focus BO-076", actor="tester")
    status = kanban_autopilot.handle_autopilot_command("status", actor="tester")

    assert pause.ok is True
    assert pause.decision["desired_mode"] == "paused"
    assert pause.decision["pause_reason"] == "waiting-for-review"
    assert focus.ok is True
    assert focus.decision["focus"] == "BO-076"
    assert status.decision["desired_mode"] == "paused"
    assert status.decision["focus"] == "BO-076"
    assert status.decision["effective_mode"] == "paused"
    assert status.decision["mutations_attempted"] == []
    assert calls == []


def test_autopilot_stop_clears_enabled_without_claiming_runtime_proof(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    handle_autopilot_command("on", actor="tester")
    result = handle_autopilot_command("stop", actor="tester")

    assert result.ok is True
    assert result.decision["desired_mode"] == "stopped"
    assert result.decision["effective_mode"] == "stopped"
    assert result.decision["state_file_enabled_is_execution_proof"] is False
    state = json.loads((tmp_path / "gateway_autopilot_state.json").read_text(encoding="utf-8"))
    assert state["desired_mode"] == "stopped"
    assert state["enabled"] is False


def test_ready_gate_rejects_raw_kanban_ready_without_executable_contract():
    from gateway.kanban_autopilot import evaluate_autopilot_ready_gate

    candidate = {
        "id": "t_raw",
        "public_id": "BO-999",
        "status": "ready",
        "title": "raw ready task",
        "body": "Please handle this later.",
    }

    result = evaluate_autopilot_ready_gate(candidate)

    assert result["autopilot_ready"] is False
    assert result["status"] == "rejected"
    assert "missing_goal" in result["reason_codes"]
    assert "missing_acceptance_criteria" in result["reason_codes"]
    assert "missing_verification_requirements" in result["reason_codes"]
    assert result["human_reason"]


def test_ready_gate_accepts_native_contract_but_never_claims_or_spawns(monkeypatch):
    from gateway import kanban_autopilot

    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("ready-gate dry-run must not claim, spawn, dispatch, or mutate Kanban")

    monkeypatch.setattr(kanban_autopilot, "dispatch_once", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "claim_task", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "spawn_worker", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "mutate_kanban", forbidden, raising=False)
    candidate = {
        "id": "t_good",
        "public_id": "BO-100",
        "status": "ready",
        "title": "Implement bounded thing",
        "body": """
Goal: implement a bounded local validator.
End-state/output: local commit candidate and Kanban evidence.
Scope/non-goals: no gateway restart and no production action.
Acceptance criteria: validator returns machine-readable reason codes.
Verification requirements: focused tests and git diff --check.
Authority boundary: Kanban BO-100 controls execution; PR/push forbidden.
Repo/lane truth: chriskim12/hermes-agent on a task branch.
Risk flags: no env, secret, prod, billing, customer-visible, or restart action.
Dependencies/blockers: none.
Review package expectation: changed files, tests, commit, boundaries.
""",
        "routing_verdict": {"verdict": "Hermes direct"},
        "admission_snapshot": {"repo_full_name": "chriskim12/hermes-agent"},
    }

    result = kanban_autopilot.evaluate_autopilot_ready_gate(candidate)

    assert result["autopilot_ready"] is True
    assert result["status"] == "accepted"
    assert result["reason_codes"] == []
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}
    assert calls == []


def test_autopilot_queue_dry_run_reports_gate_results_without_spawning(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    result = handle_autopilot_command("queue", actor="tester", candidates=[
        {"id": "t_raw", "public_id": "BO-999", "status": "ready", "title": "raw", "body": "raw"}
    ])

    assert result.ok is True
    assert result.decision["read_only"] is True
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}
    assert result.decision["candidates"][0]["autopilot_ready"] is False
    assert "missing_goal" in result.decision["candidates"][0]["reason_codes"]


def _ready_candidate(public_id: str = "BO-100") -> dict:
    return {
        "id": "t_good",
        "public_id": public_id,
        "status": "ready",
        "title": "Implement bounded thing",
        "body": """
Goal: implement a bounded local validator.
End-state/output: local commit candidate and Kanban evidence.
Scope/non-goals: no gateway restart and no production action.
Acceptance criteria: validator returns machine-readable reason codes.
Verification requirements: focused tests and git diff --check.
Authority boundary: Kanban controls execution; PR/push forbidden.
Repo/lane truth: chriskim12/hermes-agent on a task branch.
Risk flags: no env, secret, prod, billing, customer-visible, or restart action.
Dependencies/blockers: none.
Review package expectation: changed files, tests, commit, boundaries.
""",
        "routing_verdict": {"verdict": "Hermes direct"},
        "admission_snapshot": {"repo_full_name": "chriskim12/hermes-agent"},
    }


def test_dispatcher_eligibility_bridge_requires_controller_on_and_ready_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import evaluate_dispatcher_eligibility, handle_autopilot_command

    stopped = evaluate_dispatcher_eligibility([_ready_candidate()])
    assert stopped["eligible"] == []
    assert stopped["ineligible"][0]["reason_codes"] == ["autopilot_effective_mode_not_blocked_pending_dispatch_gate"]

    handle_autopilot_command("on", actor="tester")
    verdict = evaluate_dispatcher_eligibility([_ready_candidate(), {"id": "t_raw", "public_id": "BO-999", "status": "ready", "body": "raw"}])

    assert [item["public_id"] for item in verdict["eligible"]] == ["BO-100"]
    assert verdict["ineligible"][0]["public_id"] == "BO-999"
    assert "missing_goal" in verdict["ineligible"][0]["reason_codes"]
    assert verdict["handoff_target"] == "existing_kanban_dispatcher"
    assert verdict["second_dispatcher_created"] is False
    assert verdict["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}


def test_autopilot_queue_includes_dispatcher_eligibility_without_dispatching(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("eligibility bridge must not call the dispatcher")

    monkeypatch.setattr(kanban_autopilot, "dispatch_once", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "claim_task", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "spawn_worker", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "mutate_kanban", forbidden, raising=False)

    kanban_autopilot.handle_autopilot_command("on", actor="tester")
    result = kanban_autopilot.handle_autopilot_command("queue", actor="tester", candidates=[_ready_candidate("BO-101")])

    assert result.ok is True
    assert result.decision["dispatcher_eligibility"]["eligible"][0]["public_id"] == "BO-101"
    assert result.decision["dispatcher_eligibility"]["handoff_target"] == "existing_kanban_dispatcher"
    assert result.decision["dispatcher_eligibility"]["second_dispatcher_created"] is False
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}
    assert calls == []


def test_review_ready_contract_requires_pr_and_checks_without_merging():
    from gateway.kanban_autopilot import evaluate_review_ready_contract

    missing = evaluate_review_ready_contract({"work_id": "BO-081", "commit": "abc1234"})
    assert missing["review_ready"] is False
    assert "missing_pr_url" in missing["reason_codes"]
    assert "missing_checks_passed" in missing["reason_codes"]
    assert missing["merge_allowed"] is False
    assert missing["release_allowed"] is False

    satisfied = evaluate_review_ready_contract({
        "work_id": "BO-081",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "yuuka/bo-081-autopilot-review-ready-contract",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/123",
        "pr_base": "main",
        "pr_head": "yuuka/bo-081-autopilot-review-ready-contract",
        "checks_passed": True,
        "worktree_clean": True,
        "kanban_worker_done": True,
        "boundaries_confirmed": True,
    })
    assert satisfied["review_ready"] is True
    assert satisfied["reason_codes"] == []
    assert satisfied["merge_allowed"] is False
    assert satisfied["release_allowed"] is False


def test_review_ready_contract_rejects_wrong_repo_and_release_base():
    from gateway.kanban_autopilot import evaluate_review_ready_contract

    result = evaluate_review_ready_contract({
        "work_id": "BO-081",
        "repo_full_name": "evil/repo",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "feature",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/123",
        "pr_base": "prod",
        "pr_head": "other",
        "checks_passed": True,
        "worktree_clean": True,
        "kanban_worker_done": True,
        "boundaries_confirmed": True,
    }, expected_repo_full_name="chriskim12/hermes-agent")

    assert result["review_ready"] is False
    assert "repo_full_name_mismatch" in result["reason_codes"]
    assert "release_base_requires_separate_approval" in result["reason_codes"]
    assert "pr_head_task_branch_mismatch" in result["reason_codes"]
    assert result["merge_allowed"] is False


def test_hard_stop_pauses_lane_and_blocks_eligibility_until_recovered(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import evaluate_dispatcher_eligibility, handle_autopilot_command

    handle_autopilot_command("on", actor="tester")
    stopped = handle_autopilot_command("hard-stop prod_action_detected", actor="tester")
    assert stopped.ok is True
    assert stopped.decision["desired_mode"] == "hard_stopped"
    assert stopped.decision["effective_mode"] == "hard_stop"
    assert stopped.decision["dispatch_blocked"] is True
    assert stopped.decision["operator_recovery_required"] is True

    verdict = evaluate_dispatcher_eligibility([_ready_candidate("BO-082")])
    assert verdict["eligible"] == []
    assert verdict["ineligible"][0]["reason_codes"] == ["autopilot_hard_stop_active"]
    assert verdict["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}

    recovered = handle_autopilot_command("recover acknowledge-no-prod-action", actor="tester")
    assert recovered.ok is True
    assert recovered.decision["desired_mode"] == "paused"
    assert recovered.decision["effective_mode"] == "paused"
    assert recovered.decision["dispatch_blocked"] is True


def test_lane_pause_blocks_matching_tenant_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import evaluate_dispatcher_eligibility, handle_autopilot_command

    handle_autopilot_command("on", actor="tester")
    paused = handle_autopilot_command("pause-lane autopilot overloaded", actor="tester")
    assert paused.ok is True
    assert paused.decision["paused_lanes"] == ["autopilot"]

    blocked = _ready_candidate("BO-082")
    blocked["tenant"] = "autopilot"
    allowed = _ready_candidate("BO-083")
    allowed["tenant"] = "other"
    verdict = evaluate_dispatcher_eligibility([blocked, allowed])

    assert [item["public_id"] for item in verdict["eligible"]] == ["BO-083"]
    assert verdict["ineligible"][0]["public_id"] == "BO-082"
    assert verdict["ineligible"][0]["reason_codes"] == ["autopilot_lane_paused"]
    assert verdict["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}


def test_closed_loop_operating_contract_documents_state_caps_boundaries_and_future_ralplan_gate():
    from gateway.kanban_autopilot import get_closed_loop_operating_contract

    contract = get_closed_loop_operating_contract()

    assert contract["adr"] == "bounded_controller_not_executor"
    assert contract["authority_ceiling"] == "review_ready_pr"
    assert contract["dispatcher_boundary"]["autopilot_may_directly_claim_or_spawn"] is False
    assert contract["dispatcher_boundary"]["execution_owner"] == "existing_kanban_dispatcher"
    assert contract["state_machine"]["allowed_states"] == [
        "disabled",
        "dry_run",
        "single_flight",
        "bounded_multi_tick",
        "parent_scoped",
        "lane_scoped",
        "paused",
        "hard_stopped",
        "needs_human",
    ]
    assert contract["default_caps"]["max_dispatches_per_tick"] == 1
    assert contract["default_caps"]["max_open_autopilot_prs"] == 2
    assert "gateway_restart_reload" in contract["forbidden_without_current_approval"]
    assert "config_env_secret_provider_billing_pricing_mutation" in contract["forbidden_without_current_approval"]
    assert "policy_file_invalid_or_stale" in contract["stop_conditions"]
    assert "merge_release_deploy_prod_customer_visible_authority" in contract["future_ralplan_required_for"]


def test_closed_loop_policy_contract_validator_rejects_second_dispatcher_and_scope_expansion():
    from gateway.kanban_autopilot import (
        get_closed_loop_operating_contract,
        validate_closed_loop_policy_contract,
    )

    valid = validate_closed_loop_policy_contract(get_closed_loop_operating_contract())
    assert valid["ok"] is True
    assert valid["reason_codes"] == []

    unsafe = get_closed_loop_operating_contract()
    unsafe["dispatcher_boundary"] = {
        **unsafe["dispatcher_boundary"],
        "autopilot_may_directly_claim_or_spawn": True,
        "second_dispatcher_allowed": True,
    }
    unsafe["authority_ceiling"] = "merge_release_deploy"
    unsafe["scope_model"]["scope_can_silently_widen"] = True

    result = validate_closed_loop_policy_contract(unsafe)

    assert result["ok"] is False
    assert "direct_claim_or_spawn_not_allowed" in result["reason_codes"]
    assert "second_dispatcher_not_allowed" in result["reason_codes"]
    assert "authority_ceiling_must_be_review_ready_pr" in result["reason_codes"]
    assert "scope_must_not_silently_widen" in result["reason_codes"]


def test_closed_loop_simulator_selects_one_candidate_and_never_dispatches(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("read-only simulator must not dispatch, claim, spawn, or mutate Kanban")

    monkeypatch.setattr(kanban_autopilot, "dispatch_once", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "claim_task", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "spawn_worker", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "mutate_kanban", forbidden, raising=False)
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    report = kanban_autopilot.simulate_closed_loop_ticks(
        [_ready_candidate("BO-092"), {"id": "t_raw", "public_id": "BO-999", "status": "ready", "body": "raw"}],
        max_ticks=2,
    )

    assert report["mode"] == "read_only_simulation"
    assert report["would_select"][0]["public_id"] == "BO-092"
    assert report["would_handoff"][0]["target"] == "existing_kanban_dispatcher"
    assert report["would_handoff"][0]["check_only"] is True
    assert report["would_skip"][0]["public_id"] == "BO-999"
    assert "missing_goal" in report["would_skip"][0]["reason_codes"]
    assert report["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}
    assert report["mutations_attempted"] == []
    assert calls == []


def test_closed_loop_simulator_pauses_on_hard_stop_lane_pause_and_no_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command, simulate_closed_loop_ticks

    handle_autopilot_command("on", actor="tester")
    handle_autopilot_command("pause-lane autopilot overloaded", actor="tester")
    lane_candidate = _ready_candidate("BO-092")
    lane_candidate["tenant"] = "autopilot"
    lane_report = simulate_closed_loop_ticks([lane_candidate], max_ticks=2)
    assert lane_report["would_pause"][0]["reason_code"] == "no_progress"
    assert lane_report["would_skip"][0]["reason_codes"] == ["autopilot_lane_paused"]

    handle_autopilot_command("hard-stop forbidden_action", actor="tester")
    hard_report = simulate_closed_loop_ticks([_ready_candidate("BO-093")], max_ticks=2)
    assert hard_report["would_pause"][0]["reason_code"] == "hard_stop"
    assert hard_report["would_select"] == []
    assert hard_report["would_handoff"] == []

    handle_autopilot_command("recover acknowledged", actor="tester")
    blocked_report = simulate_closed_loop_ticks([{"id": "t_raw", "public_id": "BO-999", "status": "ready", "body": "raw"}], max_ticks=2)
    assert blocked_report["would_pause"][0]["reason_code"] == "no_progress"
    assert blocked_report["next_state"] == "needs_human"


def test_single_flight_activation_runs_one_check_only_handoff_and_no_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    forbidden_calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        forbidden_calls.append("forbidden")
        raise AssertionError("single-flight activation must not directly dispatch, claim, spawn, or mutate Kanban")

    monkeypatch.setattr(kanban_autopilot, "dispatch_once", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "claim_task", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "spawn_worker", forbidden, raising=False)
    monkeypatch.setattr(kanban_autopilot, "mutate_kanban", forbidden, raising=False)
    kanban_autopilot.handle_autopilot_command("on", actor="tester")
    checks: list[dict] = []

    def check_only(payload: dict) -> dict:
        checks.append(payload)
        return {"allowed": True, "reason": "mock_check_passed"}

    result = kanban_autopilot.activate_single_flight([_ready_candidate("BO-093"), _ready_candidate("BO-094")], check_only_handoff=check_only)

    assert result["status"] == "handoff_check_passed"
    assert result["selected"]["public_id"] == "BO-093"
    assert result["handoff"]["target"] == "existing_kanban_dispatcher"
    assert result["handoff"]["check_only"] is True
    assert result["handoff"]["would_dispatch"] is False
    assert result["handoff_success_is_worker_completion"] is False
    assert len(checks) == 1
    assert checks[0]["public_id"] == "BO-093"
    assert result["skipped"][0]["reason_codes"] == ["single_flight_limit_reached"]
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}
    assert forbidden_calls == []


def test_single_flight_activation_blocks_when_scope_already_has_active_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import activate_single_flight, handle_autopilot_command

    handle_autopilot_command("on", actor="tester")
    active = _ready_candidate("BO-116")
    active.update({"id": "t_active", "task_id": "t_active", "status": "running", "current_run_id": 42})
    ready = _ready_candidate("BO-117")

    result = activate_single_flight([active, ready])

    assert result["status"] == "active_flight_blocked"
    assert result["selected"] is None
    assert result["handoff"] is None
    assert result["next_state"] == "needs_human"
    assert result["handoff_success_is_worker_completion"] is False
    assert result["worker_done_observed"] is False
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}
    assert result["active_flights"] == [{"public_id": "BO-116", "task_id": "t_active", "current_run_id": 42, "claim_lock": None, "worker_pid": None}]
    assert result["skipped"][0]["public_id"] == "BO-117"
    assert result["skipped"][0]["reason_codes"] == ["active_flight_already_present"]


def test_single_flight_activation_blocks_on_failed_check_without_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import activate_single_flight, handle_autopilot_command

    handle_autopilot_command("on", actor="tester")

    result = activate_single_flight(
        [_ready_candidate("BO-093")],
        check_only_handoff=lambda payload: {"allowed": False, "reason": "mock_dispatcher_unavailable"},
    )

    assert result["status"] == "handoff_check_blocked"
    assert result["next_state"] == "needs_human"
    assert result["handoff_success_is_worker_completion"] is False
    assert result["worker_done_observed"] is False
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_closeout_progress_gate_blocks_missing_evidence_and_pr_backlog():
    from gateway.kanban_autopilot import evaluate_autopilot_closeout_progress

    missing = evaluate_autopilot_closeout_progress({"work_id": "BO-094", "kanban_worker_done": True})
    assert missing["may_continue"] is False
    assert "missing_review_ready_contract" in missing["reason_codes"]

    good = {
        "work_id": "BO-094",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "bo-094",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/123",
        "pr_base": "main",
        "pr_head": "bo-094",
        "checks_passed": True,
        "worktree_clean": True,
        "kanban_worker_done": True,
        "boundaries_confirmed": True,
    }
    backlog_blocked = evaluate_autopilot_closeout_progress(good, open_autopilot_prs=2, max_open_autopilot_prs=2)
    assert backlog_blocked["may_continue"] is False
    assert "pr_backlog_cap_reached" in backlog_blocked["reason_codes"]

    allowed = evaluate_autopilot_closeout_progress(good, open_autopilot_prs=1, max_open_autopilot_prs=2)
    assert allowed["may_continue"] is True
    assert allowed["worker_done_observed"] is True
    assert allowed["review_ready_contract"]["review_ready"] is True
    assert allowed["merge_allowed"] is False


def test_closeout_progress_gate_allows_explicit_no_code_evidence_without_pr():
    from gateway.kanban_autopilot import evaluate_autopilot_closeout_progress

    result = evaluate_autopilot_closeout_progress({
        "work_id": "BO-094",
        "no_code_task": True,
        "artifact_path": "docs/closed-loop-kanban-autopilot.md",
        "verification": "documentation reviewed and tests not applicable",
        "kanban_worker_done": True,
        "boundaries_confirmed": True,
    })

    assert result["may_continue"] is True
    assert result["review_ready_equivalent"] == "no_code_evidence"
    assert result["merge_allowed"] is False


def test_bounded_multi_tick_runs_until_cap_and_records_skipped_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command, run_bounded_multi_tick

    handle_autopilot_command("on", actor="tester")
    result = run_bounded_multi_tick([_ready_candidate("BO-095"), _ready_candidate("BO-096"), _ready_candidate("BO-097")], max_tasks=2)

    assert [item["public_id"] for item in result["executed"]] == ["BO-095", "BO-096"]
    assert result["skipped"][0]["public_id"] == "BO-097"
    assert result["skipped"][0]["reason_codes"] == ["max_tasks_per_run_reached"]
    assert result["next_state"] == "paused"
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_bounded_multi_tick_stops_on_failure_or_no_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command, run_bounded_multi_tick

    handle_autopilot_command("on", actor="tester")
    blocked = run_bounded_multi_tick([_ready_candidate("BO-095")], max_tasks=2, closeout_results=[{"may_continue": False, "reason_codes": ["missing_review_ready_contract"]}])
    assert blocked["executed"][0]["public_id"] == "BO-095"
    assert blocked["would_pause"][0]["reason_code"] == "closeout_blocked"
    assert blocked["next_state"] == "needs_human"

    no_progress = run_bounded_multi_tick([{"id": "t_raw", "public_id": "BO-999", "status": "ready", "body": "raw"}], max_tasks=2)
    assert no_progress["executed"] == []
    assert no_progress["would_pause"][0]["reason_code"] == "no_progress"
    assert no_progress["next_state"] == "needs_human"


def test_scope_filter_allows_only_parent_lane_repo_and_labels():
    from gateway.kanban_autopilot import filter_candidates_for_scope

    good = _ready_candidate("BO-096")
    good.update({"parent_public_id": "BO-090", "tenant": "autopilot", "repo_full_name": "chriskim12/hermes-agent", "labels": ["closed-loop"]})
    wrong_lane = {**good, "public_id": "BO-200", "tenant": "other"}
    wrong_parent = {**good, "public_id": "BO-201", "parent_public_id": "BO-999"}

    result = filter_candidates_for_scope([good, wrong_lane, wrong_parent], {"parent_public_id": "BO-090", "tenant": "autopilot", "repo_full_name": "chriskim12/hermes-agent", "labels": ["closed-loop"]})

    assert [item["public_id"] for item in result["in_scope"]] == ["BO-096"]
    assert result["out_of_scope"][0]["reason_codes"] == ["lane_scope_mismatch"]
    assert result["out_of_scope"][1]["reason_codes"] == ["parent_scope_mismatch"]
    assert result["scope_can_silently_widen"] is False


def test_scope_filter_marks_ambiguous_dependency_or_missing_scope_as_needs_human():
    from gateway.kanban_autopilot import filter_candidates_for_scope

    ambiguous = _ready_candidate("BO-096")
    ambiguous.update({"parent_public_id": "BO-090", "tenant": "autopilot", "relation_type": "dependency"})
    result = filter_candidates_for_scope([ambiguous], {"parent_public_id": "BO-090", "tenant": "autopilot"})

    assert result["in_scope"] == []
    assert result["out_of_scope"][0]["reason_codes"] == ["hierarchy_dependency_ambiguous"]
    assert result["next_state"] == "needs_human"


def test_operator_report_includes_selected_skipped_blocked_caps_and_next_state():
    from gateway.kanban_autopilot import generate_autopilot_run_report

    report = generate_autopilot_run_report({
        "executed": [{"public_id": "BO-097"}],
        "skipped": [{"public_id": "BO-098", "reason_codes": ["max_tasks_per_run_reached"]}],
        "would_pause": [{"reason_code": "closeout_blocked"}],
        "open_prs": ["https://github.com/chriskim12/hermes-agent/pull/1"],
        "next_state": "needs_human",
        "caps": {"max_tasks_per_run": 2},
    })

    assert report["summary"]["executed_count"] == 1
    assert report["summary"]["skipped_count"] == 1
    assert report["summary"]["blocked_count"] == 1
    assert report["summary"]["next_state"] == "needs_human"
    assert "BO-097" in report["text"]
    assert "max_tasks_per_run_reached" in report["text"]


def test_operator_report_explains_zero_work():
    from gateway.kanban_autopilot import generate_autopilot_run_report

    report = generate_autopilot_run_report({"executed": [], "skipped": [], "would_pause": [{"reason_code": "no_progress"}], "next_state": "needs_human"})

    assert report["summary"]["zero_work"] is True
    assert "zero work" in report["text"].lower()
    assert "no_progress" in report["text"]


def test_review_package_proof_summarizes_pr_readiness_without_live_authority():
    from gateway.kanban_autopilot import build_autopilot_review_package

    evidence = {
        "work_id": "BO-118",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "91d572c971956e705cbfccc19990bf1ed128c239",
        "task_branch": "bo-118-autopilot-review-package",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/118",
        "pr_base": "main",
        "pr_head": "bo-118-autopilot-review-package",
        "checks_passed": True,
        "worktree_clean": True,
        "kanban_worker_done": True,
        "boundaries_confirmed": True,
    }
    run_report = {
        "executed": [{"public_id": "BO-116"}, {"public_id": "BO-117"}],
        "skipped": [{"public_id": "BO-119", "reason_codes": ["not_in_scope_yet"]}],
        "would_pause": [],
        "next_state": "continue",
    }

    package = build_autopilot_review_package(evidence, run_report=run_report)

    assert package["review_ready"] is True
    assert package["status"] == "review_package_ready"
    assert package["work_id"] == "BO-118"
    assert package["pr"]["url"] == "https://github.com/chriskim12/hermes-agent/pull/118"
    assert package["run_report"]["summary"]["executed_count"] == 2
    assert package["authority"]["ceiling"] == "review_ready_pr"
    assert package["authority"]["merge_allowed"] is False
    assert package["authority"]["release_allowed"] is False
    assert package["authority"]["gateway_restart_reload_allowed"] is False
    assert package["authority"]["prod_customer_visible_allowed"] is False
    assert package["worker_done_observed"] is True
    assert package["handoff_success_is_worker_completion"] is False
    assert package["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}
    assert "review-ready PR package" in package["text"]
    assert "merge/release/deploy remains forbidden" in package["text"]


def test_review_package_proof_blocks_missing_pr_or_worker_evidence():
    from gateway.kanban_autopilot import build_autopilot_review_package

    package = build_autopilot_review_package({"work_id": "BO-118", "commit": "91d572c971956e705cbfccc19990bf1ed128c239"})

    assert package["review_ready"] is False
    assert package["status"] == "review_package_blocked"
    assert "worker_done_not_observed" in package["reason_codes"]
    assert "missing_review_ready_contract" in package["reason_codes"]
    assert package["authority"]["merge_allowed"] is False
    assert package["next_state"] == "needs_human"


def test_promotion_policy_allows_only_bounded_parent_scoped_check_only_after_proofs():
    from gateway.kanban_autopilot import evaluate_autopilot_promotion_policy

    result = evaluate_autopilot_promotion_policy({
        "parent_public_id": "BO-114",
        "live_pickup_smoke_passed": True,
        "single_flight_guard_passed": True,
        "review_package_ready": True,
        "kanban_worker_done_children": ["BO-116", "BO-117", "BO-118"],
        "active_flights": 0,
        "open_autopilot_prs": 0,
        "requested_mode": "bounded_multi_tick",
    })

    assert result["promotion_allowed"] is True
    assert result["promoted_mode"] == "bounded_multi_tick_check_only"
    assert result["scope"]["parent_public_id"] == "BO-114"
    assert result["caps"]["max_tasks_per_run"] == 2
    assert result["caps"]["max_active_flights"] == 1
    assert result["authority"]["worker_dispatch_claim_spawn_allowed"] is False
    assert result["authority"]["merge_allowed"] is False
    assert result["authority"]["gateway_restart_reload_allowed"] is False
    assert result["authority"]["config_env_secret_mutation_allowed"] is False
    assert result["requires_current_turn_approval_for_live_dispatch"] is True
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}
    assert result["next_state"] == "promote_check_only"


def test_promotion_policy_blocks_missing_proofs_or_authority_expansion():
    from gateway.kanban_autopilot import evaluate_autopilot_promotion_policy

    result = evaluate_autopilot_promotion_policy({
        "parent_public_id": "BO-114",
        "live_pickup_smoke_passed": True,
        "single_flight_guard_passed": False,
        "review_package_ready": False,
        "active_flights": 1,
        "open_autopilot_prs": 3,
        "requested_mode": "global_queue_draining",
        "request_live_dispatch": True,
        "request_gateway_restart_reload": True,
    })

    assert result["promotion_allowed"] is False
    assert result["promoted_mode"] == "blocked"
    assert "single_flight_guard_missing" in result["reason_codes"]
    assert "review_package_not_ready" in result["reason_codes"]
    assert "active_flight_present" in result["reason_codes"]
    assert "pr_backlog_cap_reached" in result["reason_codes"]
    assert "requested_mode_not_allowed" in result["reason_codes"]
    assert "live_dispatch_requires_current_turn_approval" in result["reason_codes"]
    assert "gateway_restart_reload_requires_current_turn_approval" in result["reason_codes"]
    assert result["next_state"] == "needs_human"
    assert result["authority"]["worker_dispatch_claim_spawn_allowed"] is False


def test_policy_hardening_rejects_forbidden_authority_expansion():
    from gateway.kanban_autopilot import get_closed_loop_operating_contract, harden_autopilot_policy

    contract = get_closed_loop_operating_contract()
    unsafe = {
        **contract,
        "authority_ceiling": "merge_release_deploy",
        "dispatcher_boundary": {**contract["dispatcher_boundary"], "autopilot_may_directly_claim_or_spawn": True},
        "forbidden_without_current_approval": [],
    }

    result = harden_autopilot_policy(unsafe)

    assert result["accepted"] is False
    assert result["next_state"] == "hard_stopped"
    assert "authority_ceiling_must_be_review_ready_pr" in result["reason_codes"]
    assert "direct_claim_or_spawn_not_allowed" in result["reason_codes"]
    assert result["recovery_required"] is True


def test_recovery_drill_covers_stale_state_worker_failure_and_policy_violation():
    from gateway.kanban_autopilot import run_autopilot_recovery_drill

    result = run_autopilot_recovery_drill([
        {"name": "stale_kanban", "trigger": "stale_kanban_state"},
        {"name": "worker_timeout", "trigger": "worker_crash_or_timeout_repeated"},
        {"name": "policy", "trigger": "policy_file_invalid_or_stale"},
    ])

    assert result["passed"] is True
    assert [item["action"] for item in result["drills"]] == ["pause_and_reread_kanban", "pause_and_require_worker_evidence", "hard_stop_and_require_recovery_ack"]
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_recovery_drill_fails_unknown_trigger_closed():
    from gateway.kanban_autopilot import run_autopilot_recovery_drill

    result = run_autopilot_recovery_drill([{"name": "surprise", "trigger": "unknown_escape"}])

    assert result["passed"] is False
    assert result["drills"][0]["action"] == "hard_stop_and_require_human_triage"
    assert result["next_state"] == "needs_human"


def test_autopilot_dry_run_command_uses_closed_loop_simulator(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    handle_autopilot_command("on", actor="tester")
    result = handle_autopilot_command("dry-run", actor="tester", candidates=[_ready_candidate("BO-090")])

    assert result.ok is True
    assert result.decision["status"] == "DRY_RUN"
    assert result.decision["closed_loop"]["would_select"][0]["public_id"] == "BO-090"
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}
    assert "would_select=1" in result.message


def test_autopilot_dry_run_loads_live_kanban_candidates_when_not_injected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    calls: list[dict] = []

    def fake_loader(*, parent_public_id=None, tenant=None, limit=50):
        calls.append({"parent_public_id": parent_public_id, "tenant": tenant, "limit": limit})
        return [_ready_candidate("BO-115")]

    monkeypatch.setattr(kanban_autopilot, "load_live_kanban_candidates", fake_loader, raising=False)
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    result = kanban_autopilot.handle_autopilot_command("dry-run BO-114", actor="tester")

    assert result.ok is True
    assert calls == [{"parent_public_id": "BO-114", "tenant": None, "limit": 50}]
    assert result.decision["closed_loop"]["would_select"][0]["public_id"] == "BO-115"
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_autopilot_once_loads_live_kanban_candidates_for_check_only_handoff(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    calls: list[dict] = []

    def fake_loader(*, parent_public_id=None, tenant=None, limit=50):
        calls.append({"parent_public_id": parent_public_id, "tenant": tenant, "limit": limit})
        return [_ready_candidate("BO-115")]

    monkeypatch.setattr(kanban_autopilot, "load_live_kanban_candidates", fake_loader, raising=False)
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    result = kanban_autopilot.handle_autopilot_command("once BO-114", actor="tester")

    assert result.ok is True
    assert calls == [{"parent_public_id": "BO-114", "tenant": None, "limit": 50}]
    assert result.decision["status"] == "CHECK_ONLY_HANDOFF_READY"
    assert result.decision["single_flight"]["selected"]["public_id"] == "BO-115"
    assert result.decision["single_flight"]["handoff"]["check_only"] is True
    assert result.decision["single_flight"]["handoff"]["would_dispatch"] is False


def test_autopilot_once_command_uses_single_flight_check_only_not_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    handle_autopilot_command("on", actor="tester")
    result = handle_autopilot_command("once", actor="tester", candidates=[_ready_candidate("BO-090"), _ready_candidate("BO-091")])

    assert result.ok is True
    assert result.decision["status"] == "CHECK_ONLY_HANDOFF_READY"
    assert result.decision["single_flight"]["selected"]["public_id"] == "BO-090"
    assert result.decision["single_flight"]["handoff_success_is_worker_completion"] is False
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}
    assert "worker_done_observed=False" in result.message
