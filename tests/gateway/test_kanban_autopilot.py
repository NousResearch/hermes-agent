"""Regression tests for the Kanban-first /autopilot surface."""

from __future__ import annotations

import json
from typing import Any


DONE_CRITERIA_BODY = """
Done criteria:
- Focused tests pass.
- Git diff check passes.
- Boundaries remain confirmed.
"""


def _done_criteria_hash(body: str = DONE_CRITERIA_BODY) -> str:
    from gateway.kanban_autopilot import build_done_criteria_ledger

    result = build_done_criteria_ledger(body)
    assert result["ok"] is True
    return result["criteria_hash"]


def _worker_done_proofs(body: str = DONE_CRITERIA_BODY) -> list[dict[str, Any]]:
    from gateway.kanban_autopilot import build_done_criteria_ledger

    ledger = build_done_criteria_ledger(body)
    assert ledger["ok"] is True
    return [
        {
            "criterion_id": criterion["id"],
            "proof": f"{criterion['text']} verified",
            "artifact_refs": [f"artifacts/{criterion['id']}.md"],
            **(
                {"tests_run": 1, "tests_passed": True, "test_command": "python -m pytest tests/gateway/test_kanban_autopilot.py -q"}
                if "test" in criterion["text"].lower()
                else {}
            ),
            **(
                {"checks_passed": True, "check_command": "git diff --check"}
                if "check" in criterion["text"].lower() or "diff" in criterion["text"].lower()
                else {}
            ),
        }
        for criterion in ledger["done_criteria_ledger"]["criteria"]
    ]


def _worker_done_evidence(body: str = DONE_CRITERIA_BODY) -> dict[str, object]:
    from gateway.kanban_autopilot import build_done_criteria_ledger

    ledger = build_done_criteria_ledger(body)
    assert ledger["ok"] is True
    return {
        "worker_done": True,
        "kanban_worker_done": True,
        "boundaries_confirmed": True,
        "authority_boundary_confirmed": True,
        "task_body": body,
        "criteria_hash": ledger["criteria_hash"],
        "criteria_proofs": _worker_done_proofs(body),
        "cleanup": {"proof": "git status --short clean", "worktree_clean": True},
        "workspace_kind": "worktree",
    }


def _review_ready_evidence(body: str = DONE_CRITERIA_BODY) -> dict[str, object]:
    return {
        "work_id": "BO-081",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "yuuka/bo-081-autopilot-review-ready-contract",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/123",
        "pr_base": "main",
        "pr_head": "yuuka/bo-081-autopilot-review-ready-contract",
        "checks_passed": True,
        "worktree_clean": True,
        "verifier_verdict": {"verdict": "PASS", "criterion_results": _verifier_results(body)},
        **_worker_done_evidence(body),
    }


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
    assert result.decision["effective_mode"] == "default_policy_loop"
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


def test_done_criteria_ledger_extracts_stable_ids_and_deterministic_hash():
    from gateway.kanban_autopilot import build_done_criteria_ledger

    body = """
Goal: ship a bounded validator.
Done criteria:
- Focused tests pass.
- Git diff check passes.
- Boundaries remain confirmed.
Verification requirements: pytest and diff check.
"""

    first = build_done_criteria_ledger(body)
    second = build_done_criteria_ledger(body)

    assert first["ok"] is True
    assert first["criteria_hash"] == second["criteria_hash"]
    assert first["criteria_version"] == 1
    assert first["criteria_ids"] == [
        "dc-01-focused-tests-pass",
        "dc-02-git-diff-check-passes",
        "dc-03-boundaries-remain-confirmed",
    ]
    assert first["done_criteria_ledger"]["schema"] == "autopilot_done_criteria_ledger.v1"


def test_done_criteria_ledger_rejects_missing_done_criteria():
    from gateway.kanban_autopilot import build_done_criteria_ledger

    result = build_done_criteria_ledger("Goal: do a thing. Verification requirements: pytest.")

    assert result["ok"] is False
    assert "missing_done_criteria_ledger" in result["reason_codes"]


def test_done_criteria_ledger_rejects_ambiguous_done_criteria():
    from gateway.kanban_autopilot import build_done_criteria_ledger

    result = build_done_criteria_ledger("""
Done criteria:
- Tests pass or some other verification is acceptable.
""")

    assert result["ok"] is False
    assert result["reason_codes"] == ["ambiguous_done_criteria_ledger"]


def test_worker_done_evidence_rejects_missing_criterion_proofs():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    result = validate_worker_done_evidence({**_worker_done_evidence(), "criteria_proofs": []}, task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is False
    assert "missing_criterion_level_evidence" in result["reason_codes"]


def test_worker_done_evidence_rejects_missing_or_wrong_criteria_hash():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    missing = validate_worker_done_evidence({k: v for k, v in _worker_done_evidence().items() if k != "criteria_hash"}, task_body=DONE_CRITERIA_BODY)
    wrong = validate_worker_done_evidence({**_worker_done_evidence(), "criteria_hash": "0" * 64}, task_body=DONE_CRITERIA_BODY)

    assert missing["worker_done_evidence_valid"] is False
    assert "missing_criteria_hash" in missing["reason_codes"]
    assert wrong["worker_done_evidence_valid"] is False
    assert "stale_criteria_hash" in wrong["reason_codes"]


def test_worker_done_evidence_rejects_missing_tests_checks_for_deterministic_criteria():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    evidence = _worker_done_evidence()
    evidence["criteria_proofs"] = [
        {**evidence["criteria_proofs"][0], "tests_run": None, "tests_passed": None, "test_command": ""},
        *evidence["criteria_proofs"][1:],
    ]

    result = validate_worker_done_evidence(evidence, task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is False
    assert any(code.startswith("missing_tests_or_checks_for_") for code in result["reason_codes"])


def test_worker_done_evidence_rejects_missing_authority_boundary_confirmation():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    result = validate_worker_done_evidence({**_worker_done_evidence(), "boundaries_confirmed": False, "authority_boundary_confirmed": False}, task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is False
    assert "missing_authority_boundary_confirmation" in result["reason_codes"]


def test_worker_done_evidence_accepts_complete_worker_proof_contract():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    result = validate_worker_done_evidence(_worker_done_evidence(), task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is True
    assert result["status"] == "accepted"
    assert result["reason_codes"] == []
    assert result["criteria_hash"] == _done_criteria_hash()
    assert result["cleanup_or_residue_proof"] is True


def _verifier_results(body: str = DONE_CRITERIA_BODY, status: str = "PASS") -> list[dict[str, Any]]:
    from gateway.kanban_autopilot import build_done_criteria_ledger

    ledger = build_done_criteria_ledger(body)
    assert ledger["ok"] is True
    return [
        {
            "criterion_id": criterion["id"],
            "status": status,
            "evidence": f"verifier independently checked {criterion['id']}",
        }
        for criterion in ledger["done_criteria_ledger"]["criteria"]
    ]


def test_verifier_verdict_pass_requires_distinct_identity_and_all_criteria_evidence():
    from gateway.kanban_autopilot import evaluate_verifier_verdict

    result = evaluate_verifier_verdict(
        {**_worker_done_evidence(), "worker_identity": "arisu"},
        {"verifier_identity": "yuuka", "criterion_results": _verifier_results()},
    )

    assert result["verdict"] == "PASS"
    assert result["review_ready_input_eligible"] is True
    assert set(item["criterion_id"] for item in result["criterion_results"]) == set(result["criteria_ids"])
    assert result["side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}


def test_verifier_verdict_rejects_worker_self_approval():
    from gateway.kanban_autopilot import evaluate_verifier_verdict

    result = evaluate_verifier_verdict(
        {**_worker_done_evidence(), "worker_identity": "arisu"},
        {"verifier_identity": "arisu", "criterion_results": _verifier_results()},
    )

    assert result["verdict"] == "FAIL"
    assert result["review_ready_input_eligible"] is False
    assert "verifier_same_as_worker" in result["reason_codes"]
    assert any("distinct" in item for item in result["remediation_instructions"])


def test_verifier_verdict_fail_includes_actionable_remediation_and_keeps_worker_done():
    from gateway.kanban_autopilot import evaluate_verifier_verdict

    criterion_results = _verifier_results()
    criterion_results[0] = {
        **criterion_results[0],
        "status": "FAIL",
        "remediation": "Add focused pytest evidence for the first criterion.",
    }
    result = evaluate_verifier_verdict(
        {**_worker_done_evidence(), "worker_identity": "arisu"},
        {"verifier_identity": "yuuka", "criterion_results": criterion_results},
    )

    assert result["verdict"] == "FAIL"
    assert result["worker_done_retained"] is True
    assert result["worker_done_evidence"]["worker_done_evidence_valid"] is True
    assert result["review_ready_input_eligible"] is False
    assert any(code.startswith("verifier_failed_") for code in result["reason_codes"])
    assert "Add focused pytest evidence for the first criterion." in result["remediation_instructions"]


def test_verifier_verdict_blocked_reports_blocker_reason_codes():
    from gateway.kanban_autopilot import evaluate_verifier_verdict

    result = evaluate_verifier_verdict(
        {**_worker_done_evidence(), "worker_identity": "arisu"},
        {"verifier_identity": "yuuka", "blocker_reason_codes": ["missing_live_pr_context"]},
    )

    assert result["verdict"] == "BLOCKED"
    assert result["review_ready_input_eligible"] is False
    assert result["blocker_reason_codes"] == ["missing_live_pr_context"]


def test_verifier_verdict_requires_refinement_for_missing_ambiguous_or_stale_ledger():
    from gateway.kanban_autopilot import evaluate_verifier_verdict

    missing = evaluate_verifier_verdict(
        {**_worker_done_evidence(), "worker_identity": "arisu", "task_body": "Goal: too vague", "criteria_hash": ""},
        {"verifier_identity": "yuuka", "criterion_results": []},
    )
    ambiguous_body = """
Done criteria:
- Maybe pass tests or etc.
"""
    ambiguous = evaluate_verifier_verdict(
        {**_worker_done_evidence(), "worker_identity": "arisu", "task_body": ambiguous_body, "criteria_hash": ""},
        {"verifier_identity": "yuuka", "criterion_results": []},
    )
    stale = evaluate_verifier_verdict(
        {**_worker_done_evidence(), "worker_identity": "arisu", "criteria_hash": "0" * 64},
        {"verifier_identity": "yuuka", "criterion_results": _verifier_results()},
    )

    assert missing["verdict"] == "REFINEMENT_REQUIRED"
    assert "missing_done_criteria_ledger" in missing["reason_codes"]
    assert ambiguous["verdict"] == "REFINEMENT_REQUIRED"
    assert "ambiguous_done_criteria_ledger" in ambiguous["reason_codes"]
    assert stale["verdict"] == "REFINEMENT_REQUIRED"
    assert "stale_criteria_hash" in stale["reason_codes"]
    assert stale["review_ready_input_eligible"] is False


def test_retry_controller_queues_exactly_one_remediation_on_verifier_fail():
    from gateway.kanban_autopilot import plan_verifier_retry_controller

    criterion_results = _verifier_results()
    criterion_results[0] = {**criterion_results[0], "status": "FAIL", "remediation": "Fix first criterion."}
    result = plan_verifier_retry_controller(
        {**_worker_done_evidence(), "worker_identity": "arisu"},
        {"verifier_identity": "yuuka", "criterion_results": criterion_results},
        attempt=1,
    )

    assert result["next_state"] == "queue_remediation"
    assert result["queued_remediation_count"] == 1
    assert result["queued_actions"][0]["type"] == "remediation"
    assert result["queued_actions"][0]["attempt"] == 2
    assert result["kanban_evidence_patch"]["verification_attempt"] == 1
    assert result["side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}


def test_retry_controller_persists_attempt_and_blocks_after_three_failures():
    from gateway.kanban_autopilot import plan_verifier_retry_controller

    criterion_results = _verifier_results()
    criterion_results[0] = {**criterion_results[0], "status": "FAIL"}
    result = plan_verifier_retry_controller(
        {**_worker_done_evidence(), "worker_identity": "arisu", "verification_attempt": 3},
        {"verifier_identity": "yuuka", "criterion_results": criterion_results},
    )

    assert result["next_state"] == "blocked"
    assert result["blocked"] is True
    assert result["queued_remediation_count"] == 0
    assert result["attempt"] == 3
    assert "max_verification_attempts_exhausted" in result["reason_codes"]
    assert result["kanban_evidence_patch"]["next_controller_state"] == "blocked"


def test_retry_controller_pass_allows_review_ready_input_without_dispatch():
    from gateway.kanban_autopilot import plan_verifier_retry_controller

    result = plan_verifier_retry_controller(
        {**_worker_done_evidence(), "worker_identity": "arisu"},
        {"verifier_identity": "yuuka", "criterion_results": _verifier_results()},
    )

    assert result["next_state"] == "verifier_pass"
    assert result["review_ready_input_eligible"] is True
    assert result["queued_actions"] == []
    assert result["side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}


def test_retry_controller_refinement_required_does_not_queue_worker_retry():
    from gateway.kanban_autopilot import plan_verifier_retry_controller

    result = plan_verifier_retry_controller(
        {**_worker_done_evidence(), "worker_identity": "arisu", "task_body": "Goal: too vague", "criteria_hash": ""},
        {"verifier_identity": "yuuka", "criterion_results": []},
    )

    assert result["next_state"] == "refinement_required"
    assert result["queued_remediation_count"] == 0
    assert result["review_ready_input_eligible"] is False
    assert "missing_done_criteria_ledger" in result["reason_codes"]


def test_retry_controller_blocked_verdict_does_not_queue_remediation():
    from gateway.kanban_autopilot import plan_verifier_retry_controller

    result = plan_verifier_retry_controller(
        {**_worker_done_evidence(), "worker_identity": "arisu"},
        {"verifier_identity": "yuuka", "blocker_reason_codes": ["missing_authority"]},
    )

    assert result["next_state"] == "blocked"
    assert result["blocked"] is True
    assert result["queued_actions"] == []
    assert "missing_authority" in result["reason_codes"]


def test_ready_gate_rejects_missing_or_ambiguous_done_criteria_and_accepts_explicit_criteria():
    from gateway.kanban_autopilot import evaluate_autopilot_ready_gate

    missing = _ready_candidate("BO-146")
    missing["body"] = missing["body"].split("Done criteria:", 1)[0]
    missing_result = evaluate_autopilot_ready_gate(missing)
    assert missing_result["autopilot_ready"] is False
    assert "missing_done_criteria_ledger" in missing_result["reason_codes"]

    ambiguous = _ready_candidate("BO-147")
    ambiguous["body"] = ambiguous["body"].replace(
        "- Focused tests pass.",
        "- Focused tests pass or some other verification is fine.",
    )
    ambiguous_result = evaluate_autopilot_ready_gate(ambiguous)
    assert ambiguous_result["autopilot_ready"] is False
    assert "ambiguous_done_criteria_ledger" in ambiguous_result["reason_codes"]

    explicit = evaluate_autopilot_ready_gate(_ready_candidate("BO-148"))
    assert explicit["autopilot_ready"] is True
    assert explicit["criteria_hash"]
    assert explicit["criteria_ids"]


def test_review_ready_contract_rejects_stale_criteria_hash_and_requires_worktree_cleanup_proof():
    from gateway.kanban_autopilot import evaluate_review_ready_contract

    evidence = {
        **_worker_done_evidence(),
        "work_id": "BO-146",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "work/BO-146-done-criteria-ledger",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/146",
        "pr_base": "main",
        "pr_head": "work/BO-146-done-criteria-ledger",
        "checks_passed": True,
        "worktree_clean": True,
        "task_body": DONE_CRITERIA_BODY,
        "criteria_hash": "0" * 64,
        "workspace_kind": "worktree",
        "cleanup": {},
        "verifier_verdict": {"verdict": "PASS", "criterion_results": _verifier_results()},
    }

    stale = evaluate_review_ready_contract(evidence)
    assert stale["review_ready"] is False
    assert "stale_criteria_hash" in stale["reason_codes"]
    assert "missing_cleanup_proof" in stale["reason_codes"]

    current_hash = _done_criteria_hash()
    missing_cleanup = evaluate_review_ready_contract({**evidence, "criteria_hash": current_hash})
    assert missing_cleanup["review_ready"] is False
    assert "stale_criteria_hash" not in missing_cleanup["reason_codes"]
    assert "missing_cleanup_proof" in missing_cleanup["reason_codes"]

    retained = evaluate_review_ready_contract({
        **evidence,
        "criteria_hash": current_hash,
        "cleanup": {},
        "residue": {
            "items": [
                {"path": ".worktrees/bo-146-done-criteria-ledger", "disposition": "retained", "reason": "review evidence", "ttl": "2026-06-01"}
            ]
        },
    })
    assert retained["review_ready"] is True
    assert retained["reason_codes"] == []


def test_review_ready_contract_requires_verifier_pass():
    from gateway.kanban_autopilot import evaluate_review_ready_contract

    missing = evaluate_review_ready_contract({k: v for k, v in _review_ready_evidence().items() if k != "verifier_verdict"})
    failed = evaluate_review_ready_contract({**_review_ready_evidence(), "verifier_verdict": {"verdict": "FAIL", "reason_codes": ["criterion_failed"]}})
    passed = evaluate_review_ready_contract(_review_ready_evidence())

    assert missing["review_ready"] is False
    assert "missing_verifier_pass" in missing["reason_codes"]
    assert failed["review_ready"] is False
    assert "missing_verifier_pass" in failed["reason_codes"]
    assert passed["review_ready"] is True
    assert "missing_verifier_pass" not in passed["reason_codes"]


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
Done criteria:
- Focused tests pass.
- Git diff check passes.
- Boundaries remain confirmed.
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
        **_worker_done_evidence(),
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
Done criteria:
- Focused tests pass.
- Git diff check passes.
- Boundaries remain confirmed.
""",
        "routing_verdict": {"verdict": "Hermes direct"},
        "admission_snapshot": {"repo_full_name": "chriskim12/hermes-agent"},
    }


def test_dispatcher_eligibility_bridge_requires_controller_on_and_ready_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import evaluate_dispatcher_eligibility, handle_autopilot_command

    stopped = evaluate_dispatcher_eligibility([_ready_candidate()])
    assert stopped["eligible"] == []
    assert stopped["ineligible"][0]["reason_codes"] == ["autopilot_effective_mode_not_dispatch_enabled"]

    handle_autopilot_command("on", actor="tester")
    verdict = evaluate_dispatcher_eligibility([_ready_candidate(), {"id": "t_raw", "public_id": "BO-999", "status": "ready", "body": "raw"}])

    assert [item["public_id"] for item in verdict["eligible"]] == ["BO-100"]
    assert verdict["ineligible"][0]["public_id"] == "BO-999"
    assert "missing_goal" in verdict["ineligible"][0]["reason_codes"]
    assert verdict["handoff_target"] == "existing_kanban_dispatcher"
    assert verdict["second_dispatcher_created"] is False
    assert verdict["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0}


def test_dispatcher_eligibility_rejects_legacy_enabled_blocked_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway_autopilot_state.json").write_text(
        json.dumps({"version": 1, "enabled": True, "desired_mode": "enabled", "updated_by": "legacy"}),
        encoding="utf-8",
    )

    from gateway.kanban_autopilot import evaluate_dispatcher_eligibility

    verdict = evaluate_dispatcher_eligibility([_ready_candidate("BO-LEGACY")])

    assert verdict["controller_effective_mode"] == "blocked"
    assert verdict["eligible"] == []
    assert verdict["ineligible"][0]["public_id"] == "BO-LEGACY"
    assert verdict["ineligible"][0]["reason_codes"] == ["autopilot_effective_mode_not_dispatch_enabled"]


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
        **_worker_done_evidence(),
        "work_id": "BO-081",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "yuuka/bo-081-autopilot-review-ready-contract",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/123",
        "pr_base": "main",
        "pr_head": "yuuka/bo-081-autopilot-review-ready-contract",
        "checks_passed": True,
        "worktree_clean": True,
        "task_body": DONE_CRITERIA_BODY,
        "verifier_verdict": {"verdict": "PASS", "criterion_results": _verifier_results()},
    })
    assert satisfied["review_ready"] is True
    assert satisfied["reason_codes"] == []
    assert satisfied["merge_allowed"] is False
    assert satisfied["release_allowed"] is False


def test_review_ready_contract_rejects_wrong_repo_and_release_base():
    from gateway.kanban_autopilot import evaluate_review_ready_contract

    result = evaluate_review_ready_contract({
        **_worker_done_evidence(),
        "work_id": "BO-081",
        "repo_full_name": "evil/repo",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "feature",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/123",
        "pr_base": "prod",
        "pr_head": "other",
        "checks_passed": True,
        "worktree_clean": True,
    }, expected_repo_full_name="chriskim12/hermes-agent")

    assert result["review_ready"] is False
    assert "repo_full_name_mismatch" in result["reason_codes"]
    assert "release_base_requires_separate_approval" in result["reason_codes"]
    assert "pr_head_task_branch_mismatch" in result["reason_codes"]
    assert result["merge_allowed"] is False


def test_worker_done_evidence_requires_criterion_level_proofs():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    evidence = _worker_done_evidence()
    evidence.pop("criteria_proofs")

    result = validate_worker_done_evidence(evidence, task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is False
    assert "missing_criterion_level_evidence" in result["reason_codes"]


def test_worker_done_evidence_rejects_stale_criteria_hash():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    evidence = _worker_done_evidence()
    evidence["criteria_hash"] = "0" * 64

    result = validate_worker_done_evidence(evidence, task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is False
    assert "stale_criteria_hash" in result["reason_codes"]


def test_worker_done_evidence_rejects_missing_boundary_confirmation():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    evidence = _worker_done_evidence()
    evidence.pop("authority_boundary_confirmed")
    evidence.pop("boundaries_confirmed")

    result = validate_worker_done_evidence(evidence, task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is False
    assert "missing_authority_boundary_confirmation" in result["reason_codes"]


def test_worker_done_evidence_rejects_missing_deterministic_verification_for_test_criteria():
    from gateway.kanban_autopilot import validate_worker_done_evidence

    evidence = _worker_done_evidence()
    for proof in evidence["criteria_proofs"]:
        if proof["criterion_id"] == "dc-01-focused-tests-pass":
            proof.pop("tests_run", None)
            proof.pop("tests_passed", None)
            proof.pop("test_command", None)
            break

    result = validate_worker_done_evidence(evidence, task_body=DONE_CRITERIA_BODY)

    assert result["worker_done_evidence_valid"] is False
    assert "missing_tests_or_checks_for_dc-01-focused-tests-pass" in result["reason_codes"]


def test_default_live_candidate_load_includes_active_flights_for_global_single_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))

    from gateway.kanban_autopilot import activate_single_flight, load_live_kanban_candidates
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        active_id = kb.create_task(
            conn,
            title="active autopilot worker",
            body="already running",
            assignee="worker",
            public_id="BO-ACTIVE",
        )
        kb.create_task(
            conn,
            title="ready autopilot worker",
            body=_ready_candidate("BO-READY")["body"],
            assignee="worker",
            public_id="BO-READY",
            routing_verdict={"verdict": "Hermes direct", "reason": "test"},
            closeout_evidence=_worker_done_evidence(),
        )
        conn.execute(
            "UPDATE tasks SET status = 'running', current_run_id = 'run-1', claim_lock = 'lock-1', worker_pid = 4242 WHERE id = ?",
            (active_id,),
        )

    candidates = load_live_kanban_candidates()

    assert {candidate["public_id"] for candidate in candidates} >= {"BO-ACTIVE", "BO-READY"}
    decision = activate_single_flight(candidates)
    assert decision["status"] == "active_flight_blocked"
    assert decision["active_flights"][0]["public_id"] == "BO-ACTIVE"


def test_default_live_candidate_load_does_not_limit_out_active_flights(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))

    from gateway.kanban_autopilot import load_live_kanban_candidates
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        active_id = kb.create_task(
            conn,
            title="low priority active worker",
            body="already running",
            assignee="worker",
            public_id="BO-ACTIVE-LIMIT",
            priority=-100,
        )
        conn.execute(
            "UPDATE tasks SET status = 'running', current_run_id = 'run-limit', claim_lock = 'lock-limit' WHERE id = ?",
            (active_id,),
        )
        for index in range(60):
            kb.create_task(
                conn,
                title=f"high priority ready {index}",
                body=_ready_candidate(f"BO-READY-{index}")["body"],
                assignee="worker",
                public_id=f"BO-READY-{index}",
                priority=100,
                routing_verdict={"verdict": "Hermes direct", "reason": "test"},
                closeout_evidence=_worker_done_evidence(),
            )

    candidates = load_live_kanban_candidates(limit=50)

    assert "BO-ACTIVE-LIMIT" in {candidate["public_id"] for candidate in candidates}


def test_parent_focused_dry_run_loads_only_hierarchy_children_without_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))

    from gateway import kanban_autopilot
    from gateway.kanban_autopilot import handle_autopilot_command, load_live_kanban_candidates
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        parent_id = kb.create_task(
            conn,
            title="parent envelope",
            body="Parent envelope task.",
            assignee="manager",
            public_id="BO-PARENT",
        )
        ready_child_id = kb.create_task(
            conn,
            title="eligible child",
            body=_ready_candidate("BO-CHILD-READY")["body"],
            assignee="worker",
            public_id="BO-CHILD-READY",
            routing_verdict={"verdict": "Hermes direct", "reason": "test"},
            closeout_evidence=_worker_done_evidence(),
        )
        raw_child_id = kb.create_task(
            conn,
            title="raw child",
            body="Please handle this later.",
            assignee="worker",
            public_id="BO-CHILD-RAW",
        )
        kb.link_tasks(conn, parent_id, ready_child_id, relation_type="hierarchy")
        kb.link_tasks(conn, parent_id, raw_child_id, relation_type="hierarchy")
        other_parent_id = kb.create_task(
            conn,
            title="other parent",
            body="Other parent.",
            assignee="manager",
            public_id="BO-OTHER-PARENT",
        )
        other_child_id = kb.create_task(
            conn,
            title="other child",
            body=_ready_candidate("BO-OTHER-READY")["body"],
            assignee="worker",
            public_id="BO-OTHER-READY",
            routing_verdict={"verdict": "Hermes direct", "reason": "test"},
            closeout_evidence=_worker_done_evidence(),
        )
        kb.link_tasks(conn, other_parent_id, other_child_id, relation_type="hierarchy")

    candidates = load_live_kanban_candidates(parent_public_id="BO-PARENT")
    child_ids = {candidate["public_id"] for candidate in candidates}

    assert child_ids == {"BO-CHILD-READY", "BO-CHILD-RAW"}
    assert all(candidate["parent_public_id"] == "BO-PARENT" for candidate in candidates)

    handle_autopilot_command("on", actor="tester")

    calls: list[str] = []

    def forbidden_dispatch(*_args, **_kwargs):
        calls.append("dispatch")
        raise AssertionError("dry-run must not dispatch")

    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", forbidden_dispatch, raising=False)
    result = handle_autopilot_command("dry-run BO-PARENT", actor="tester")

    assert result.ok is True
    assert calls == []
    assert result.decision["status"] == "DRY_RUN"
    assert result.decision["closed_loop"]["would_select"][0]["public_id"] == "BO-CHILD-READY"
    would_skip_ids = {item["public_id"] for item in result.decision["closed_loop"]["would_skip"]}
    assert would_skip_ids == {"BO-CHILD-RAW"}
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


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
        **_worker_done_evidence(),
        "work_id": "BO-094",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "8f0cbc56db6498e16c628e42430dbd6156d99fb3",
        "task_branch": "bo-094",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/123",
        "pr_base": "main",
        "pr_head": "bo-094",
        "checks_passed": True,
        "worktree_clean": True,
        "task_body": DONE_CRITERIA_BODY,
        "verifier_verdict": {"verdict": "PASS", "criterion_results": _verifier_results()},
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
        "task_body": DONE_CRITERIA_BODY,
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
        **_worker_done_evidence(),
        "work_id": "BO-118",
        "repo_full_name": "chriskim12/hermes-agent",
        "commit": "91d572c971956e705cbfccc19990bf1ed128c239",
        "task_branch": "bo-118-autopilot-review-package",
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/118",
        "pr_base": "main",
        "pr_head": "bo-118-autopilot-review-package",
        "checks_passed": True,
        "worktree_clean": True,
        "task_body": DONE_CRITERIA_BODY,
        "verifier_verdict": {"verdict": "PASS", "criterion_results": _verifier_results()},
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


def test_autopilot_once_loads_live_kanban_candidates_for_selected_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    calls: list[dict] = []

    def fake_loader(*, parent_public_id=None, tenant=None, limit=50):
        calls.append({"parent_public_id": parent_public_id, "tenant": tenant, "limit": limit})
        return [_ready_candidate("BO-115")]

    monkeypatch.setattr(kanban_autopilot, "load_live_kanban_candidates", fake_loader, raising=False)
    monkeypatch.setattr(
        kanban_autopilot,
        "dispatch_selected_once",
        lambda candidate, **_kwargs: {"dispatched": True, "spawned": [{"task_id": candidate["task_id"], "assignee": "default", "workspace": "/tmp/ws"}]},
        raising=False,
    )
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    result = kanban_autopilot.handle_autopilot_command("once BO-114", actor="tester")

    assert result.ok is True
    assert calls == [{"parent_public_id": "BO-114", "tenant": None, "limit": 50}]
    assert result.decision["status"] == "DISPATCHED"
    assert result.decision["single_flight"]["selected"]["public_id"] == "BO-115"
    assert result.decision["single_flight"]["handoff"]["check_only"] is False
    assert result.decision["single_flight"]["handoff"]["would_dispatch"] is True


def test_autopilot_once_command_uses_single_flight_dispatch_not_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    monkeypatch.setattr(
        kanban_autopilot,
        "dispatch_selected_once",
        lambda candidate, **_kwargs: {"dispatched": True, "spawned": [{"task_id": candidate["task_id"], "assignee": "default", "workspace": "/tmp/ws"}]},
        raising=False,
    )
    kanban_autopilot.handle_autopilot_command("on", actor="tester")
    result = kanban_autopilot.handle_autopilot_command("once", actor="tester", candidates=[_ready_candidate("BO-090"), _ready_candidate("BO-091")])

    assert result.ok is True
    assert result.decision["status"] == "DISPATCHED"
    assert result.decision["single_flight"]["selected"]["public_id"] == "BO-090"
    assert result.decision["single_flight"]["handoff_success_is_worker_completion"] is False
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 1, "mutated": 0, "dispatched": 1}
    assert "worker_done_observed=False" in result.message


def test_autopilot_once_dispatches_selected_candidate_through_existing_dispatcher(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    dispatched: list[dict] = []

    def fake_dispatch(candidate: dict, **_kwargs) -> dict:
        dispatched.append(candidate)
        return {
            "dispatched": True,
            "spawned": [{"task_id": candidate["task_id"], "assignee": "default", "workspace": "/tmp/ws"}],
            "reclaimed": 0,
            "crashed": [],
            "timed_out": [],
            "auto_blocked": [],
            "promoted": 0,
            "skipped_unassigned": [],
            "skipped_nonspawnable": [],
        }

    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", fake_dispatch, raising=False)
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    result = kanban_autopilot.handle_autopilot_command(
        "once",
        actor="tester",
        candidates=[_ready_candidate("BO-122"), _ready_candidate("BO-123")],
    )

    assert result.ok is True
    assert result.decision["status"] == "DISPATCHED"
    assert result.decision["single_flight"]["selected"]["public_id"] == "BO-122"
    assert result.decision["single_flight"]["handoff"]["check_only"] is False
    assert result.decision["single_flight"]["handoff"]["would_dispatch"] is True
    assert result.decision["dispatch_result"]["spawned"][0]["task_id"] == "t_good"
    assert [item["public_id"] for item in dispatched] == ["BO-122"]


def test_autopilot_once_blocks_without_dispatch_when_selected_dispatcher_spawns_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    monkeypatch.setattr(
        kanban_autopilot,
        "dispatch_selected_once",
        lambda candidate, **_kwargs: {"dispatched": False, "spawned": [], "skipped_unassigned": [candidate["task_id"]]},
        raising=False,
    )
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    result = kanban_autopilot.handle_autopilot_command("once", actor="tester", candidates=[_ready_candidate("BO-122")])

    assert result.ok is True
    assert result.decision["status"] == "DISPATCH_BLOCKED"
    assert result.decision["single_flight"]["selected"]["public_id"] == "BO-122"
    assert result.decision["dispatch_result"]["spawned"] == []
    assert result.decision["single_flight"]["worker_done_observed"] is False


def test_autopilot_on_with_parent_scope_dispatches_one_bounded_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    dispatched: list[str] = []

    def fake_dispatch(candidate: dict, **_kwargs) -> dict:
        dispatched.append(candidate["public_id"])
        return {"dispatched": True, "spawned": [{"task_id": candidate["task_id"], "assignee": "default", "workspace": "/tmp/ws"}]}

    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", fake_dispatch, raising=False)

    result = kanban_autopilot.handle_autopilot_command(
        "on BO-114",
        actor="tester",
        candidates=[_ready_candidate("BO-122"), _ready_candidate("BO-123")],
    )

    assert result.ok is True
    assert result.decision["status"] == "BOUNDED_DISPATCHED"
    assert result.decision["scope"] == {"mode": "parent", "parent_public_id": "BO-114", "tenant": None}
    assert result.decision["dispatched_count"] == 1
    assert result.decision["dry_run_side_effects"] == {"claimed": 0, "spawned": 1, "mutated": 0, "dispatched": 1}
    assert dispatched == ["BO-122"]


def test_autopilot_on_without_explicit_or_focused_parent_scope_stays_control_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unscoped /autopilot on must not dispatch")

    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", forbidden, raising=False)
    result = kanban_autopilot.handle_autopilot_command("on", actor="tester", candidates=[_ready_candidate("BO-122")])

    assert result.ok is True
    assert result.decision["desired_mode"] == "on"
    assert result.decision["dispatch_once_called"] is False


def test_autopilot_on_parent_persists_focus_for_continuous_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    result = handle_autopilot_command("on BO-114", actor="tester", candidates=[])
    status = handle_autopilot_command("status", actor="tester")
    state = json.loads((tmp_path / "gateway_autopilot_state.json").read_text(encoding="utf-8"))

    assert result.ok is True
    assert state["desired_mode"] == "on"
    assert state["enabled"] is True
    assert state["focus"] == "BO-114"
    assert status.decision["focus"] == "BO-114"


def test_autopilot_continuous_tick_uses_persisted_parent_focus_and_dispatches_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    loaded: list[dict] = []
    dispatched: list[str] = []

    def fake_loader(*, parent_public_id=None, tenant=None, limit=50):
        loaded.append({"parent_public_id": parent_public_id, "tenant": tenant, "limit": limit})
        return [_ready_candidate("BO-201"), _ready_candidate("BO-202")]

    def fake_dispatch(candidate: dict, **_kwargs) -> dict:
        dispatched.append(candidate["public_id"])
        return {"dispatched": True, "spawned": [{"task_id": candidate["task_id"], "assignee": "default", "workspace": "/tmp/ws"}]}

    monkeypatch.setattr(kanban_autopilot, "load_live_kanban_candidates", fake_loader, raising=False)
    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", fake_dispatch, raising=False)

    kanban_autopilot.handle_autopilot_command("on BO-114", actor="tester", candidates=[])
    result = kanban_autopilot.autopilot_continuous_tick(actor="gateway-loop")

    assert result["status"] == "BOUNDED_DISPATCHED"
    assert result["scope"] == {"mode": "parent", "parent_public_id": "BO-114", "tenant": None}
    assert result["caps"] == {"max_active_flights": 1, "max_dispatches_per_tick": 1}
    assert result["dispatched_count"] == 1
    assert loaded == [{"parent_public_id": "BO-114", "tenant": None, "limit": 50}]
    assert dispatched == ["BO-201"]


def test_autopilot_continuous_tick_without_parent_scope_uses_default_policy_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default-policy loop must not dispatch without an eligible ready-gate candidate")

    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", forbidden, raising=False)
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    result = kanban_autopilot.autopilot_continuous_tick(actor="gateway-loop")

    assert result["status"] == "BOUNDED_BLOCKED"
    assert result["effective_mode"] == "default_policy_loop"
    assert result["scope"] == {"mode": "default_policy", "parent_public_id": None, "tenant": None}
    assert result["dispatched_count"] == 0


def test_autopilot_on_without_parent_enters_default_policy_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    result = handle_autopilot_command("on", actor="tester")
    status = handle_autopilot_command("status", actor="tester")

    assert result.ok is True
    assert result.decision["desired_mode"] == "on"
    assert result.decision["effective_mode"] == "default_policy_loop"
    assert result.decision["focus"] is None
    assert result.decision["scope_mode"] == "default_policy"
    assert result.decision["dispatch_blocked"] is False
    assert status.decision["effective_mode"] == "default_policy_loop"
    assert "default/global policy" in status.message


def test_autopilot_unscoped_continuous_tick_dispatches_one_default_policy_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    dispatched = []

    def fake_dispatch(candidate):
        dispatched.append(candidate["public_id"])
        return {"dispatched": True, "spawned": [{"pid": 123}], "task_ids": [candidate["task_id"]]}

    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", fake_dispatch)
    kanban_autopilot.handle_autopilot_command("on", actor="tester")

    result = kanban_autopilot.autopilot_continuous_tick(
        actor="loop",
        candidates=[
            {"id": "t_raw", "public_id": "BO-999", "status": "ready", "body": "raw", "routing_verdict": {"verdict": "Hermes direct"}},
            _ready_candidate("BO-133"),
        ],
    )

    assert result["effective_mode"] == "default_policy_loop"
    assert result["scope"] == {"mode": "default_policy", "parent_public_id": None, "tenant": None}
    assert result["status"] == "BOUNDED_DISPATCHED"
    assert result["dispatched_count"] == 1
    assert dispatched == ["BO-133"]
    assert result["single_flight"]["selected"]["public_id"] == "BO-133"
    assert [item["public_id"] for item in result["single_flight"].get("skipped") or []] == ["BO-999"]


def test_autopilot_parent_scope_still_limits_continuous_tick(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway import kanban_autopilot

    captured = []

    def fake_dispatch(candidate):
        captured.append(candidate["public_id"])
        return {"dispatched": True, "spawned": [], "task_ids": [candidate["task_id"]]}

    monkeypatch.setattr(kanban_autopilot, "dispatch_selected_once", fake_dispatch)
    kanban_autopilot.handle_autopilot_command("on BO-114", actor="tester", candidates=[])

    result = kanban_autopilot.autopilot_continuous_tick(
        actor="loop",
        candidates=[
            {**_ready_candidate("BO-133"), "parent_public_id": "BO-090"},
            {**_ready_candidate("BO-134"), "parent_public_id": "BO-114"},
        ],
    )

    assert result["effective_mode"] == "parent_scoped"
    assert result["scope"]["parent_public_id"] == "BO-114"
    assert captured == ["BO-134"]
    assert result["single_flight"]["selected"]["public_id"] == "BO-134"


def test_autopilot_dispatch_injects_review_ready_pr_worker_env(monkeypatch):
    from gateway import kanban_autopilot

    captured = {}

    class DummyConn:
        def close(self):
            captured["closed"] = True

    class DummyDispatchResult:
        reclaimed = []
        crashed = []
        timed_out = []
        auto_blocked = []
        promoted = []
        spawned = [("t_ws012", "default", "/tmp/ws012")]
        skipped_unassigned = []
        skipped_nonspawnable = []

    def fake_connect(*, board=None):
        captured["board"] = board
        return DummyConn()

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return DummyDispatchResult()

    import hermes_cli.kanban_db as kb

    monkeypatch.setattr(kb, "connect", fake_connect)
    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)

    result = kanban_autopilot.dispatch_selected_once(
        {
            "task_id": "t_ws012",
            "repo_full_name": "chriskim12/whystarve",
            "public_id": "WS-012",
        },
        board="brain-os",
    )

    assert result["dispatched"] is True
    worker_env = captured["worker_env"]
    assert worker_env["HERMES_KANBAN_AUTOPILOT"] == "1"
    assert worker_env["HERMES_KANBAN_REVIEW_READY_PR_REQUIRED"] == "1"
    assert worker_env["HERMES_KANBAN_EXPECTED_REPO_FULL_NAME"] == "chriskim12/whystarve"


def test_worktree_cleanup_registry_allows_removed_or_review_safe_retained_entries():
    from gateway.kanban_autopilot import build_worktree_cleanup_registry, evaluate_pre_review_cleanup_gate

    registry = build_worktree_cleanup_registry(
        [
            {
                "public_id": "BO-151",
                "path": "/repo/.worktrees/bo-151",
                "branch": "work/BO-151-cleanup-registry",
                "registered_git_worktree": True,
                "cleanup_state": "removed",
                "git_worktree_remove_verified": True,
            },
            {
                "public_id": "BO-152",
                "path": "/repo/.worktrees/bo-152",
                "branch": "work/BO-152-operator-reporting",
                "registered_git_worktree": True,
                "cleanup_state": "retained",
                "retained_reason": "open stacked fork PR review evidence",
                "ttl": "until fork PR stack is merged or abandoned",
                "review_safe": True,
            },
        ],
        bundle_id="BO-145",
    )

    assert registry["review_ready_allowed"] is True
    assert registry["cleanup_required"] is False
    assert registry["destructive_cleanup_performed"] is False
    gate = evaluate_pre_review_cleanup_gate(registry)
    assert gate["review_ready_allowed"] is True
    assert gate["post_merge_reconcile_still_required"] is True


def test_worktree_cleanup_registry_blocks_pending_active_or_unverified_entries():
    from gateway.kanban_autopilot import build_worktree_cleanup_registry, evaluate_pre_review_cleanup_gate

    registry = build_worktree_cleanup_registry(
        [
            {"public_id": "BO-151", "path": "/repo/.worktrees/bo-151", "registered_git_worktree": True, "cleanup_state": "pending"},
            {"public_id": "BO-152", "path": "/repo/.worktrees/bo-152", "registered_git_worktree": True, "cleanup_state": "retained", "retained_reason": "debug", "active_worker_pid": 123},
            {"public_id": "BO-153", "path": "/repo/.worktrees/bo-153", "cleanup_state": "removed", "git_worktree_remove_verified": True},
        ],
        bundle_id="BO-145",
    )

    gate = evaluate_pre_review_cleanup_gate(registry)
    assert gate["review_ready_allowed"] is False
    assert "cleanup_required" in gate["reason_codes"]
    assert "cleanup_pending" in gate["reason_codes"]
    assert "active_reference_blocks_cleanup" in gate["reason_codes"]
    assert "registered_git_worktree_not_verified" in gate["reason_codes"]


def test_post_merge_cleanup_reconcile_requires_merged_pr_truth_and_resolved_registry():
    from gateway.kanban_autopilot import build_worktree_cleanup_registry, reconcile_post_merge_cleanup

    registry = build_worktree_cleanup_registry(
        [
            {
                "public_id": "BO-151",
                "path": "/repo/.worktrees/bo-151",
                "registered_git_worktree": True,
                "cleanup_state": "removed",
                "git_worktree_remove_verified": True,
            },
            {
                "public_id": "BO-152",
                "path": "/repo/.worktrees/bo-152",
                "registered_git_worktree": True,
                "cleanup_state": "pending",
            },
        ],
        bundle_id="BO-145",
    )

    result = reconcile_post_merge_cleanup(
        registry,
        merged_prs=[{"public_id": "BO-151", "state": "MERGED", "merge_commit_sha": "abc1234"}],
    )

    assert result["closed_allowed"] is False
    assert result["gateway_restart_reload_allowed"] is False
    assert result["canonical_materialization_allowed"] is False
    assert result["reconciled"][0]["status"] == "reconciled"
    assert result["stale_entries"][0]["public_id"] == "BO-152"
    assert "missing_merged_pr_truth" in result["stale_entries"][0]["reason_codes"]
    assert "pre_review_cleanup_not_resolved" in result["stale_entries"][0]["reason_codes"]


def test_verifier_failure_operator_report_shows_retry_and_missing_criteria():
    from gateway.kanban_autopilot import generate_verifier_failure_operator_report

    report = generate_verifier_failure_operator_report(
        {"verdict": "FAIL", "missing_criteria": ["dc-02-tests", "dc-04-cleanup"]},
        {"retry_allowed": True, "retry_count": 0, "max_retries": 1},
    )

    assert report["summary"]["verdict"] == "FAIL"
    assert report["summary"]["next_state"] == "retry_queued"
    assert report["summary"]["retry_allowed"] is True
    assert report["missing_criteria"] == ["dc-02-tests", "dc-04-cleanup"]
    assert "dc-04-cleanup" in report["text"]
    assert report["authority"]["review_ready_allowed"] is False
    assert report["authority"]["merge_allowed"] is False


def test_verifier_failure_operator_report_shows_remediation_child_or_needs_human():
    from gateway.kanban_autopilot import generate_verifier_failure_operator_report

    child = generate_verifier_failure_operator_report(
        {"verdict": "FAIL", "reason_codes": ["missing_pr"]},
        {"retry_allowed": False, "retry_count": 1, "max_retries": 1, "remediation_child": "BO-999"},
    )
    blocked = generate_verifier_failure_operator_report(
        {"verdict": "BLOCKED", "missing_criteria": ["scope_escape"]},
        {"retry_allowed": False, "retry_count": 1, "max_retries": 1, "blocked_reason": "scope expansion needs Chris"},
    )

    assert child["summary"]["next_state"] == "remediation_child_queued"
    assert child["remediation"]["remediation_child"] == "BO-999"
    assert blocked["summary"]["needs_human"] is True
    assert blocked["summary"]["next_state"] == "needs_human"
    assert "scope expansion needs Chris" in blocked["text"]


def test_verifier_pass_operator_report_allows_only_review_ready_gate():
    from gateway.kanban_autopilot import generate_verifier_failure_operator_report

    report = generate_verifier_failure_operator_report({"verdict": "PASS"}, {"retry_allowed": False, "retry_count": 0, "max_retries": 1})

    assert report["summary"]["next_state"] == "review_ready_gate"
    assert report["authority"]["review_ready_allowed"] is True
    assert report["authority"]["gateway_restart_reload_allowed"] is False
    assert report["authority"]["config_env_secret_mutation_allowed"] is False


def test_e2e_check_only_worker_verifier_retry_and_success_paths_pass():
    from gateway.kanban_autopilot import prove_worker_verifier_retry_loop_check_only

    failure_path = prove_worker_verifier_retry_loop_check_only([
        {"kind": "worker_completed", "public_id": "BO-153"},
        {"kind": "verifier_intake", "public_id": "BO-153"},
        {"kind": "verifier_result", "verdict": "FAIL", "missing_criteria": ["dc-02"]},
        {"kind": "retry_queued", "public_id": "BO-153", "attempt": 2},
        {"kind": "parent_matrix_updated", "parent_public_id": "BO-145"},
    ])
    success_path = prove_worker_verifier_retry_loop_check_only([
        {"kind": "worker_completed", "public_id": "BO-153"},
        {"kind": "verifier_intake", "public_id": "BO-153"},
        {"kind": "verifier_result", "verdict": "PASS"},
        {"kind": "cleanup_checked", "public_id": "BO-153"},
        {"kind": "review_ready_promoted", "public_id": "BO-153"},
        {"kind": "parent_matrix_updated", "parent_public_id": "BO-145"},
    ])

    assert failure_path["passed"] is True
    assert failure_path["outcomes"][0]["next_state"] == "retry_or_remediation"
    assert success_path["passed"] is True
    assert success_path["outcomes"][0]["next_state"] == "review_ready"
    assert success_path["authority"]["check_only"] is True
    assert success_path["authority"]["gateway_restart_reload_allowed"] is False
    assert success_path["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_e2e_check_only_blocks_pass_without_cleanup_or_parent_matrix():
    from gateway.kanban_autopilot import prove_worker_verifier_retry_loop_check_only

    result = prove_worker_verifier_retry_loop_check_only([
        {"kind": "worker_completed", "public_id": "BO-153"},
        {"kind": "verifier_intake", "public_id": "BO-153"},
        {"kind": "verifier_result", "verdict": "PASS"},
    ])

    assert result["passed"] is False
    assert "verifier_pass_without_cleanup_check" in result["reason_codes"]
    assert "verifier_pass_without_review_ready_promotion" in result["reason_codes"]
    assert "missing_parent_matrix_update" in result["reason_codes"]
    assert result["next_state"] == "needs_human"


def test_e2e_check_only_blocks_unblocked_forbidden_mutation_attempt():
    from gateway.kanban_autopilot import prove_worker_verifier_retry_loop_check_only

    blocked = prove_worker_verifier_retry_loop_check_only([
        {"kind": "worker_completed"},
        {"kind": "verifier_intake"},
        {"kind": "verifier_result", "verdict": "FAIL"},
        {"kind": "remediation_child_queued", "public_id": "BO-999"},
        {"kind": "forbidden_mutation_attempt", "mutation": "gateway_restart"},
        {"kind": "authority_blocked", "mutation": "gateway_restart"},
        {"kind": "parent_matrix_updated"},
    ])
    unblocked = prove_worker_verifier_retry_loop_check_only([
        {"kind": "worker_completed"},
        {"kind": "verifier_intake"},
        {"kind": "verifier_result", "verdict": "FAIL"},
        {"kind": "retry_queued"},
        {"kind": "forbidden_mutation_attempt", "mutation": "gateway_restart"},
        {"kind": "parent_matrix_updated"},
    ])

    assert blocked["passed"] is True
    assert unblocked["passed"] is False
    assert "forbidden_attempt_not_blocked" in unblocked["reason_codes"]


def _promotable_child(public_id: str = "BO-PROMOTE") -> dict:
    child = _ready_candidate(public_id)
    child["id"] = "t_promote"
    child["status"] = "todo"
    child["assignee"] = None
    child["parent_public_id"] = "BO-PARENT"
    child["relation_type"] = "hierarchy"
    return child


def test_parent_scoped_child_promotion_makes_todo_child_dispatchable_without_second_dispatcher(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command, promote_parent_scoped_children

    handle_autopilot_command("on BO-PARENT", actor="tester")
    result = promote_parent_scoped_children([_promotable_child()], parent_public_id="BO-PARENT", dry_run=False)

    assert result["promoted"][0]["public_id"] == "BO-PROMOTE"
    assert result["candidates"][0]["status"] == "ready"
    assert result["candidates"][0]["assignee"] == "arisu"
    assert result["handoff_target"] == "existing_kanban_dispatcher"
    assert result["second_dispatcher_created"] is False
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 1, "dispatched": 0}


def test_parent_scoped_child_promotion_blocks_ambiguous_child_instead_of_guessing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import promote_parent_scoped_children

    child = _promotable_child("BO-AMBIGUOUS")
    child["body"] = "maybe improve this somehow"

    result = promote_parent_scoped_children([child], parent_public_id="BO-PARENT", dry_run=False)

    assert result["promoted"] == []
    assert result["blocked"][0]["public_id"] == "BO-AMBIGUOUS"
    assert "missing_goal" in result["blocked"][0]["reason_codes"]
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_parent_scoped_child_promotion_ignores_out_of_parent_children(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import promote_parent_scoped_children

    child = _promotable_child("BO-OTHER")
    child["parent_public_id"] = "BO-OTHER-PARENT"

    result = promote_parent_scoped_children([child], parent_public_id="BO-PARENT", dry_run=False)

    assert result["promoted"] == []
    assert result["out_of_scope"][0]["reason_codes"] == ["parent_scope_mismatch"]
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_parent_scoped_child_promotion_keeps_dependency_blocked_child_non_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import promote_parent_scoped_children

    child = _promotable_child("BO-DEP")
    child["relation_type"] = "dependency"

    result = promote_parent_scoped_children([child], parent_public_id="BO-PARENT", dry_run=False)

    assert result["promoted"] == []
    assert result["blocked"][0]["reason_codes"] == ["dependency_relation_blocks_ready_promotion"]


def test_parent_scoped_child_promotion_dry_run_reports_candidate_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import promote_parent_scoped_children

    result = promote_parent_scoped_children([_promotable_child()], parent_public_id="BO-PARENT", dry_run=True)

    assert result["would_promote"][0]["public_id"] == "BO-PROMOTE"
    assert result["promoted"] == []
    assert result["candidates"][0]["status"] == "ready"
    assert result["dry_run_side_effects"] == {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0}


def test_parent_on_promotes_child_before_single_flight_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from gateway.kanban_autopilot import handle_autopilot_command

    result = handle_autopilot_command("on BO-PARENT", actor="tester", candidates=[_promotable_child()])

    assert result.ok is True
    assert result.decision["promotion"]["promoted"][0]["public_id"] == "BO-PROMOTE"
    assert result.decision["single_flight"]["selected"]["public_id"] == "BO-PROMOTE"
    assert result.decision["single_flight"]["selected"]["task_id"] == "t_promote"
    assert result.decision["dispatch_result"] is not None
    assert result.decision["single_flight"]["handoff"]["target"] == "existing_kanban_dispatcher"
