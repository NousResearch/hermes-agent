import json
import subprocess
import sys

from benchmarks.hermes_memory_bench.core import DIMENSIONS, POLICY, run_benchmark


REQUIRED_TOP_LEVEL_KEYS = {
    "benchmark_type",
    "generated_at",
    "suite",
    "scores",
    "cases",
    "aggregate",
    "policy",
}


def test_smoke_benchmark_schema_and_policy():
    report = run_benchmark("smoke")

    assert REQUIRED_TOP_LEVEL_KEYS <= report.keys()
    assert report["benchmark_type"] == "hermes_memory_bench_v0.1"
    assert report["suite"] == "smoke"
    assert set(DIMENSIONS) <= report["scores"].keys()
    assert report["aggregate"]["overall_score"] == 1.0
    assert report["aggregate"]["case_count"] >= 6
    assert report["policy"] == POLICY

    for case in report["cases"]:
        assert {"id", "dimension", "query", "expected_answer", "actual_answer", "score", "latency_ms", "passed", "evidence"} <= case.keys()
        assert case["dimension"] in DIMENSIONS


def test_smoke_benchmark_cli_writes_report(tmp_path):
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.hermes_memory_bench.run",
            "--suite",
            "smoke",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aggregate"]["overall_score"] == 1.0
    assert report["policy"] == POLICY


def test_benchmark_does_not_write_graph_or_operation_ledger(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    report = run_benchmark("smoke")

    assert report["policy"]["read_only"] is True
    assert report["policy"]["would_write_memory"] is False
    assert report["policy"]["would_modify_config"] is False
    assert report["policy"]["would_write_graph"] is False
    assert report["policy"]["does_not_create_operation_events"] is True
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()


def test_memory_real_proposal_dry_run_smoke_case_is_preview_only():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_real_proposal_dry_run"
    )

    dry_run = case["evidence"]["real_proposal_dry_run_candidates"][0]
    assert case["actual_answer"] == "manual_final_preflight_required"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}
    assert dry_run["proposal_record_preview"]["written"] is False
    assert dry_run["operation_ledger_preview"]["written"] is False
    assert dry_run["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False


def test_memory_real_proposal_write_lock_gate_smoke_case_is_token_eligible_without_writes():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_real_proposal_write_lock_gate"
    )

    gate = case["evidence"]["real_proposal_write_lock_gate_candidates"][0]
    assert case["actual_answer"] == "eligible_for_human_approval_token"
    assert gate["gate_validation"] == {"valid": True, "errors": []}
    assert gate["gate_status"] == "eligible_for_human_approval_token"
    assert gate["lock_reason"] is None
    assert gate["proposal_record_preview"]["written"] is False
    assert gate["operation_ledger_preview"]["written"] is False
    assert gate["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False


def test_memory_human_approval_token_request_smoke_case_is_review_required_without_issuing_token():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_request"
    )

    request = case["evidence"]["human_approval_token_request_candidates"][0]
    assert case["actual_answer"] == "approval_token_review_required"
    assert request["request_validation"] == {"valid": True, "errors": []}
    assert request["request_status"] == "approval_token_review_required"
    assert request["lock_reason"] is None
    assert request["proposal_record_preview"]["written"] is False
    assert request["operation_ledger_preview"]["written"] is False
    assert request["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False


def test_memory_human_approval_token_review_gate_smoke_case_approves_without_issuing_token():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_review_gate"
    )

    outcome = case["evidence"]["human_approval_token_review_outcome_candidates"][0]
    assert case["actual_answer"] == "approve_token_issuance"
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert outcome["review_outcome_status"] == "token_review_outcome_candidate"
    assert outcome["outcome"] == "approve_token_issuance"
    assert outcome["proposal_record_preview"]["written"] is False
    assert outcome["operation_ledger_preview"]["written"] is False
    assert outcome["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False


def test_memory_human_approval_token_issuance_plan_smoke_case_requires_manual_plan_without_issuing_token():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_issuance_plan"
    )

    plan = case["evidence"]["human_approval_token_issuance_plan_candidates"][0]
    assert case["actual_answer"] == "manual_token_issuance_plan_required"
    assert plan["plan_validation"] == {"valid": True, "errors": []}
    assert plan["plan_status"] == "manual_token_issuance_plan_required"
    assert plan["lock_reason"] is None
    assert plan["outcome"] == "approve_token_issuance"
    assert plan["proposal_record_preview"]["written"] is False
    assert plan["operation_ledger_preview"]["written"] is False
    assert plan["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False


def test_memory_human_approval_token_issuance_dry_run_smoke_case_requires_final_preflight_without_issuing_token():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_issuance_dry_run"
    )

    dry_run = case["evidence"]["human_approval_token_issuance_dry_run_candidates"][0]
    assert case["actual_answer"] == "manual_token_issuance_final_preflight_required"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}
    assert dry_run["dry_run_status"] == "manual_token_issuance_final_preflight_required"
    assert dry_run["lock_reason"] is None
    assert dry_run["approval_token_record_preview"]["preview_only"] is True
    assert dry_run["approval_token_record_preview"]["token_issued"] is False
    assert dry_run["approval_token_record_preview"]["persisted"] is False
    assert dry_run["approval_audit_record_preview"]["preview_only"] is True
    assert dry_run["approval_audit_record_preview"]["created_operation_event"] is False
    assert dry_run["token_target_paths_preview"]["preview_only"] is True
    assert dry_run["token_target_paths_preview"]["writes_token_files"] is False
    assert dry_run["token_target_paths_preview"]["writes_approval_audit"] is False
    assert dry_run["proposal_record_preview"]["written"] is False
    assert dry_run["operation_ledger_preview"]["written"] is False
    assert dry_run["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False


def test_memory_human_approval_token_write_lock_gate_smoke_case_is_final_confirmation_eligible_without_token_writes():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_write_lock_gate"
    )

    gate = case["evidence"]["human_approval_token_write_lock_gate_candidates"][0]
    assert case["actual_answer"] == "eligible_for_final_human_confirmation"
    assert gate["gate_validation"] == {"valid": True, "errors": []}
    assert gate["gate_status"] == "eligible_for_final_human_confirmation"
    assert gate["lock_reason"] is None
    assert gate["approval_token_record_preview"]["preview_only"] is True
    assert gate["approval_token_record_preview"]["token_issued"] is False
    assert gate["approval_token_record_preview"]["persisted"] is False
    assert gate["approval_audit_record_preview"]["preview_only"] is True
    assert gate["approval_audit_record_preview"]["created_operation_event"] is False
    assert gate["token_target_paths_preview"]["preview_only"] is True
    assert gate["token_target_paths_preview"]["writes_token_files"] is False
    assert gate["token_target_paths_preview"]["writes_approval_audit"] is False
    assert gate["proposal_record_preview"]["written"] is False
    assert gate["operation_ledger_preview"]["written"] is False
    assert gate["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False


def test_memory_human_approval_token_final_confirmation_request_smoke_case_is_review_required_without_issuing_token():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_final_confirmation_request"
    )

    request = case["evidence"]["human_approval_token_final_confirmation_request_candidates"][0]
    assert case["actual_answer"] == "final_confirmation_review_required"
    assert request["request_validation"] == {"valid": True, "errors": []}
    assert request["request_status"] == "final_confirmation_review_required"
    assert request["lock_reason"] is None
    assert request["approval_token_record_preview"]["preview_only"] is True
    assert request["approval_token_record_preview"]["token_issued"] is False
    assert request["approval_token_record_preview"]["persisted"] is False
    assert request["approval_audit_record_preview"]["preview_only"] is True
    assert request["approval_audit_record_preview"]["created_operation_event"] is False
    assert request["token_target_paths_preview"]["preview_only"] is True
    assert request["token_target_paths_preview"]["writes_token_files"] is False
    assert request["token_target_paths_preview"]["writes_approval_audit"] is False
    assert request["proposal_record_preview"]["written"] is False
    assert request["operation_ledger_preview"]["written"] is False
    assert request["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False


def test_memory_human_approval_token_final_confirmation_review_gate_smoke_case_confirms_without_writes():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_final_confirmation_review_gate"
    )

    outcome = case["evidence"]["human_approval_token_final_confirmation_review_outcome_candidates"][0]
    assert case["actual_answer"] == "confirm_token_write"
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert outcome["outcome"] == "confirm_token_write"
    assert outcome["routing"] == "manual_token_write_still_required_after_final_confirmation"
    assert outcome["approval_token_record_preview"]["preview_only"] is True
    assert outcome["approval_token_record_preview"]["token_issued"] is False
    assert outcome["approval_token_record_preview"]["persisted"] is False
    assert outcome["approval_audit_record_preview"]["preview_only"] is True
    assert outcome["approval_audit_record_preview"]["created_operation_event"] is False
    assert outcome["token_target_paths_preview"]["preview_only"] is True
    assert outcome["token_target_paths_preview"]["writes_token_files"] is False
    assert outcome["token_target_paths_preview"]["writes_approval_audit"] is False
    assert outcome["proposal_record_preview"]["written"] is False
    assert outcome["operation_ledger_preview"]["written"] is False
    assert outcome["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False


def test_memory_human_approval_token_write_execution_plan_smoke_case_requires_dry_run_without_writes():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_write_execution_plan"
    )

    plan = case["evidence"]["human_approval_token_write_execution_plan_candidates"][0]
    assert case["actual_answer"] == "manual_token_write_execution_plan_required"
    assert plan["plan_validation"] == {"valid": True, "errors": []}
    assert plan["plan_status"] == "manual_token_write_execution_plan_required"
    assert plan["routing"] == "manual_token_write_dry_run_required_before_any_token_write"
    assert plan["lock_reason"] is None
    assert plan["approval_token_record_preview"]["preview_only"] is True
    assert plan["approval_token_record_preview"]["token_issued"] is False
    assert plan["approval_token_record_preview"]["persisted"] is False
    assert plan["approval_audit_record_preview"]["preview_only"] is True
    assert plan["approval_audit_record_preview"]["created_operation_event"] is False
    assert plan["token_target_paths_preview"]["preview_only"] is True
    assert plan["token_target_paths_preview"]["writes_token_files"] is False
    assert plan["token_target_paths_preview"]["writes_approval_audit"] is False
    assert plan["proposal_record_preview"]["written"] is False
    assert plan["operation_ledger_preview"]["written"] is False
    assert plan["operation_ledger_preview"]["created_operation_event"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False


def test_memory_human_approval_token_write_execution_dry_run_smoke_case_requires_final_preflight_without_writes():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_write_execution_dry_run"
    )

    dry_run = case["evidence"]["human_approval_token_write_execution_dry_run_candidates"][0]
    assert case["actual_answer"] == "manual_token_write_final_preflight_required"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}
    assert dry_run["dry_run_status"] == "manual_token_write_final_preflight_required"
    assert dry_run["routing"] == "manual_token_write_final_gate_required_before_any_token_write"
    assert dry_run["lock_reason"] is None
    assert dry_run["approval_token_write_payload_preview"]["preview_only"] is True
    assert dry_run["approval_token_write_payload_preview"]["token_issued"] is False
    assert dry_run["approval_token_write_payload_preview"]["persisted"] is False
    assert dry_run["approval_token_write_payload_preview"]["written"] is False
    assert dry_run["approval_audit_write_payload_preview"]["preview_only"] is True
    assert dry_run["approval_audit_write_payload_preview"]["created_operation_event"] is False
    assert dry_run["approval_audit_write_payload_preview"]["writes_approval_audit"] is False
    assert dry_run["approval_audit_write_payload_preview"]["writes_operation_ledger"] is False
    assert dry_run["token_write_target_paths_preview"]["preview_only"] is True
    assert dry_run["token_write_target_paths_preview"]["writes_token_files"] is False
    assert dry_run["token_write_target_paths_preview"]["writes_approval_audit"] is False
    assert dry_run["token_write_target_paths_preview"]["writes_operation_ledger"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False


def test_memory_human_approval_token_write_final_gate_smoke_case_is_executor_eligible_without_writes():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_write_final_gate"
    )

    gate = case["evidence"]["human_approval_token_write_final_gate_candidates"][0]
    assert case["actual_answer"] == "eligible_for_real_token_write_executor"
    assert gate["gate_validation"] == {"valid": True, "errors": []}
    assert gate["gate_status"] == "eligible_for_real_token_write_executor"
    assert gate["routing"] == "real_token_write_executor_required_but_not_invoked"
    assert gate["lock_reason"] is None
    assert gate["approval_token_write_payload_preview"]["preview_only"] is True
    assert gate["approval_token_write_payload_preview"]["token_issued"] is False
    assert gate["approval_token_write_payload_preview"]["persisted"] is False
    assert gate["approval_token_write_payload_preview"]["written"] is False
    assert gate["approval_audit_write_payload_preview"]["preview_only"] is True
    assert gate["approval_audit_write_payload_preview"]["created_operation_event"] is False
    assert gate["approval_audit_write_payload_preview"]["writes_approval_audit"] is False
    assert gate["approval_audit_write_payload_preview"]["writes_operation_ledger"] is False
    assert gate["token_write_target_paths_preview"]["preview_only"] is True
    assert gate["token_write_target_paths_preview"]["writes_token_files"] is False
    assert gate["token_write_target_paths_preview"]["writes_approval_audit"] is False
    assert gate["token_write_target_paths_preview"]["writes_operation_ledger"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False
    assert case["evidence"]["invokes_real_token_write_executor"] is False


def test_memory_human_approval_token_real_write_executor_contract_smoke_case_requires_contract_without_writes():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"] == "memory_human_approval_token_real_write_executor_contract"
    )

    contract = case["evidence"][
        "human_approval_token_real_write_executor_contract_candidates"
    ][0]
    assert case["actual_answer"] == "real_token_write_executor_contract_required"
    assert contract["contract_validation"] == {"valid": True, "errors": []}
    assert contract["contract_status"] == "real_token_write_executor_contract_required"
    assert (
        contract["routing"]
        == "real_token_write_executor_contract_review_required_before_implementation"
    )
    assert contract["lock_reason"] is None
    assert "invoke_real_token_write_executor" in contract["executor_forbidden_side_effects"]
    assert "implement_real_token_write_executor" in contract["executor_forbidden_side_effects"]
    assert "write_token_files" in contract["executor_forbidden_side_effects"]
    assert "write_approval_audit_files" in contract["executor_forbidden_side_effects"]
    assert contract["approval_token_write_payload_preview"]["preview_only"] is True
    assert contract["approval_token_write_payload_preview"]["token_issued"] is False
    assert contract["approval_token_write_payload_preview"]["persisted"] is False
    assert contract["approval_token_write_payload_preview"]["written"] is False
    assert contract["approval_audit_write_payload_preview"]["preview_only"] is True
    assert contract["approval_audit_write_payload_preview"]["created_operation_event"] is False
    assert contract["approval_audit_write_payload_preview"]["writes_approval_audit"] is False
    assert contract["approval_audit_write_payload_preview"]["writes_operation_ledger"] is False
    assert contract["token_write_target_paths_preview"]["preview_only"] is True
    assert contract["token_write_target_paths_preview"]["writes_token_files"] is False
    assert contract["token_write_target_paths_preview"]["writes_approval_audit"] is False
    assert contract["token_write_target_paths_preview"]["writes_operation_ledger"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False
    assert case["evidence"]["invokes_real_token_write_executor"] is False
    assert case["evidence"]["implements_real_token_write_executor"] is False


def test_memory_human_approval_token_real_write_executor_contract_review_gate_smoke_case_approves_without_writes_or_executor():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"]
        == "memory_human_approval_token_real_write_executor_contract_review_gate"
    )

    outcome = case["evidence"][
        "human_approval_token_real_write_executor_contract_review_outcome_candidates"
    ][0]
    assert case["actual_answer"] == "approve_executor_contract"
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert (
        outcome["review_outcome_status"]
        == "real_write_executor_contract_review_outcome_candidate"
    )
    assert (
        outcome["routing"]
        == "real_token_write_executor_implementation_plan_required_after_contract_approval"
    )
    assert outcome["outcome"] == "approve_executor_contract"
    assert outcome["contract_validation"] == {"valid": True, "errors": []}
    assert "real_token_write_executor_not_invoked" in outcome["contract_review_checklist"]
    assert "real_token_write_executor_not_implemented" in outcome[
        "contract_review_checklist"
    ]
    assert outcome["approval_token_write_payload_preview"]["preview_only"] is True
    assert outcome["approval_token_write_payload_preview"]["token_issued"] is False
    assert outcome["approval_token_write_payload_preview"]["persisted"] is False
    assert outcome["approval_token_write_payload_preview"]["written"] is False
    assert outcome["approval_audit_write_payload_preview"]["preview_only"] is True
    assert outcome["approval_audit_write_payload_preview"]["created_operation_event"] is False
    assert outcome["approval_audit_write_payload_preview"]["writes_approval_audit"] is False
    assert outcome["approval_audit_write_payload_preview"]["writes_operation_ledger"] is False
    assert outcome["token_write_target_paths_preview"]["preview_only"] is True
    assert outcome["token_write_target_paths_preview"]["writes_token_files"] is False
    assert outcome["token_write_target_paths_preview"]["writes_approval_audit"] is False
    assert outcome["token_write_target_paths_preview"]["writes_operation_ledger"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False
    assert case["evidence"]["invokes_real_token_write_executor"] is False
    assert case["evidence"]["implements_real_token_write_executor"] is False


def test_memory_human_approval_token_real_write_executor_implementation_plan_smoke_case_requires_plan_without_writes_or_executor():
    report = run_benchmark("smoke")
    case = next(
        case
        for case in report["cases"]
        if case["dimension"]
        == "memory_human_approval_token_real_write_executor_implementation_plan"
    )

    plan = case["evidence"][
        "human_approval_token_real_write_executor_implementation_plan_candidates"
    ][0]
    assert case["actual_answer"] == "real_token_write_executor_implementation_plan_required"
    assert plan["plan_validation"] == {"valid": True, "errors": []}
    assert plan["plan_status"] == "real_token_write_executor_implementation_plan_required"
    assert (
        plan["routing"]
        == "real_token_write_executor_implementation_dry_run_required_before_code_implementation"
    )
    assert plan["lock_reason"] is None
    assert plan["outcome"] == "approve_executor_contract"
    assert plan["contract_review_outcome_validation"] == {"valid": True, "errors": []}
    assert all(
        interface["implemented_in_v0_1"] is False
        for interface in plan["implementation_plan_interfaces"]
    )
    assert all(
        interface["invoked_in_v0_1"] is False
        for interface in plan["implementation_plan_interfaces"]
    )
    assert all(file_plan["create_in_v0_1"] is False for file_plan in plan["implementation_plan_files"])
    assert all(
        file_plan["contains_executor_code_in_v0_1"] is False
        for file_plan in plan["implementation_plan_files"]
    )
    assert (
        plan["implementation_plan_audit_strategy"]["v0_1_effect"]
        == "no_approval_audit_file_write_and_no_operation_ledger_event"
    )
    assert (
        plan["implementation_plan_rollback_strategy"]["v0_1_effect"]
        == "no_rollback_action_because_no_write_is_performed"
    )
    for forbidden in (
        "implement_real_token_write_executor",
        "invoke_real_token_write_executor",
        "write_token_files",
        "write_approval_audit_files",
        "write_proposal_files",
        "write_operation_ledger",
    ):
        assert forbidden in plan["implementation_plan_forbidden_actions"]
    assert plan["approval_token_write_payload_preview"]["preview_only"] is True
    assert plan["approval_token_write_payload_preview"]["token_issued"] is False
    assert plan["approval_token_write_payload_preview"]["persisted"] is False
    assert plan["approval_token_write_payload_preview"]["written"] is False
    assert plan["approval_audit_write_payload_preview"]["preview_only"] is True
    assert plan["approval_audit_write_payload_preview"]["created_operation_event"] is False
    assert plan["approval_audit_write_payload_preview"]["writes_approval_audit"] is False
    assert plan["approval_audit_write_payload_preview"]["writes_operation_ledger"] is False
    assert plan["token_write_target_paths_preview"]["preview_only"] is True
    assert plan["token_write_target_paths_preview"]["writes_token_files"] is False
    assert plan["token_write_target_paths_preview"]["writes_approval_audit"] is False
    assert plan["token_write_target_paths_preview"]["writes_operation_ledger"] is False
    assert case["evidence"]["token_issued"] is False
    assert case["evidence"]["persisted_approval"] is False
    assert case["evidence"]["approved"] is False
    assert case["evidence"]["created_real_proposal"] is False
    assert case["evidence"]["created_operation_event"] is False
    assert case["evidence"]["writes_proposal_files"] is False
    assert case["evidence"]["writes_operation_ledger"] is False
    assert case["evidence"]["writes_token_files"] is False
    assert case["evidence"]["writes_approval_audit"] is False
    assert case["evidence"]["invokes_real_token_write_executor"] is False
    assert case["evidence"]["implements_real_token_write_executor"] is False
