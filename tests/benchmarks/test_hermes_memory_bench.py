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
