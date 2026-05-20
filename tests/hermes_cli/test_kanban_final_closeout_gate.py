"""Tests for the read-only kanban final closeout gate helper."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hermes_cli.kanban_final_closeout_gate import build_closeout_gate_receipt, main


APPROVED_HEAD = "abc123approved"
CHANGED_HEAD = "def456changed"
ISSUE_URL = "https://github.com/GTZhou/TianGongKaiWu/issues/157"
PR_URL = "https://github.com/NousResearch/hermes-agent/pull/99999"


def _executor_manifest(*, head: str = APPROVED_HEAD, trace: bool = True, public_summary: str = "clean public text") -> dict:
    return {
        "schema": "tiangongkaiwu.executor_handoff.v1",
        "handoff_state": "ready_for_review",
        "issue": {"repo": "GTZhou/TianGongKaiWu", "number": 157, "url": ISSUE_URL},
        "pr": {"url": PR_URL, "number": 99999, "branch": "feat/issue-157-final-closeout-gate", "base_branch": "main", "head_sha": head},
        "artifact": {
            "type": "pr",
            "url": PR_URL,
            "commit_or_head_sha": head,
            "changed_files": ["hermes_cli/kanban_final_closeout_gate.py"],
            "evidence_files": [],
        },
        "kanban": {
            "board": "tiangongkaiwu",
            "task_id": "t_48bed874",
            "run_id": 213,
            "assignee": "jiangzuodajiang",
            "parent_task_ids": ["t_4ca7e33c", "t_7fbdfa7f"],
            "child_task_ids": ["t_ee5afc59"],
            "downstream_review_task_id": "t_ee5afc59",
        },
        "actor": {
            "agent_name": "将作大匠",
            "profile": "jiangzuodajiang",
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "reasoning_effort": None,
        },
        "scope": {
            "risk_level": "small",
            "acceptance_criteria": ["输出 pass/fail、缺口、证据 URL 和下一步"],
            "non_goals": ["不 merge", "不 close", "不改 label", "不发 trace"],
            "known_gaps": [],
            "public_summary": public_summary,
        },
        "tests": {"status": "passed", "commands": ["python3 -m pytest tests/hermes_cli/test_kanban_final_closeout_gate.py -q -o addopts="], "passed": ["target tests"], "failed": [], "known_failures": []},
        "trace": {
            "receipts": [
                {
                    "event": "submitted",
                    "idempotency_key": "t_48bed874-submitted-abc123approved",
                    "delivery_status": "sent",
                    "receipt_url_or_comment_url": f"{ISSUE_URL}#issuecomment-submitted",
                    "target_alias": "telegram:中书门下",
                }
            ]
            if trace
            else []
        },
        "labels": {"before": ["执行中"], "after": ["已提交"], "target_state": "submitted"},
        "security_scan": {
            "secret_scan": {"status": "passed", "scope": "changed_files", "summary": "no credential material"},
            "raw_locator_scan": {"status": "passed", "scope": "changed_files", "summary": "no raw platform locator"},
            "credential_material_touched": False,
            "no_core_or_profile_changes": True,
        },
        "identity_block": {
            "required": True,
            "location": "issue_comment",
            "exact_header": "将作大匠｜openai-codex/gpt-5.5｜reasoning_effort=null",
            "verified_against_runtime": True,
        },
        "next": {"owner": "reviewer", "review_child_task_id": "t_ee5afc59", "suggested_action": "review"},
    }


def _review_manifest(*, decision: str = "approved", approved_head: str = APPROVED_HEAD, trace: bool = True) -> dict:
    return {
        "schema": "tiangongkaiwu.review_decision.v1",
        "review_decision": decision,
        "issue": {"repo": "GTZhou/TianGongKaiWu", "number": 157, "url": ISSUE_URL},
        "artifact": {"pr_url": PR_URL, "artifact_url": PR_URL, "submitted_head_sha_or_artifact_id": approved_head, "changed_files": ["hermes_cli/kanban_final_closeout_gate.py"]},
        "kanban": {"board": "tiangongkaiwu", "review_task_id": "t_ee5afc59", "review_run_id": 214, "executor_task_id": "t_48bed874"},
        "actor": {"agent_name": "御史大夫", "profile": "yushidafu", "model": "gpt-5.5", "provider": "openai-codex", "reasoning_effort": None},
        "review_input": {
            "executor_manifest_url_or_comment_url": f"{ISSUE_URL}#issuecomment-executor",
            "review_scope": ["helper module", "tests", "docs"],
            "acceptance_criteria": ["gate-only closeout check"],
            "non_goals": ["no merge", "no close"],
            "required_gates": {
                "head_or_artifact_matches": "passed",
                "tests_evidence_present": "passed",
                "secret_scan_present": "passed",
                "raw_locator_scan_present": "passed",
                "lifecycle_label_readable": "passed",
                "trace_receipts_readable": "passed" if trace else "failed",
                "identity_block_verified": "passed",
            },
        },
        "checks_reviewed": {"commands_or_queries": [], "files_or_urls_reviewed": [PR_URL], "tests_or_ci_reviewed": ["target pytest"], "skipped_checks": []},
        "blocking_findings": [],
        "non_blocking_notes": [],
        "safety_check": {"conclusion": "passed", "secret_scan_summary": "clean", "raw_locator_scan_summary": "clean", "credential_material_touched": False},
        "trace": {
            "receipts": [
                {
                    "event": "approved",
                    "idempotency_key": "t_ee5afc59-approved-abc123approved",
                    "delivery_status": "sent",
                    "receipt_url_or_comment_url": f"{ISSUE_URL}#issuecomment-approved",
                    "target_alias": "telegram:中书门下",
                }
            ]
            if trace
            else []
        },
        "labels": {"before": ["已提交"], "after": ["已审核通过"], "target_state": "approved"},
        "identity_block": {"required": True, "location": "review_comment", "exact_header": "御史大夫｜openai-codex/gpt-5.5｜reasoning_effort=null", "verified_against_runtime": True},
        "next": {"owner": "closeout", "action": "final_closeout"},
    }


def _closeout_manifest(*, current_head: str = APPROVED_HEAD, child_terminal: bool = True, trace: bool = True) -> dict:
    return {
        "schema": "tiangongkaiwu.final_closeout_gate.v1",
        "closeout_gate": "passed",
        "issue": {"repo": "GTZhou/TianGongKaiWu", "number": 157, "url": ISSUE_URL, "state_before": "open", "state_after": "open"},
        "artifact": {"pr_url": PR_URL, "artifact_url": PR_URL, "approved_head_sha_or_artifact_id": APPROVED_HEAD, "current_head_sha_or_artifact_id": current_head, "head_unchanged_since_approval": current_head == APPROVED_HEAD},
        "kanban": {"board": "tiangongkaiwu", "closeout_task_id": "t_final", "closeout_run_id": 215, "executor_task_id": "t_48bed874", "review_task_id": "t_ee5afc59", "rework_task_ids": [], "child_tasks_terminal": child_terminal},
        "actor": {"agent_name": "枢密使", "profile": "shumishi", "model": "gpt-5.5", "provider": "openai-codex", "reasoning_effort": None},
        "approved_inputs": {
            "executor_manifest_url_or_comment_url": f"{ISSUE_URL}#issuecomment-executor",
            "review_decision_url_or_comment_url": f"{ISSUE_URL}#issuecomment-review",
            "rework_response_urls_or_comment_urls": [],
            "approved_review_decision": "approved",
            "approved_at": "2026-05-20T10:00:00Z",
        },
        "gate_checks": {
            "artifact_matches_approval": {"status": "passed", "evidence": PR_URL},
            "review_decision_approved": {"status": "passed", "evidence": f"{ISSUE_URL}#issuecomment-review"},
            "required_tests_or_checks": {"status": "passed", "evidence": ["target pytest passed"]},
            "lifecycle_labels_readable": {"status": "passed", "labels_before": ["已审核通过"], "labels_after": ["已审核通过"]},
            "trace_receipts_complete": {"status": "passed" if trace else "failed", "receipts": []},
            "kanban_task_graph_terminal": {"status": "passed" if child_terminal else "failed", "evidence": "all child tasks terminal" if child_terminal else "review child still running"},
            "public_text_sanitation": {"status": "passed", "secret_scan_summary": "clean", "raw_locator_scan_summary": "clean"},
            "identity_block_verified": {"status": "passed", "evidence": "identity block matches actor/model/provider"},
        },
        "closeout_action": {"closeout_comment_url": None, "pr_merged": False, "issue_closed": False, "label_transition": "not_performed", "no_reaudit_performed": True, "no_deploy_or_restart_performed": True},
        "failure": {"reason": None, "next_owner": "none", "next_action": "none"},
        "trace": {"receipts": [] if not trace else [{"event": "final_review_started", "idempotency_key": "t_final-final-review-started", "delivery_status": "sent", "receipt_url_or_comment_url": f"{ISSUE_URL}#issuecomment-final", "target_alias": "telegram:中书门下"}]},
        "identity_block": {"required": True, "location": "closeout_comment", "exact_header": "枢密使｜openai-codex/gpt-5.5｜reasoning_effort=null", "verified_against_runtime": True},
    }


def _guard_receipt(*, duplicate_groups: int = 0) -> dict:
    return {
        "schema": "kanban-duplicate-child-guard:receipt:v1",
        "mode": "dry_run",
        "ok": True,
        "board": {"slug": "tiangongkaiwu", "read_only": True},
        "detector": {"duplicate_groups": [{} for _ in range(duplicate_groups)], "insufficiently_marked_review_children": []},
        "dry_run_plan": {"apply_enabled": False, "apply_supported": False, "actions": []},
        "public_safety": {"no_mutations_performed": True, "raw_platform_locator_included": False, "credentials_or_tokens_touched": False},
    }


def _pass_receipt(**overrides):
    manifests = [
        overrides.pop("executor", _executor_manifest()),
        overrides.pop("review", _review_manifest()),
        overrides.pop("closeout", _closeout_manifest()),
    ]
    return build_closeout_gate_receipt(
        manifests=manifests,
        guard_receipts=[overrides.pop("guard", _guard_receipt())],
        issue=157,
        pr=99999,
        task_id="t_48bed874",
        repo="GTZhou/TianGongKaiWu",
        board="tiangongkaiwu",
        **overrides,
    )


def _gap_codes(receipt: dict) -> set[str]:
    return {gap["code"] for gap in receipt["gaps"]}


def test_gate_passes_when_approved_manifests_and_guard_receipt_are_consistent():
    receipt = _pass_receipt()

    assert receipt["schema"] == "kanban-final-closeout-gate:receipt:v1"
    assert receipt["ok"] is True
    assert receipt["closeout_gate"] == "passed"
    assert receipt["gaps"] == []
    assert receipt["gate_checks"]["artifact_matches_approval"]["status"] == "passed"
    assert receipt["gate_checks"]["duplicate_child_guard_receipt"]["status"] == "passed"
    assert receipt["public_safety"]["no_mutations_performed"] is True
    assert receipt["next"]["action"] == "closeout"


def test_gate_fails_when_current_head_changed_after_approval():
    receipt = _pass_receipt(closeout=_closeout_manifest(current_head=CHANGED_HEAD), current_head=CHANGED_HEAD)

    assert receipt["closeout_gate"] == "failed"
    assert "head_changed_after_approval" in _gap_codes(receipt)
    assert receipt["gate_checks"]["artifact_matches_approval"]["status"] == "failed"
    assert receipt["next"]["action"] == "re_review"


def test_gate_fails_when_trace_receipts_are_missing():
    receipt = _pass_receipt(
        executor=_executor_manifest(trace=False),
        review=_review_manifest(trace=False),
        closeout=_closeout_manifest(trace=False),
    )

    assert receipt["closeout_gate"] == "failed"
    assert "trace_receipt_missing" in _gap_codes(receipt)
    assert receipt["gate_checks"]["trace_receipts_complete"]["status"] == "failed"
    assert receipt["next"]["action"] == "provide_evidence"


def test_gate_fails_when_child_task_graph_is_not_terminal():
    receipt = _pass_receipt(closeout=_closeout_manifest(child_terminal=False))

    assert receipt["closeout_gate"] == "failed"
    assert "child_not_terminal" in _gap_codes(receipt)
    assert receipt["gate_checks"]["kanban_task_graph_terminal"]["status"] == "failed"


def test_gate_fails_when_public_text_scan_finds_secret_or_raw_locator():
    fake_token = "gho_" + "123456789012345678901234"
    raw_locator = "telegram:" + "-1001234567890" + ":17585"
    receipt = _pass_receipt(public_text=[f"leaked token {fake_token} and {raw_locator}"])

    assert receipt["closeout_gate"] == "failed"
    assert "public_text_scan_failed" in _gap_codes(receipt)

    check = receipt["gate_checks"]["public_text_sanitation"]
    assert check["status"] == "failed"
    assert check["secret_scan_summary"].startswith("failed")
    assert check["raw_locator_scan_summary"].startswith("failed")


def test_gate_fails_when_public_text_scan_finds_json_encoded_sensitive_fields():
    secret_field_name = "api" + "_key"
    locator_field_name = "chat" + "_id"
    fake_secret_value = "abcdefghijklmnop1234"
    raw_locator_value = int("-100" + "1234567890")
    public_text = json.dumps({secret_field_name: fake_secret_value, locator_field_name: raw_locator_value})

    receipt = _pass_receipt(public_text=[public_text])

    assert receipt["closeout_gate"] == "failed"
    assert "public_text_scan_failed" in _gap_codes(receipt)
    check = receipt["gate_checks"]["public_text_sanitation"]
    assert check["secret_scan_summary"].startswith("failed")
    assert check["raw_locator_scan_summary"].startswith("failed")


def test_failed_trace_receipt_output_redacts_raw_locator_and_secret_values():
    raw_chat_id = "-100" + "1234567890"
    raw_locator = "telegram:" + raw_chat_id + ":17585"
    fake_secret_value = "abcdefghijklmnop1234"
    secret_field_name = "access" + "_token"
    executor = _executor_manifest()
    executor["trace"]["receipts"].append(
        {
            "event": "submitted",
            "delivery_status": "failed",
            "receipt_url_or_comment_url": raw_locator,
            "target_alias": "telegram:中书门下",
            "chat_id": int(raw_chat_id),
            secret_field_name: fake_secret_value,
        }
    )

    receipt = _pass_receipt(executor=executor)
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)

    assert receipt["closeout_gate"] == "failed"
    assert "public_text_scan_failed" in _gap_codes(receipt)
    assert receipt["gate_checks"]["trace_receipts_complete"]["status"] == "failed"
    assert receipt["gate_checks"]["public_text_sanitation"]["status"] == "failed"
    raw_locator_present = raw_locator in serialized
    raw_chat_id_present = raw_chat_id in serialized
    fake_secret_present = fake_secret_value in serialized
    assert raw_locator_present is False
    assert raw_chat_id_present is False
    assert fake_secret_present is False
    assert "telegram:中书门下" in serialized


def test_gate_fails_when_identity_block_is_missing_from_required_manifest():
    executor = _executor_manifest()
    del executor["identity_block"]

    receipt = _pass_receipt(executor=executor)

    assert receipt["closeout_gate"] == "failed"
    assert "identity_block_mismatch" in _gap_codes(receipt)
    assert receipt["gate_checks"]["identity_block_verified"]["status"] == "failed"


def test_gate_requires_review_manifest_decision_not_closeout_claim_only():
    review = _review_manifest()
    del review["review_decision"]

    receipt = _pass_receipt(review=review)

    assert receipt["closeout_gate"] == "failed"
    assert "review_not_approved" in _gap_codes(receipt)
    assert receipt["gate_checks"]["review_decision_approved"]["status"] == "failed"


def test_gate_fails_when_duplicate_guard_has_dry_run_actions():
    guard = _guard_receipt()
    guard["dry_run_plan"]["actions"] = [{"type": "would_create_review_child"}]

    receipt = _pass_receipt(guard=guard)

    assert receipt["closeout_gate"] == "failed"
    assert "duplicate_child_guard_failed" in _gap_codes(receipt)
    assert receipt["gate_checks"]["duplicate_child_guard_receipt"]["status"] == "failed"


def test_gate_fails_when_trace_receipts_are_only_partially_present():
    review = _review_manifest(trace=False)
    review["review_input"]["required_gates"]["trace_receipts_readable"] = "passed"
    closeout = _closeout_manifest(trace=False)
    closeout["gate_checks"]["trace_receipts_complete"]["status"] = "passed"

    receipt = _pass_receipt(review=review, closeout=closeout)

    assert receipt["closeout_gate"] == "failed"
    assert "trace_receipt_missing" in _gap_codes(receipt)
    check = receipt["gate_checks"]["trace_receipts_complete"]
    assert check["status"] == "failed"
    assert "tiangongkaiwu.review_decision.v1" in check["missing_receipt_schemas"]
    assert "tiangongkaiwu.final_closeout_gate.v1" in check["missing_receipt_schemas"]


def test_gate_fails_when_artifact_match_gate_is_declared_failed():
    closeout = _closeout_manifest()
    closeout["gate_checks"]["artifact_matches_approval"]["status"] = "failed"

    receipt = _pass_receipt(closeout=closeout)

    assert receipt["closeout_gate"] == "failed"
    assert "artifact_head_declared_failed" in _gap_codes(receipt)
    assert receipt["gate_checks"]["artifact_matches_approval"]["status"] == "failed"


def test_cli_check_writes_json_receipt_and_markdown_report(tmp_path: Path):
    manifest_file = tmp_path / "manifests.yaml"
    manifest_file.write_text("\n---\n".join(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False) for doc in [_executor_manifest(), _review_manifest(), _closeout_manifest()]), encoding="utf-8")
    guard_file = tmp_path / "guard.json"
    guard_file.write_text(json.dumps(_guard_receipt(), ensure_ascii=False), encoding="utf-8")
    receipt_file = tmp_path / "receipt.json"
    report_file = tmp_path / "report.md"

    exit_code = main([
        "check",
        "--issue",
        "157",
        "--pr",
        "99999",
        "--task-id",
        "t_48bed874",
        "--repo",
        "GTZhou/TianGongKaiWu",
        "--board",
        "tiangongkaiwu",
        "--manifest",
        str(manifest_file),
        "--guard-receipt",
        str(guard_file),
        "--receipt-file",
        str(receipt_file),
        "--markdown-report-file",
        str(report_file),
        "--json",
    ])

    assert exit_code == 0
    data = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert data["closeout_gate"] == "passed"
    report = report_file.read_text(encoding="utf-8")
    assert "closeout_gate: passed" in report
    assert "No mutations performed" in report
