#!/usr/bin/env python3
"""Focused regressions for autopilot delegated ingress wiring and retry routing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from walletdb.autopilot_delegated_ingress import (  # noqa: E402
    completion_packet_sidecar_path,
    ingest_autopilot_delegated_completion,
    plan_contract_failure_action,
    sha256_file,
)
from walletdb.delegation_contract import (  # noqa: E402
    GATE_ACCEPTED,
    GATE_REJECTED_INVALID,
    GATE_REJECTED_RETRY,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _step() -> dict:
    return {
        "id": "audit_alignment",
        "title": "Audit alignment",
        "kind": "agent",
        "prompt": "Run deterministic contract-bound audit and write required artifacts.",
    }


def _valid_packet(repo: Path) -> dict:
    md = repo / "autopilot_artifacts" / "audit_alignment.md"
    js = repo / "autopilot_artifacts" / "audit_alignment.json"
    return {
        "schema_version": "1.0.0",
        "task_id": "autopilot:audit_alignment",
        "run_id": "run_contract_ok",
        "task_class": "implementation",
        "started_at": "2026-03-10T00:00:00Z",
        "ended_at": "2026-03-10T00:01:00Z",
        "outcome": {
            "code": "SUCCESS",
            "reason": "Produced required delegated artifacts and validated deterministic checks.",
        },
        "operator_summary": "Delegated audit completed with required markdown/json artifacts and gate-bound self-check evidence for deterministic ingestion.",
        "deliverables": [
            {
                "deliverable_id": "artifact_1",
                "path": "autopilot_artifacts/audit_alignment.md",
                "type": "report",
                "sha256": sha256_file(md),
                "description": "Audit report",
            },
            {
                "deliverable_id": "artifact_2",
                "path": "autopilot_artifacts/audit_alignment.json",
                "type": "json",
                "sha256": sha256_file(js),
                "description": "Structured audit findings",
            },
        ],
        "evidence_summary": {
            "claims": [
                {
                    "claim": "required artifacts emitted",
                    "supporting_evidence_refs": [
                        "autopilot_artifacts/audit_alignment.md",
                        "autopilot_artifacts/audit_alignment.json",
                    ],
                }
            ],
            "sources": [
                {"ref": "autopilot_artifacts/audit_alignment.md", "kind": "file"}
            ],
        },
        "uncertainties": [
            {
                "item": "Residual edge-case coverage unknown.",
                "impact": "low",
                "recommended_next_step": "Expand breaktests if required by validator.",
            }
        ],
        "self_checks": {
            "results": [
                {
                    "test_id": "autopilot_exit_code",
                    "passed": True,
                    "exit_code": 0,
                    "notes": "Delegated execution exit code recorded as success.",
                }
            ]
        },
    }


def scenario_accepts_valid_contract_packet(repo: Path) -> None:
    artifacts = repo / "autopilot_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "audit_alignment.md").write_text("ok\n", encoding="utf-8")
    (artifacts / "audit_alignment.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    result = ingest_autopilot_delegated_completion(
        step=_step(),
        repo_path=repo,
        raw_completion=_valid_packet(repo),
        started_ts=1773100000,
    )

    _assert(result["gate_outcome"] == GATE_ACCEPTED, f"expected ACCEPTED, got {result['gate_outcome']}")
    _assert(Path(result["decision_path"]).exists(), "expected persisted gate decision")
    _assert(isinstance(result.get("gate_summary"), dict), "expected structured gate_summary")
    _assert(result["gate_summary"].get("queue_reason") == "autopilot_step_completed", "expected completed queue reason")


def scenario_sidecar_first_ingestion_persists_packet_evidence(repo: Path) -> None:
    artifacts = repo / "autopilot_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "audit_alignment.md").write_text("ok\n", encoding="utf-8")
    (artifacts / "audit_alignment.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    sidecar = completion_packet_sidecar_path(runs_dir=repo / "ops" / "autopilot" / "runs", run_base="20260310T000000Z_audit_alignment")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(_valid_packet(repo), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = ingest_autopilot_delegated_completion(
        step=_step(),
        repo_path=repo,
        raw_completion="provider_error: should be ignored because sidecar exists",
        started_ts=1773100000,
        completion_sidecar_path=sidecar,
    )

    _assert(result["gate_outcome"] == GATE_ACCEPTED, f"expected ACCEPTED, got {result['gate_outcome']}")
    _assert(result.get("completion_packet_source") == "sidecar", f"expected sidecar source, got {result}")
    _assert(result.get("completion_packet_path") == str(sidecar), f"expected packet path, got {result}")
    _assert(result.get("completion_packet_sha256") == sha256_file(sidecar), f"expected packet hash, got {result}")


def scenario_retry_class_routes_to_bounded_retry(repo: Path) -> None:
    result = ingest_autopilot_delegated_completion(
        step=_step(),
        repo_path=repo,
        raw_completion="Completed: wrote narrative only; artifacts omitted.",
        started_ts=1773100000,
    )

    _assert(result["gate_outcome"] == GATE_REJECTED_RETRY, f"expected REJECTED_RETRY, got {result['gate_outcome']}")

    first = plan_contract_failure_action(
        gate_outcome=GATE_REJECTED_RETRY,
        attempts=0,
        max_attempts=2,
        now_ts=1773100000,
        retry_policy=result.get("retry_policy") if isinstance(result.get("retry_policy"), dict) else None,
        retry_reasons=result.get("retry_reasons") if isinstance(result.get("retry_reasons"), list) else None,
        gate_summary=result.get("gate_summary") if isinstance(result.get("gate_summary"), dict) else None,
    )
    _assert(first["status"] == "queued", f"expected queued retry, got {first}")
    _assert(first["queue_reason"] == "autopilot_delegated_contract_retry_backoff", f"unexpected queue reason: {first}")
    _assert(int(first.get("retry_backoff_sec") or 0) == 120, f"expected contract-payload backoff=120s, got {first}")

    exhausted = plan_contract_failure_action(
        gate_outcome=GATE_REJECTED_RETRY,
        attempts=1,
        max_attempts=2,
        now_ts=1773100000,
        retry_policy=result.get("retry_policy") if isinstance(result.get("retry_policy"), dict) else None,
        retry_reasons=result.get("retry_reasons") if isinstance(result.get("retry_reasons"), list) else None,
        gate_summary=result.get("gate_summary") if isinstance(result.get("gate_summary"), dict) else None,
    )
    _assert(exhausted["status"] == "blocked", f"expected blocked retry exhaustion, got {exhausted}")
    _assert(
        exhausted["queue_reason"] == "autopilot_delegated_contract_retry_exhausted",
        f"unexpected exhausted reason: {exhausted}",
    )


def scenario_provider_error_text_does_not_bleed_into_gate_summary(repo: Path) -> None:
    raw = (
        "provider_error: OpenAIResponsesError status=503 endpoint=https://api.openai.com/v1/responses\n"
        "request_id=req_leak_123 token=sk-live-THIS_SHOULD_NOT_LEAK\n"
        "Traceback (most recent call last): ..."
    )
    result = ingest_autopilot_delegated_completion(
        step=_step(),
        repo_path=repo,
        raw_completion=raw,
        started_ts=1773100000,
    )

    _assert(result["gate_outcome"] == GATE_REJECTED_RETRY, f"expected REJECTED_RETRY, got {result['gate_outcome']}")
    summary = result.get("gate_summary") if isinstance(result.get("gate_summary"), dict) else {}
    primary_reason = str(summary.get("primary_reason") or "")
    summary_text = json.dumps(summary, ensure_ascii=False)

    _assert(summary.get("queue_reason") == "autopilot_delegated_provider_retry_backoff", f"unexpected queue reason: {summary}")
    _assert(summary.get("provider_failure_detected") is True, f"expected provider metadata: {summary}")
    _assert("sk-live" not in primary_reason, f"primary_reason leaked token-like text: {primary_reason}")
    _assert("sk-live" not in summary_text, f"summary leaked token-like text: {summary_text}")
    _assert("api.openai.com" not in summary_text, f"summary leaked provider endpoint: {summary_text}")
    _assert("req_leak_123" not in summary_text, f"summary leaked provider request id: {summary_text}")


def scenario_provider_nonretryable_escalates_invalid(repo: Path) -> None:
    raw = "provider_error: OpenAIResponsesError status=401 unauthorized invalid api key"
    result = ingest_autopilot_delegated_completion(
        step=_step(),
        repo_path=repo,
        raw_completion=raw,
        started_ts=1773100000,
    )

    _assert(result["gate_outcome"] == GATE_REJECTED_INVALID, f"expected REJECTED_INVALID, got {result['gate_outcome']}")
    summary = result.get("gate_summary") if isinstance(result.get("gate_summary"), dict) else {}
    _assert(summary.get("queue_reason") == "autopilot_delegated_provider_blocked_nonretryable", f"unexpected queue reason: {summary}")


def scenario_invalid_class_escalates_without_retry(repo: Path) -> None:
    env_probe = "python -V\nwhich python\n.venv/bin/python\n"
    result = ingest_autopilot_delegated_completion(
        step=_step(),
        repo_path=repo,
        raw_completion=env_probe,
        started_ts=1773100000,
    )

    _assert(result["gate_outcome"] == GATE_REJECTED_INVALID, f"expected REJECTED_INVALID, got {result['gate_outcome']}")
    _assert(result.get("gate_summary", {}).get("queue_reason") == "autopilot_delegated_contract_invalid", "expected invalid queue reason in summary")
    action = plan_contract_failure_action(
        gate_outcome=GATE_REJECTED_INVALID,
        attempts=0,
        max_attempts=5,
        now_ts=1773100000,
    )
    _assert(action["status"] == "blocked", f"expected blocked invalid escalation, got {action}")
    _assert(action["queue_reason"] == "autopilot_delegated_contract_invalid", f"unexpected invalid reason: {action}")


SCENARIOS = [
    ("accepts_valid_contract_packet", scenario_accepts_valid_contract_packet),
    ("sidecar_first_ingestion_persists_packet_evidence", scenario_sidecar_first_ingestion_persists_packet_evidence),
    ("retry_class_routes_to_bounded_retry", scenario_retry_class_routes_to_bounded_retry),
    ("provider_error_text_does_not_bleed_into_gate_summary", scenario_provider_error_text_does_not_bleed_into_gate_summary),
    ("provider_nonretryable_escalates_invalid", scenario_provider_nonretryable_escalates_invalid),
    ("invalid_class_escalates_without_retry", scenario_invalid_class_escalates_without_retry),
]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="autopilot_delegated_ingress_") as td:
        repo = Path(td)
        passed = 0
        for name, fn in SCENARIOS:
            fn(repo)
            passed += 1
            print(f"PASS: {name}")
    print(f"SUMMARY: {passed}/{len(SCENARIOS)} scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
