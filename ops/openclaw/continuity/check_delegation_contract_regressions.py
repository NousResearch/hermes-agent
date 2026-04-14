#!/usr/bin/env python3
"""Focused regression harness for delegated completion ingress fail-close behavior."""

from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from walletdb.delegation_contract import (  # noqa: E402
    GATE_ACCEPTED,
    GATE_REJECTED_INVALID,
    GATE_REJECTED_RETRY,
    INGRESS_INVALID,
    INGRESS_RETRY_REQUIRED,
    evaluate_gate,
    evaluate_ingress_gate,
)


def _brief(task_id: str = "task.impl.contractslice") -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "created_at": "2026-03-10T00:00:00+07:00",
        "task_class": "implementation",
        "objective": "Implement delegated completion ingress hardening and deterministic gate verification.",
        "scope": {
            "must_do": ["Enforce fail-closed ingress", "Verify deliverables/tests independently"],
            "must_not_do": ["Do not bypass contract validation"],
        },
        "inputs": {
            "workspace_root": ".",
            "artifacts": [{"name": "repo", "type": "directory", "ref": "."}],
        },
        "deliverables_spec": [
            {
                "deliverable_id": "contract_report",
                "type": "report",
                "path": "reports/contract_slice.md",
                "required": True,
            }
        ],
        "acceptance": {
            "criteria": [
                {
                    "criterion_id": "ingress_fail_closed",
                    "type": "machine",
                    "statement": "Malformed delegated ingress is rejected deterministically.",
                    "required": True,
                }
            ],
            "tests": [
                {
                    "test_id": "pytest_contract",
                    "runner": "shell",
                    "command": "./.venv/bin/pytest -q tests/test_delegation_contract.py",
                    "expected_exit_code": 0,
                    "required": True,
                }
            ],
        },
        "quality_bar": "normal",
    }


def _completion(sha_value: str, *, path: str = "reports/contract_slice.md", exit_code: int = 0) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "task_id": "task.impl.contractslice",
        "run_id": "run_20260310T163000Z",
        "task_class": "implementation",
        "started_at": "2026-03-10T16:25:00+07:00",
        "ended_at": "2026-03-10T16:29:00+07:00",
        "outcome": {
            "code": "SUCCESS",
            "confidence": 0.9,
            "intent_alignment": 0.91,
            "reason": "Completed delegated execution contract hardening with validation updates.",
        },
        "operator_summary": "Hardened delegated completion ingress and gate-side verification with deterministic contract checks and focused regression coverage.",
        "deliverables": [
            {
                "deliverable_id": "contract_report",
                "path": path,
                "type": "report",
                "sha256": sha_value,
                "description": "Delegated contract hardening report",
            }
        ],
        "evidence_summary": {
            "highlights": "Ingress is compiled through a fail-closed path. Gate verifies required artifact path/integrity and expected test exit codes before accepting success.",
            "claims": [
                {
                    "claim": "Malformed delegated ingress now receives deterministic retry/invalid outcomes.",
                    "supporting_evidence_refs": [
                        path,
                        "ops/openclaw/continuity/check_delegation_contract_regressions.py",
                    ],
                }
            ],
            "sources": [{"source_id": "repo", "kind": "repo", "ref": "."}],
        },
        "uncertainties": [],
        "self_checks": {
            "acceptance_tests_ran": True,
            "results": [{"test_id": "pytest_contract", "exit_code": exit_code, "passed": exit_code == 0}],
        },
    }


def _write_file(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256(text.encode("utf-8")).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scenario_accepts_valid_contract_packet(workspace: Path) -> None:
    digest = _write_file(workspace / "reports/contract_slice.md", "delegated contract slice")
    decision = evaluate_gate(_brief(), _completion(digest), workspace_root=workspace)
    _assert(decision["gate_outcome"] == GATE_ACCEPTED, f"expected ACCEPTED, got {decision['gate_outcome']}")


def scenario_rejects_env_probe_ingress_invalid(workspace: Path) -> None:
    raw = "python --version\nwhich python\nls -la .venv\npwd"
    decision = evaluate_ingress_gate(_brief(), raw, workspace_root=workspace)
    _assert(decision["gate_outcome"] == GATE_REJECTED_INVALID, f"expected REJECTED_INVALID, got {decision['gate_outcome']}")
    _assert(decision.get("ingress", {}).get("classification") == INGRESS_INVALID, "expected ingress classification INVALID")


def scenario_rejects_partial_packet_retry(workspace: Path) -> None:
    partial = {
        "task_id": "task.impl.contractslice",
        "operator_summary": "Synthesis-only packet without canonical fields.",
    }
    decision = evaluate_ingress_gate(_brief(), partial, workspace_root=workspace)
    _assert(decision["gate_outcome"] == GATE_REJECTED_RETRY, f"expected REJECTED_RETRY, got {decision['gate_outcome']}")
    _assert(decision.get("ingress", {}).get("classification") == INGRESS_RETRY_REQUIRED, "expected ingress classification RETRY_REQUIRED")


def scenario_rejects_required_path_mismatch_invalid(workspace: Path) -> None:
    _write_file(workspace / "reports/contract_slice.md", "delegated contract slice")
    bad_digest = _write_file(workspace / "reports/unexpected.md", "unexpected artifact")
    decision = evaluate_gate(_brief(), _completion(bad_digest, path="reports/unexpected.md"), workspace_root=workspace)
    _assert(decision["gate_outcome"] == GATE_REJECTED_INVALID, f"expected REJECTED_INVALID, got {decision['gate_outcome']}")
    _assert(any("path mismatch" in item for item in decision.get("reasons", {}).get("invalid", [])), "expected path mismatch invalid reason")


def scenario_rejects_success_exit_code_mismatch_invalid(workspace: Path) -> None:
    digest = _write_file(workspace / "reports/contract_slice.md", "delegated contract slice")
    completion = _completion(digest, exit_code=1)
    completion["self_checks"]["results"][0]["passed"] = True
    decision = evaluate_gate(_brief(), completion, workspace_root=workspace)
    _assert(decision["gate_outcome"] == GATE_REJECTED_INVALID, f"expected REJECTED_INVALID, got {decision['gate_outcome']}")
    _assert(any("exit_code mismatch" in item for item in decision.get("reasons", {}).get("invalid", [])), "expected exit_code mismatch invalid reason")


def scenario_provider_transient_routes_retry(workspace: Path) -> None:
    raw = "provider_error: OpenAIResponsesError status=503 service unavailable"
    decision = evaluate_ingress_gate(_brief(), raw, workspace_root=workspace)
    _assert(decision["gate_outcome"] == GATE_REJECTED_RETRY, f"expected REJECTED_RETRY, got {decision['gate_outcome']}")
    ingress = decision.get("ingress") if isinstance(decision.get("ingress"), dict) else {}
    _assert(ingress.get("classification") == INGRESS_RETRY_REQUIRED, f"expected ingress retry classification, got {ingress}")
    pf = ingress.get("provider_failure") if isinstance(ingress, dict) else None
    _assert(isinstance(pf, dict) and pf.get("retryable") is True, f"expected retryable provider metadata, got {ingress}")


def scenario_provider_nonretryable_routes_invalid(workspace: Path) -> None:
    raw = "provider_error: OpenAIResponsesError status=401 unauthorized invalid api key"
    decision = evaluate_ingress_gate(_brief(), raw, workspace_root=workspace)
    _assert(decision["gate_outcome"] == GATE_REJECTED_INVALID, f"expected REJECTED_INVALID, got {decision['gate_outcome']}")
    ingress = decision.get("ingress") if isinstance(decision.get("ingress"), dict) else {}
    _assert(ingress.get("classification") == INGRESS_INVALID, f"expected ingress invalid classification, got {ingress}")
    pf = ingress.get("provider_failure") if isinstance(ingress, dict) else None
    _assert(isinstance(pf, dict) and pf.get("retryable") is False, f"expected non-retryable provider metadata, got {ingress}")


SCENARIOS: list[tuple[str, Callable[[Path], None]]] = [
    ("accepts_valid_contract_packet", scenario_accepts_valid_contract_packet),
    ("rejects_env_probe_ingress_invalid", scenario_rejects_env_probe_ingress_invalid),
    ("rejects_partial_packet_retry", scenario_rejects_partial_packet_retry),
    ("rejects_required_path_mismatch_invalid", scenario_rejects_required_path_mismatch_invalid),
    ("rejects_success_exit_code_mismatch_invalid", scenario_rejects_success_exit_code_mismatch_invalid),
    ("provider_transient_routes_retry", scenario_provider_transient_routes_retry),
    ("provider_nonretryable_routes_invalid", scenario_provider_nonretryable_routes_invalid),
]


def main() -> int:
    results: list[dict[str, Any]] = []
    failed = 0

    with tempfile.TemporaryDirectory(prefix="delegation_contract_regressions_") as td:
        workspace = Path(td)
        for name, fn in SCENARIOS:
            try:
                fn(workspace)
                results.append({"name": name, "ok": True})
                print(f"PASS {name}")
            except Exception as exc:
                failed += 1
                results.append({"name": name, "ok": False, "error": str(exc)})
                print(f"FAIL {name}: {exc}")

    summary = {
        "ok": failed == 0,
        "total": len(results),
        "failed": failed,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
