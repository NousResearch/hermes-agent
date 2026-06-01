#!/usr/bin/env python3
"""Classify PR risk/complexity and required CI/review lanes.

This script is intentionally deterministic and dependency-free so it can run as
an early GitHub Actions gate before expensive jobs. It reads changed paths and a
PR body, emits ci-classification.json / ci-classification.md, and exits non-zero
when a PR body or lane declaration is weaker than the classifier requires.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

RISK_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5"]
COMPLEXITY_ORDER = ["C0", "C1", "C2", "C3", "C4", "C5"]

RISK_TO_CI = {
    "R0": {"pr_body_contract", "diff_check"},
    "R1": {"pr_body_contract", "diff_check", "docs_impact"},
    "R2": {
        "pr_body_contract",
        "diff_check",
        "docs_impact",
        "typecheck",
        "targeted_runtime_tests",
        "backend_runtime",
        "build",
    },
    "R3": {
        "pr_body_contract",
        "diff_check",
        "docs_impact",
        "typecheck",
        "targeted_runtime_tests",
        "backend_runtime",
        "build",
        "governance_required",
    },
    "R4": {
        "pr_body_contract",
        "diff_check",
        "docs_impact",
        "typecheck",
        "targeted_runtime_tests",
        "backend_runtime",
        "build",
        "governance_required",
        "security_required",
        "protected_claim_gate",
        "human_gate_required",
        "privacy_data_gate",
        "rollback_proof",
        "audit_log_validation",
    },
    "R5": {
        "pr_body_contract",
        "diff_check",
        "docs_impact",
        "typecheck",
        "targeted_runtime_tests",
        "backend_runtime",
        "build",
        "governance_required",
        "security_required",
        "protected_claim_gate",
        "human_gate_required",
        "privacy_data_gate",
        "rollback_proof",
        "audit_log_validation",
        "production_readiness_gate",
        "incident_response_check",
        "support_readiness_check",
        "marketing_claims_check",
        "data_deletion_export_check",
        "monitoring_observability_check",
    },
}

SURFACE_TO_CI = {
    "docs_only": {"pr_body_contract", "diff_check", "docs_impact"},
    "receipt_only": {"pr_body_contract", "diff_check", "docs_impact"},
    "test_only": {"pr_body_contract", "diff_check", "targeted_tests"},
    "runtime_backend": {"typecheck", "targeted_runtime_tests", "backend_runtime", "build"},
    "runtime_frontend": {"typecheck", "ui_quality", "targeted_frontend_tests", "build"},
    "api_route": {"backend_runtime", "route_smoke_tests", "contract_tests", "non_claim_tests"},
    "reducer": {"backend_runtime", "reducer_tests", "fixture_provenance_finality_tests"},
    "storage": {"backend_runtime", "storage_conformance_tests", "rollback_tests"},
    "sql_database": {"migration_lint", "dry_run", "rollback_proof", "db_admission_gate", "protected_human_review"},
    "roadmap_os": {"validate_roadmap_os", "validate_pr_roadmap_delta", "reduce_roadmap_events", "check_reducer_parity", "generate_roadmap_views_check"},
    "accepted_event": {"validate_roadmap_os", "authorized_reducer_review"},
    "candidate_event": {"event_schema_validation", "no_accepted_state_mutation_check"},
    "prd_packet": {"validate_prd_packets", "runtimePayloadContract_validation", "blocker_exemption_validation"},
    "conductor_dispatch": {"backend_runtime", "conductor_tests", "executive_gate_tests", "governance_required"},
    "hermes_sidecar": {"hermes_sidecar_tests", "sidecar_authority_policy_tests", "no_merge_evidence_guard"},
    "ci_workflow": {"workflow_lint", "pr_impact_classifier_tests", "governance_required"},
    "security_permissions": {"security_required", "permissions_tests", "protected_human_review"},
    "auth_identity": {"auth_tests", "session_isolation_tests", "security_required"},
    "secrets_tokens": {"secret_scan", "token_storage_tests", "encryption_tests", "protected_human_review"},
    "live_connector": {"connector_sandbox_tests", "webhook_tests", "token_vault_tests", "consent_audit_tests", "protected_human_review"},
    "webhook": {"signature_validation_tests", "replay_idempotency_tests", "failure_rollback_tests"},
    "pii_customer_data": {"privacy_data_gate", "deletion_export_tests", "audit_log_tests", "incident_response_check"},
    "finance_interpretation": {"finance_sensitive_review", "finality_non_claim_tests", "no_advice_claim_guard"},
    "tax_accounting": {"protected_human_review", "tax_accounting_non_claim_guard", "finality_tests"},
    "money_movement": {"human_gate_required", "protected_claim_gate"},
    "launch_production": {"production_readiness_gate", "monitoring_observability_check", "rollback_check", "incident_response_check"},
    "support_customer_ops": {"support_readiness_check", "customer_safe_language_check", "escalation_policy_check"},
    "marketing_claims": {"marketing_claims_check", "regulated_claim_guard"},
    "mac_app": {"mac_app_check", "typecheck"},
    "gbrain_gstack": {"gbrain_gstack_contract_tests", "read_only_write_back_gate"},
    "product_intelligence": {"rubric_tests", "evaluation_score_drift_tests"},
}

PROTECTED_SURFACES = {
    "sql_database",
    "security_permissions",
    "auth_identity",
    "secrets_tokens",
    "live_connector",
    "webhook",
    "pii_customer_data",
    "finance_interpretation",
    "tax_accounting",
    "money_movement",
    "launch_production",
}

RUNTIME_SURFACES = {
    "runtime_backend",
    "runtime_frontend",
    "api_route",
    "reducer",
    "storage",
    "sql_database",
    "conductor_dispatch",
    "hermes_sidecar",
    "ci_workflow",
    "auth_identity",
    "secrets_tokens",
    "live_connector",
    "webhook",
}

CLASSIFICATION_HEADER = "Risk, Complexity, Review, and CI Classification"
RUNTIME_PAYLOAD_CONTRACT_FIELDS = {
    "user_or_operator_visible_outcome",
    "runtime_surface_touched",
    "product_or_platform_capability_advanced",
    "why_this_is_not_only_docs_or_scaffolding",
    "tests_that_prove_runtime_behavior",
    "acceptance_gate",
    "rollback",
    "protected_non_claims",
}


def _max(values: Iterable[str], order: list[str]) -> str:
    return max(values, key=order.index)


def _bump(value: str, order: list[str], steps: int = 1) -> str:
    return order[min(order.index(value) + steps, len(order) - 1)]


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def changed_files_from_git(base: str | None, head: str | None) -> list[str]:
    if base and head:
        diff_range = f"{base}...{head}"
    else:
        diff_range = os.getenv("GITHUB_BASE_REF") or "origin/main...HEAD"
    out = _run_git(["diff", "--name-only", diff_range])
    return [line for line in out.splitlines() if line]


def additions_from_git(base: str | None, head: str | None) -> int:
    try:
        diff_range = f"{base}...{head}" if base and head else (os.getenv("GITHUB_BASE_REF") or "origin/main...HEAD")
        out = _run_git(["diff", "--numstat", diff_range])
    except Exception:
        return 0
    total = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if parts and parts[0].isdigit():
            total += int(parts[0])
    return total


def read_body(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).exists():
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        return (event.get("pull_request") or {}).get("body") or ""
    return ""


def parse_declared_field(body: str, label: str) -> str | None:
    match = re.search(rf"^-\s*{re.escape(label)}\s*:\s*(.+?)\s*$", body, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value.lower() in {"", "n/a", "na", "todo", "tbd"} else value


def parse_yes_no(body: str, label: str) -> str | None:
    value = parse_declared_field(body, label)
    if value is None:
        return None
    lowered = value.lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return value


def split_lanes(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().strip("`") for item in re.split(r"[,\n]", value) if item.strip()}


def parse_runtime_payload_contract(body: str) -> dict[str, str] | None:
    """Parse the required Markdown RuntimePayloadContract section."""

    lines = body.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^##\s+RuntimePayloadContract\s*$", line):
            fields: dict[str, str] = {}
            for contract_line in lines[index + 1 :]:
                if re.match(r"^##\s+", contract_line):
                    break
                match = re.match(r"^\s*-\s*([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$", contract_line)
                if match:
                    fields[match.group(1)] = match.group(2).strip()
            return fields
    return None


def missing_runtime_payload_contract_fields(fields: dict[str, str] | None) -> list[str]:
    if fields is None:
        return sorted(RUNTIME_PAYLOAD_CONTRACT_FIELDS)
    return sorted(field for field in RUNTIME_PAYLOAD_CONTRACT_FIELDS if not fields.get(field))


def infer_surfaces(files: list[str], body: str) -> set[str]:
    surfaces: set[str] = set()
    if not files:
        return {"receipt_only"}

    docs_like = True
    tests_like = True
    receipt_like = True
    for file in files:
        p = Path(file)
        suffix = p.suffix.lower()
        parts = set(p.parts)
        docs_like = docs_like and (suffix in {".md", ".rst", ".txt"} or "docs" in parts or file == ".github/PULL_REQUEST_TEMPLATE.md")
        tests_like = tests_like and ("tests" in parts or p.name.startswith("test_"))
        receipt_like = receipt_like and ("receipt" in p.name.lower() or "receipts" in parts)

        if file.startswith(".github/workflows/"):
            surfaces.add("ci_workflow")
        if file.startswith("gateway/") or "dispatch" in file or "conductor" in file:
            surfaces.add("conductor_dispatch")
        if file.startswith(("agent/", "tools/", "hermes_cli/", "cron/", "tui_gateway/", "plugins/", "gateway/")):
            surfaces.add("runtime_backend")
        if file.startswith(("ui-tui/", "website/src/", "website/web/")) or suffix in {".tsx", ".jsx", ".css"}:
            surfaces.add("runtime_frontend")
        if "route" in file or "api" in parts:
            surfaces.add("api_route")
        if "reducer" in file:
            surfaces.add("reducer")
        if "storage" in file or "database" in file:
            surfaces.add("storage")
        if suffix == ".sql" or "migrations" in parts:
            surfaces.add("sql_database")
        if "roadmap" in file.lower():
            surfaces.add("roadmap_os")
        if "accepted" in file.lower() and "event" in file.lower():
            surfaces.add("accepted_event")
        if "candidate" in file.lower() and "event" in file.lower():
            surfaces.add("candidate_event")
        if "prd" in file.lower() or "packet" in file.lower():
            surfaces.add("prd_packet")
        if "hermes" in file.lower() and ("sidecar" in file.lower() or "watchdog" in file.lower()):
            surfaces.add("hermes_sidecar")
        if any(term in file.lower() for term in ("auth", "permission", "oauth", "token", "secret")):
            surfaces.add("security_permissions")
        if "oauth" in file.lower() or "identity" in file.lower() or "session" in file.lower():
            surfaces.add("auth_identity")
        if "token" in file.lower() or "secret" in file.lower() or "vault" in file.lower():
            surfaces.add("secrets_tokens")
        if any(term in file.lower() for term in ("connector", "plaid", "bank")):
            surfaces.add("live_connector")
        if "webhook" in file.lower():
            surfaces.add("webhook")
        if any(term in file.lower() for term in ("pii", "customer_data", "privacy", "delete", "export")):
            surfaces.add("pii_customer_data")
        if any(term in file.lower() for term in ("finance", "ledger", "finality", "portfolio", "heloc", "loan")):
            surfaces.add("finance_interpretation")
        if any(term in file.lower() for term in ("tax", "accounting")):
            surfaces.add("tax_accounting")
        if "money_movement" in file.lower() or "payment" in file.lower():
            surfaces.add("money_movement")
        if any(term in file.lower() for term in ("production", "deploy", "launch")):
            surfaces.add("launch_production")
        if "support" in file.lower():
            surfaces.add("support_customer_ops")
        if "marketing" in file.lower() or "landing" in file.lower():
            surfaces.add("marketing_claims")
        if "mac" in file.lower() or file.endswith(".app"):
            surfaces.add("mac_app")
        if any(term in file.lower() for term in ("gbrain", "gstack")):
            surfaces.add("gbrain_gstack")

    if tests_like:
        surfaces.add("test_only")
    if docs_like:
        surfaces.add("docs_only")
    if receipt_like:
        surfaces.add("receipt_only")

    body_lower = body.lower()
    for surface in SURFACE_TO_CI:
        if surface.replace("_", " ") in body_lower or surface in body_lower:
            surfaces.add(surface)
    return surfaces or {"docs_only"}


def infer_risk(surfaces: set[str]) -> str:
    if "money_movement" in surfaces or "launch_production" in surfaces:
        return "R5"
    if surfaces & PROTECTED_SURFACES:
        return "R4"
    if surfaces & {"conductor_dispatch", "hermes_sidecar", "ci_workflow", "roadmap_os", "accepted_event", "support_customer_ops", "marketing_claims"}:
        return "R3"
    if surfaces & RUNTIME_SURFACES or "prd_packet" in surfaces:
        return "R2"
    if surfaces & {"test_only", "candidate_event"}:
        return "R1"
    return "R0"


def infer_complexity(files: list[str], additions: int, surfaces: set[str]) -> str:
    file_count = len(files)
    if surfaces & {"money_movement", "launch_production"}:
        return "C5"
    if file_count <= 2 and additions <= 100 and not (surfaces & RUNTIME_SURFACES):
        complexity = "C0"
    elif file_count <= 5 and additions <= 250:
        complexity = "C1"
    elif file_count <= 10 and additions <= 750:
        complexity = "C2"
    elif file_count <= 20 and additions <= 1500:
        complexity = "C3"
    else:
        complexity = "C4"

    # C5 is reserved for exceptional protected launch/production/security/data/
    # token/SQL/financial authority. Ordinary upgrade triggers may make a PR
    # large, but should not manufacture protected-review semantics by themselves.
    def bump_non_protected(value: str) -> str:
        return COMPLEXITY_ORDER[min(COMPLEXITY_ORDER.index(value) + 1, COMPLEXITY_ORDER.index("C4"))]

    if "ci_workflow" in surfaces:
        complexity = bump_non_protected(complexity)
    if len({s for s in surfaces if s not in {"docs_only", "test_only", "receipt_only"}}) > 1:
        complexity = bump_non_protected(complexity)
    return complexity


def required_reviews(risk: str, complexity: str, surfaces: set[str]) -> set[str]:
    reviews: set[str] = set()
    if risk in {"R2", "R3", "R4", "R5"} or complexity in {"C3", "C4", "C5"}:
        reviews.add("secondary_review_required")
    if risk in {"R3", "R4", "R5"} or complexity in {"C3", "C4", "C5"} or surfaces & RUNTIME_SURFACES:
        reviews.add("adversarial_review_required")
    if risk in {"R4", "R5"} or complexity in {"C3", "C4", "C5"}:
        reviews.add("opposite_provider_adversarial_required")
    if risk in {"R4", "R5"} or surfaces & PROTECTED_SURFACES:
        reviews.add("protected_human_review_required")
    if risk == "R5" or surfaces & {"launch_production", "marketing_claims"}:
        reviews.add("founder_review_required")
    if surfaces & {"security_permissions", "auth_identity", "secrets_tokens", "live_connector", "webhook", "pii_customer_data"}:
        reviews.add("security_review_required")
    if surfaces & {"finance_interpretation", "tax_accounting", "money_movement"}:
        reviews.add("finance_sensitive_review_required")
    if not reviews:
        reviews.add("no_secondary_review_required")
    return reviews


def token_class(risk: str, complexity: str) -> str:
    if risk == "R5" or complexity == "C5":
        return "XL"
    if risk == "R4" or complexity in {"C3", "C4"}:
        return "L"
    if risk in {"R2", "R3"} or complexity == "C2":
        return "M"
    return "S"


@dataclass
class Classification:
    pr_number: str = "unknown"
    repo: str = "unknown"
    title: str = "unknown"
    risk_class: str = "R0"
    complexity_class: str = "C0"
    impacted_surfaces: list[str] = field(default_factory=list)
    protected_flags: list[str] = field(default_factory=list)
    runtime_payload_contract_present: bool = False
    blocker_exemption_present: bool = False
    required_ci_lanes: list[str] = field(default_factory=list)
    optional_ci_lanes: list[str] = field(default_factory=list)
    skipped_ci_lanes: list[str] = field(default_factory=list)
    required_reviews: list[str] = field(default_factory=list)
    secondary_review_required: bool = False
    adversarial_review_required: bool = False
    opposite_provider_required: bool = False
    human_gate_required: bool = False
    founder_review_required: bool = False
    reason: str = ""
    token_class: str = "S"
    merge_blocking_conditions: list[str] = field(default_factory=list)
    allowed_to_mark_ready: bool = True

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def classify(files: list[str], body: str, additions: int = 0, pr_number: str = "unknown", repo: str = "unknown", title: str = "unknown") -> Classification:
    surfaces = infer_surfaces(files, body)
    inferred_risk = infer_risk(surfaces)
    inferred_complexity = infer_complexity(files, additions, surfaces)

    declared_risk = parse_declared_field(body, "Risk class")
    declared_complexity = parse_declared_field(body, "Complexity class")
    risk = _max([inferred_risk, declared_risk if declared_risk in RISK_ORDER else inferred_risk], RISK_ORDER)
    complexity = _max([inferred_complexity, declared_complexity if declared_complexity in COMPLEXITY_ORDER else inferred_complexity], COMPLEXITY_ORDER)

    lanes = set(RISK_TO_CI[risk])
    for surface in surfaces:
        lanes |= SURFACE_TO_CI.get(surface, set())
    lanes.add("pr_body_contract")
    lanes.add("diff_check")

    reviews = required_reviews(risk, complexity, surfaces)
    runtime_contract_fields = parse_runtime_payload_contract(body)
    missing_contract_fields = missing_runtime_payload_contract_fields(runtime_contract_fields)
    runtime_contract_present = (
        parse_yes_no(body, "RuntimePayloadContract present") == "yes"
        and runtime_contract_fields is not None
        and not missing_contract_fields
    )
    blocker_exemption = parse_declared_field(body, "Blocker exemption, if any") is not None

    blocking: list[str] = []
    if CLASSIFICATION_HEADER.lower() not in body.lower():
        blocking.append("missing_required_pr_body_classification_section")
    if declared_risk not in RISK_ORDER:
        blocking.append("missing_or_invalid_declared_risk_class")
    if declared_complexity not in COMPLEXITY_ORDER:
        blocking.append("missing_or_invalid_declared_complexity_class")
    if surfaces & RUNTIME_SURFACES and not runtime_contract_present:
        blocking.append("runtime_surface_without_runtimePayloadContract")
        if runtime_contract_fields is not None and missing_contract_fields:
            blocking.append("runtimePayloadContract_missing_required_fields:" + ",".join(missing_contract_fields))
    if reviews - {"no_secondary_review_required"} and parse_declared_field(body, "Expected state change") is None:
        blocking.append("missing_expected_state_change")
    if surfaces & PROTECTED_SURFACES and not (runtime_contract_fields or {}).get("protected_non_claims") and "Protected non-claims" not in body:
        blocking.append("protected_surface_without_protected_non_claims")

    declared_lanes_value = parse_declared_field(body, "Required CI lanes")
    declared_lanes = split_lanes(declared_lanes_value)
    if not declared_lanes:
        blocking.append("missing_required_ci_lanes")
    else:
        missing_lanes = sorted(lanes - declared_lanes)
        if missing_lanes:
            blocking.append("declared_ci_lanes_weaker_than_classifier:" + ",".join(missing_lanes))

    return Classification(
        pr_number=pr_number,
        repo=repo,
        title=title,
        risk_class=risk,
        complexity_class=complexity,
        impacted_surfaces=sorted(surfaces),
        protected_flags=sorted(surfaces & PROTECTED_SURFACES),
        runtime_payload_contract_present=runtime_contract_present,
        blocker_exemption_present=blocker_exemption,
        required_ci_lanes=sorted(lanes),
        optional_ci_lanes=[],
        skipped_ci_lanes=sorted({lane for r, risk_lanes in RISK_TO_CI.items() if RISK_ORDER.index(r) > RISK_ORDER.index(risk) for lane in risk_lanes} - lanes),
        required_reviews=sorted(reviews),
        secondary_review_required="secondary_review_required" in reviews,
        adversarial_review_required="adversarial_review_required" in reviews,
        opposite_provider_required="opposite_provider_adversarial_required" in reviews,
        human_gate_required=bool(reviews & {"protected_human_review_required", "security_review_required", "finance_sensitive_review_required"}),
        founder_review_required="founder_review_required" in reviews,
        reason=f"Inferred from {len(files)} changed file(s), {additions} addition(s), surfaces: {', '.join(sorted(surfaces))}.",
        token_class=token_class(risk, complexity),
        merge_blocking_conditions=blocking,
        allowed_to_mark_ready=not blocking,
    )


def markdown_report(data: dict[str, object]) -> str:
    lines = ["# CI Classification", ""]
    for key, value in data.items():
        if isinstance(value, list):
            rendered = ", ".join(str(v) for v in value) if value else "none"
        else:
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"- {key}: {rendered}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--pr-body")
    parser.add_argument("--additions", type=int)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--pr-number", default=os.getenv("PR_NUMBER", "unknown"))
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "unknown"))
    parser.add_argument("--title", default=os.getenv("PR_TITLE", "unknown"))
    args = parser.parse_args(argv)

    files = args.changed_file or changed_files_from_git(args.base, args.head)
    body = read_body(args.pr_body)
    additions = args.additions if args.additions is not None else additions_from_git(args.base, args.head)
    classification = classify(files, body, additions, args.pr_number, args.repo, args.title).as_dict()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ci-classification.json").write_text(json.dumps(classification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "ci-classification.md").write_text(markdown_report(classification), encoding="utf-8")
    print(json.dumps(classification, indent=2, sort_keys=True))
    return 1 if classification["merge_blocking_conditions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
