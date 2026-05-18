#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

try:
    from scripts.hermes_pm.development_workstream_packet import (
        DEFAULT_DEVELOPMENT_CANDIDATE_SPECS,
        DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_development_workstream_packet,
        redact_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from development_workstream_packet import (  # type: ignore[no-redef]
        DEFAULT_DEVELOPMENT_CANDIDATE_SPECS,
        DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_development_workstream_packet,
        redact_text,
    )


DEVELOPMENT_SLICE_PACKET_SCHEMA_VERSION = (
    "hermes.pm.development_slice_packet.v1"
)

DEFAULT_FORBIDDEN_PATHS = [
    ".env",
    ".env.*",
    ".gitea/workflows/",
    "data/",
    "logs/",
    "launchd/",
    "app/adapters/broker/",
    "app/daemon.py",
    "scripts/run_trading_worker.py",
    "scripts/run_research_worker.py",
    "scripts/run_etl_worker.py",
    "scripts/run_orchestrator_worker.py",
    "scripts/hermes_operator/branch_local_writer.py",
    "scripts/hermes_operator/apply_approved_write_plan.py",
    "scripts/hermes_pm/execute_forge_issue_create.py",
]

SLICE_SPECS: dict[str, dict[str, Any]] = {
    "dev11r-001": {
        "slice_id": "slice-dev11r-001-validation-baseline-hygiene",
        "objective": (
            "Remove the known AutoResearch validation baseline blocker while "
            "preserving the guardrail that prevents executable workflow, "
            "runtime, broker, trading, secret, or deploy behavior."
        ),
        "allowed_paths": [
            "app/services/autoresearch_provider_guarantee_service.py",
            "tests/autoresearch_*",
            "scripts/validation/validate-autoresearch-trials.sh",
            "docs/implementation/hermes_pm_checkpoint_12a_plan.md",
            "docs/architecture/hermes_operator_repo_intelligence.md",
        ],
        "forbidden_paths": DEFAULT_FORBIDDEN_PATHS,
        "expected_changes": [
            (
                "Inspect the exact validation failure without executing "
                "workflows or runtime services."
            ),
            (
                "Adjust the guarded advisory text or validator-facing evidence "
                "so the safety rule remains strict without false-positive "
                "baseline failure."
            ),
            (
                "Add focused regression coverage for the accepted "
                "advisory/evidence shape."
            ),
            "Document before/after evidence in the PM-12A checkpoint plan.",
        ],
        "required_tests": [
            "python3 -m pytest -q tests/hermes_pm",
            (
                "python3 -m pytest -q "
                "tests/hermes_operator/test_operator_policy.py "
                "tests/hermes_operator/test_gitea_ci_evidence_contracts.py"
            ),
            "bash scripts/validation/validate-governance-baseline.sh",
            "bash scripts/validation/validate-secrets-discipline.sh",
            (
                "ruff check scripts/hermes_pm scripts/hermes_operator "
                "tests/hermes_pm tests/hermes_operator"
            ),
            "git diff --check",
        ],
        "authority_required": "branch_write",
        "approval_required": True,
        "branch_name_suggestion": (
            "codex/hermes-pm-checkpoint-12a-validation-baseline-hygiene"
        ),
        "rollback_notes": (
            "Revert only the PM-12A branch-scoped diff if the validation "
            "baseline or regression tests fail. Do not alter runtime state, "
            "Gitea issues, workflows, runners, broker paths, or secrets."
        ),
        "definition_of_done": [
            (
                "The known AutoResearch validation baseline failure is either "
                "fixed or converted into a precise documented blocker with a "
                "passing safety check."
            ),
            (
                "Regression tests prove command-like advisory material cannot "
                "become execution authority."
            ),
            "Required PM/operator tests and governance/secret validations pass.",
            (
                "No Gitea writes, workflow runs, runner starts, runtime "
                "actions, deploys, financial actions, secret access, "
                "branch-writer invocation, or issue-executor invocation occur."
            ),
        ],
        "next_codex_prompt_summary": (
            "Implement PM-12A for candidate dev11r-001 only after explicit "
            "Operator branch-write approval. Keep the work bounded to the "
            "validation baseline hygiene paths, run the required checks, and "
            "do not touch broker/trading/runtime/secrets/workflows/runners."
        ),
        "next_hermes_prompt_summary": (
            "Ask the Operator to approve PM-12A branch-write authority for "
            "dev11r-001. Explain the allowed paths, forbidden surfaces, tests, "
            "rollback notes, and proof that no write or runtime action is "
            "authorized until approval is granted."
        ),
    }
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _all_candidates(workstream_packet: dict[str, Any] | None) -> list[dict[str, Any]]:
    if isinstance(workstream_packet, dict):
        candidates = workstream_packet.get("development_candidates")
        if isinstance(candidates, list):
            return [item for item in candidates if isinstance(item, dict)]
    return [{**candidate} for candidate in DEFAULT_DEVELOPMENT_CANDIDATE_SPECS]


def _recommended_candidate_id(workstream_packet: dict[str, Any] | None) -> str:
    if isinstance(workstream_packet, dict):
        recommended = workstream_packet.get("recommended_first_development_slice")
        if isinstance(recommended, dict) and recommended.get("candidate_id"):
            return str(recommended["candidate_id"])
    fallback = build_development_workstream_packet()
    recommended = fallback.get("recommended_first_development_slice") or {}
    return str(recommended.get("candidate_id") or "dev11r-001")


def _find_candidate(
    candidates: list[dict[str, Any]],
    candidate_id: str,
) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise ValueError(f"Unknown development candidate ID: {candidate_id}")


def _default_slice_spec(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "unknown")
    blocked = bool(candidate.get("blocked"))
    return {
        "slice_id": f"slice-{candidate_id}",
        "objective": candidate.get("rationale") or "Review selected candidate.",
        "allowed_paths": [],
        "forbidden_paths": DEFAULT_FORBIDDEN_PATHS,
        "expected_changes": [
            (
                "Do not implement this candidate until a future packet defines "
                "exact allowed paths."
            )
        ],
        "required_tests": candidate.get("required_tests_or_checks") or [],
        "authority_required": candidate.get("required_authority_class") or "propose",
        "approval_required": True,
        "branch_name_suggestion": "",
        "rollback_notes": (
            "No rollback is available because this packet is non-mutating."
        ),
        "definition_of_done": [
            (
                "Candidate remains blocked and unimplemented."
                if blocked
                else "A future checkpoint defines exact implementation scope."
            )
        ],
        "next_codex_prompt_summary": (
            "Prepare an exact future implementation scope before any write."
        ),
        "next_hermes_prompt_summary": (
            "Ask the Operator whether this candidate should receive a future "
            "approval-gated slice packet."
        ),
    }


def build_development_slice_packet(
    *,
    workstream_packet: dict[str, Any] | None = None,
    candidate_id: str | None = None,
    project_id: str = "crypto_bot",
    created_at: str | None = None,
) -> dict[str, Any]:
    candidates = _all_candidates(workstream_packet)
    resolved_candidate_id = candidate_id or _recommended_candidate_id(
        workstream_packet
    )
    candidate = _find_candidate(candidates, resolved_candidate_id)
    spec = {
        **_default_slice_spec(candidate),
        **SLICE_SPECS.get(resolved_candidate_id, {}),
    }
    seed = {
        "project_id": project_id,
        "candidate_id": resolved_candidate_id,
        "slice_id": spec["slice_id"],
        "workstream_schema": (
            workstream_packet.get("schema_version")
            if isinstance(workstream_packet, dict)
            else DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION
        ),
    }
    packet = {
        "schema_version": DEVELOPMENT_SLICE_PACKET_SCHEMA_VERSION,
        "slice_id": spec["slice_id"],
        "packet_id": f"development-slice-{_sha256_payload(seed)[:16]}",
        "created_at": created_at or _utc_now(),
        "project_id": project_id,
        "candidate_id": resolved_candidate_id,
        "title": candidate.get("title"),
        "objective": spec["objective"],
        "allowed_paths": spec["allowed_paths"],
        "forbidden_paths": spec["forbidden_paths"],
        "expected_changes": spec["expected_changes"],
        "required_tests": spec["required_tests"],
        "authority_required": spec["authority_required"],
        "approval_required": bool(spec["approval_required"]),
        "branch_name_suggestion": spec["branch_name_suggestion"],
        "rollback_notes": spec["rollback_notes"],
        "definition_of_done": spec["definition_of_done"],
        "next_codex_prompt_summary": spec["next_codex_prompt_summary"],
        "next_hermes_prompt_summary": spec["next_hermes_prompt_summary"],
        "candidate_blocked": bool(candidate.get("blocked")),
        "candidate_risk_level": candidate.get("risk_level"),
        "source_workstream_packet_id": (
            workstream_packet.get("packet_id")
            if isinstance(workstream_packet, dict)
            else None
        ),
        "calls_gitea_write_api": False,
        "writes_files": False,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
    return _redact(packet)


def format_development_slice_text(packet: dict[str, Any]) -> str:
    lines = [
        "Hermes PM development slice packet",
        f"candidate: {packet.get('candidate_id')} - {packet.get('title')}",
        f"slice: {packet.get('slice_id')}",
        f"authority_required: {packet.get('authority_required')}",
        f"approval_required: {str(packet.get('approval_required')).lower()}",
        f"allowed_paths: {len(packet.get('allowed_paths') or [])}",
        f"forbidden_paths: {len(packet.get('forbidden_paths') or [])}",
        f"definition_of_done_items: {len(packet.get('definition_of_done') or [])}",
        "writes_files: false",
        "gitea_write_api: false",
    ]
    return redact_text("\n".join(lines))
