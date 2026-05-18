#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.issue_lifecycle_status import (
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )


DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION = (
    "hermes.pm.development_workstream_packet.v1"
)

FROZEN_PM11_PAYLOAD_SHA256 = (
    "4f338e0bc2ed24b9e4c46ac6327618010d5925552ba9f10cf346e1d435d66887"
)

DIRECT_COMPLETION_CATEGORIES = {
    "product_completion",
    "safety_hardening",
    "ci_evidence",
    "simulation_paper",
}

PM_PROCESS_CATEGORIES = {
    "pm_platform",
    "docs_governance",
}

FORBIDDEN_SURFACES = [
    "broker",
    "trading",
    "live-market",
    "account",
    "order",
    "position",
    "wallet",
    "exchange",
    "financial",
    "runtime service control",
    "daemon",
    "worker startup",
    "scheduler",
    "launchd",
    "workflow execution",
    "runner start",
    "deploy",
    "secret",
    "token",
    "credential",
    "keychain",
]

NON_ACTION_BOOLEANS = {
    "writes_files": False,
    "creates_issues": False,
    "creates_labels": False,
    "creates_comments": False,
    "mutates_projects": False,
    "starts_workflows": False,
    "starts_runners": False,
    "branch_writer_invoked": False,
    "issue_executor_invoked": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
}

SECRET_RE = re.compile(
    r"(?i)((?:token|secret|password|passwd|api[_-]?key|private[_ -]?key|"
    r"credential|authorization|cookie)\s*[:=]\s*)[^\s,;\"']+"
)

SAFE_DOC_PATHS = (
    "docs/architecture/hermes_operator_repo_intelligence.md",
    "docs/implementation/hermes_operator_checklist.md",
    "docs/pm/hermes_pm_backlog_selection.md",
    "docs/pm/hermes_pm_selected_candidate_execution_payload.md",
    "docs/pm/hermes_telegram_pm_startup_prompt.md",
)

DEFAULT_DEVELOPMENT_CANDIDATE_SPECS: list[dict[str, Any]] = [
    {
        "candidate_id": "dev11r-001",
        "title": "Reconcile AutoResearch validation baseline failure",
        "category": "ci_evidence",
        "rationale": (
            "Repo intelligence records a current global validation blocker: "
            "the Python quality validator stops in the AutoResearch trial "
            "governance check because guarded command references are detected "
            "as unsafe execution intent. A bounded hygiene slice can preserve "
            "the safety rule while making validation evidence usable again."
        ),
        "expected_artifacts": [
            "branch-scoped code/test diff for the failing validation boundary",
            "regression evidence that advisory command references remain blocked",
            "updated checkpoint note with before/after validation evidence",
        ],
        "required_authority_class": "branch_write",
        "recommended_next_checkpoint": (
            "HERMES PM CHECKPOINT 12A: AutoResearch validation baseline hygiene"
        ),
        "required_tests_or_checks": [
            "python3 -m pytest -q tests/hermes_pm",
            (
                "python3 -m pytest -q "
                "tests/hermes_operator/test_operator_policy.py "
                "tests/hermes_operator/test_gitea_ci_evidence_contracts.py"
            ),
            "bash scripts/validation/validate-governance-baseline.sh",
            "bash scripts/validation/validate-secrets-discipline.sh",
            "git diff --check",
        ],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "medium",
        "blocked": False,
        "blockers": [],
        "why_this_advances_completion": (
            "A project cannot be driven to completion if the canonical local "
            "validation baseline has a known blocker. Fixing this first gives "
            "Hermes an evidence-producing, non-broker development slice."
        ),
        "approval_required": True,
        "non_executable": True,
    },
    {
        "candidate_id": "dev11r-002",
        "title": "Add advisory security evidence wrappers without workflow execution",
        "category": "safety_hardening",
        "rationale": (
            "Hermes needs repeatable evidence for security and governance "
            "checks before it supervises implementation work, but this can be "
            "done through local read/proposal wrappers instead of CI runs."
        ),
        "expected_artifacts": [
            "read-only evidence wrapper for selected validation scripts",
            "tests proving wrapper output is redacted and non-mutating",
            "operator-facing summary of approval gates for CI execution",
        ],
        "required_authority_class": "branch_write",
        "recommended_next_checkpoint": (
            "HERMES PM CHECKPOINT 12B: advisory security evidence wrappers"
        ),
        "required_tests_or_checks": [
            "python3 -m pytest -q tests/hermes_pm",
            "bash scripts/validation/validate-secrets-discipline.sh",
            "git diff --check",
        ],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "low",
        "blocked": False,
        "blockers": [],
        "why_this_advances_completion": (
            "It improves the proof surface Hermes can use before approving "
            "future product changes, without running workflows or runners."
        ),
        "approval_required": True,
        "non_executable": True,
    },
    {
        "candidate_id": "dev11r-003",
        "title": "Refresh API route risk map for non-broker completion work",
        "category": "product_completion",
        "rationale": (
            "Repo intelligence records hundreds of API routes and mutating "
            "surfaces. A focused route-risk refresh can identify safe product "
            "work around read-only, signed, and simulation-only boundaries."
        ),
        "expected_artifacts": [
            "updated repo/API route risk evidence",
            "tests or fixtures for non-broker route classification",
            "completion backlog notes that separate safe routes from blocked ones",
        ],
        "required_authority_class": "branch_write",
        "recommended_next_checkpoint": (
            "HERMES PM CHECKPOINT 12C: API route risk refresh"
        ),
        "required_tests_or_checks": [
            "python3 -m pytest -q tests/hermes_operator/test_operator_policy.py",
            "bash scripts/validation/validate-governance-baseline.sh",
            "git diff --check",
        ],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "medium",
        "blocked": False,
        "blockers": [],
        "why_this_advances_completion": (
            "It helps Hermes select future coding work against safe control "
            "plane and simulation boundaries instead of broker or runtime paths."
        ),
        "approval_required": True,
        "non_executable": True,
    },
    {
        "candidate_id": "dev11r-004",
        "title": "Document and test paper/simulation-only completion boundaries",
        "category": "simulation_paper",
        "rationale": (
            "Hermes can only supervise development safely if paper and "
            "simulation boundaries are explicit, testable, and separated from "
            "live broker/account behavior."
        ),
        "expected_artifacts": [
            "paper/simulation boundary checklist",
            "focused tests around safe non-live behavior",
            "blocked-path evidence for live trading and broker mutations",
        ],
        "required_authority_class": "branch_write",
        "recommended_next_checkpoint": (
            "HERMES PM CHECKPOINT 12D: simulation boundary evidence"
        ),
        "required_tests_or_checks": [
            "python3 -m pytest -q tests/hermes_pm",
            "bash scripts/validation/validate-governance-baseline.sh",
            "git diff --check",
        ],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "medium",
        "blocked": False,
        "blockers": [],
        "why_this_advances_completion": (
            "It narrows future completion work to simulation/paper evidence "
            "while keeping live trading prohibited."
        ),
        "approval_required": True,
        "non_executable": True,
    },
    {
        "candidate_id": "dev11r-005",
        "title": "Add Hermes coding-work packet generation for approved slices",
        "category": "pm_platform",
        "rationale": (
            "The PM platform needs a deterministic way to turn a selected "
            "development candidate into an approval-gated coding packet before "
            "any branch write occurs."
        ),
        "expected_artifacts": [
            "proposal-only coding packet schema",
            "tests proving no branch writer or issue executor is registered",
            "Telegram instructions for asking Operator approval",
        ],
        "required_authority_class": "branch_write",
        "recommended_next_checkpoint": (
            "HERMES PM CHECKPOINT 12E: coding packet approval scaffolding"
        ),
        "required_tests_or_checks": [
            "python3 -m pytest -q tests/hermes_pm",
            "ruff check scripts/hermes_pm tests/hermes_pm",
            "git diff --check",
        ],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "low",
        "blocked": False,
        "blockers": [],
        "why_this_advances_completion": (
            "It is PM/process mechanics only; useful for supervision, but it "
            "does not directly improve crypto_bot product readiness."
        ),
        "approval_required": True,
        "non_executable": True,
    },
    {
        "candidate_id": "dev11r-006",
        "title": "Prepare runner smoke-trial readiness plan without starting runners",
        "category": "ci_evidence",
        "rationale": (
            "Runner evidence is eventually needed, but PM-11R must keep runner "
            "and workflow execution non-executable. This slice would prepare "
            "readiness criteria only."
        ),
        "expected_artifacts": [
            "runner smoke-trial readiness checklist",
            "approval-gate matrix for future ci_trial authority",
            "negative tests proving no runner/workflow command is exposed",
        ],
        "required_authority_class": "ci_trial",
        "recommended_next_checkpoint": (
            "HERMES PM CHECKPOINT 12F: runner smoke-trial readiness plan"
        ),
        "required_tests_or_checks": [
            "python3 -m pytest -q tests/hermes_pm",
            (
                "python3 -m pytest -q "
                "tests/hermes_operator/test_gitea_ci_evidence_contracts.py"
            ),
            "git diff --check",
        ],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "high",
        "blocked": False,
        "blockers": [
            (
                "Runner start and workflow execution require a future explicit "
                "ci_trial approval and remain non-executable in this packet."
            )
        ],
        "why_this_advances_completion": (
            "It defines the evidence gate needed before trusting CI, while "
            "still forbidding runner starts and workflow execution now."
        ),
        "approval_required": True,
        "non_executable": True,
    },
    {
        "candidate_id": "dev11r-007",
        "title": "Automate live broker/account reconciliation",
        "category": "blocked_runtime_or_secret",
        "rationale": (
            "Live broker reconciliation could be useful someday, but it would "
            "touch financial account, order, broker, and trading surfaces."
        ),
        "expected_artifacts": [],
        "required_authority_class": "financial",
        "recommended_next_checkpoint": "none: blocked in Hermes PM scope",
        "required_tests_or_checks": [],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "critical",
        "blocked": True,
        "blockers": [
            (
                "Financial, broker, account, order, position, wallet, and "
                "trading APIs are prohibited."
            ),
            "Live trading remains outside Hermes PM development authority.",
        ],
        "why_this_advances_completion": (
            "It does not advance this PM track because it is outside allowed "
            "authority and would cross into live financial behavior."
        ),
        "approval_required": True,
        "non_executable": True,
    },
    {
        "candidate_id": "dev11r-008",
        "title": "Read secrets and restart runtime deploy services",
        "category": "blocked_runtime_or_secret",
        "rationale": (
            "Secret inspection, runtime control, and deployment could alter "
            "system behavior and expose credential material."
        ),
        "expected_artifacts": [],
        "required_authority_class": "secret",
        "recommended_next_checkpoint": "none: blocked in Hermes PM scope",
        "required_tests_or_checks": [],
        "forbidden_surfaces_checked": FORBIDDEN_SURFACES,
        "risk_level": "critical",
        "blocked": True,
        "blockers": [
            "Secret access is prohibited.",
            "Runtime service control is prohibited.",
            "Deploy actions are prohibited.",
        ],
        "why_this_advances_completion": (
            "It is intentionally blocked; completion planning must proceed "
            "without secrets, runtime service control, or deployment."
        ),
        "approval_required": True,
        "non_executable": True,
    },
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def redact_text(value: str) -> str:
    return SECRET_RE.sub(r"\1<redacted>", value)


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


def load_safe_doc_summaries(repo_root: Path | None) -> list[dict[str, Any]]:
    root = (repo_root or Path.cwd()).resolve(strict=False)
    summaries: list[dict[str, Any]] = []
    for rel in SAFE_DOC_PATHS:
        path = root / rel
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        title = ""
        for line in lines[:20]:
            if line.startswith("# "):
                title = line.removeprefix("# ").strip()
                break
        summaries.append(
            {
                "path": rel,
                "title": redact_text(title or path.name),
                "line_count": len(lines),
                "source": "safe_repo_doc",
            }
        )
    return summaries


def _issue_ref(
    *,
    issue_lifecycle: dict[str, Any] | None,
    gitea_snapshot: dict[str, Any] | None,
    issue_index: int,
    expected_title: str,
) -> dict[str, Any]:
    lifecycle_ref = compact_lifecycle_summary(issue_lifecycle)
    if lifecycle_ref.get("exists"):
        return lifecycle_ref
    snapshot_ref = summarize_seed_issue_from_snapshot(
        gitea_snapshot,
        issue_index=issue_index,
        expected_title=expected_title,
    )
    if snapshot_ref.get("exists"):
        return snapshot_ref
    return {
        "issue_index": issue_index,
        "title": expected_title,
        "exists": False,
        "state": None,
        "lifecycle_state": "unknown",
        "issue_url": None,
        "source": "expected_issue",
    }


def _project_id(pm_status: dict[str, Any] | None, project_id: str | None) -> str:
    if project_id:
        return project_id
    project = (
        pm_status.get("project")
        if isinstance(pm_status, dict) and isinstance(pm_status.get("project"), dict)
        else {}
    )
    return str(project.get("project_id") or "crypto_bot")


def _git_state(pm_status: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pm_status, dict):
        return {"branch": None, "dirty": None}
    git = pm_status.get("git")
    if not isinstance(git, dict):
        return {"branch": None, "dirty": None}
    return {
        "branch": git.get("branch"),
        "dirty": git.get("dirty"),
        "latest_commit_redacted": git.get("latest_commit_redacted"),
    }


def _current_pm_state(
    *,
    pm_status: dict[str, Any] | None,
    issue_lifecycle: dict[str, Any] | None,
    gitea_snapshot: dict[str, Any] | None,
    backlog_selection_packet: dict[str, Any] | None,
    issue_index: int,
    expected_title: str,
) -> dict[str, Any]:
    selected_ids: list[str] = []
    if isinstance(backlog_selection_packet, dict):
        selected_ids = [
            str(item.get("candidate_id"))
            for item in backlog_selection_packet.get("selected_candidates") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
    if "pm8-002" not in selected_ids:
        selected_ids.append("pm8-002")
    return {
        "git": _git_state(pm_status),
        "issue_1": _issue_ref(
            issue_lifecycle=issue_lifecycle,
            gitea_snapshot=gitea_snapshot,
            issue_index=issue_index,
            expected_title=expected_title,
        ),
        "pm11b_status": "parked_not_abandoned",
        "selected_pm_candidate": {
            "candidate_id": "pm8-002",
            "state": "selected_payload_prepared_not_created",
            "stable_execution_payload_sha256": FROZEN_PM11_PAYLOAD_SHA256,
            "source_checkpoint": "PM-10B",
            "future_issue_creation_allowed_now": False,
        },
        "selected_pm_candidate_ids": sorted(dict.fromkeys(selected_ids)),
        "issue_creation_allowed_now": False,
        "branch_write_allowed_now": False,
        "runner_or_workflow_execution_allowed_now": False,
        "runtime_or_financial_action_allowed_now": False,
    }


def _preference_recommended_candidate_id(
    operator_preferences: dict[str, Any] | None,
) -> str | None:
    if not isinstance(operator_preferences, dict):
        return None
    raw = operator_preferences.get("recommended_development_candidate_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _candidate_by_id(
    candidates: list[dict[str, Any]],
    candidate_id: str,
) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return None


def _recommended_first_candidate(
    candidates: list[dict[str, Any]],
    operator_preferences: dict[str, Any] | None,
) -> dict[str, Any]:
    preferred_id = _preference_recommended_candidate_id(operator_preferences)
    if preferred_id:
        preferred = _candidate_by_id(candidates, preferred_id)
        if preferred and not preferred.get("blocked"):
            return preferred
    for candidate in candidates:
        if candidate.get("candidate_id") == "dev11r-001":
            return candidate
    return next(candidate for candidate in candidates if not candidate.get("blocked"))


def build_development_workstream_packet(
    *,
    pm_status: dict[str, Any] | None = None,
    issue_lifecycle: dict[str, Any] | None = None,
    gitea_snapshot: dict[str, Any] | None = None,
    repo_intelligence_docs: list[dict[str, Any]] | None = None,
    hermes_operator_checklist: dict[str, Any] | None = None,
    pm_backlog_candidates: dict[str, Any] | None = None,
    safe_planning_docs: list[dict[str, Any]] | None = None,
    operator_preferences: dict[str, Any] | None = None,
    project_id: str | None = None,
    issue_index: int = DEFAULT_ISSUE_INDEX,
    expected_title: str = EXPECTED_PM_SEED_ISSUE_TITLE,
    candidate_specs: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    candidates = [
        {**candidate}
        for candidate in (candidate_specs or DEFAULT_DEVELOPMENT_CANDIDATE_SPECS)
    ]
    recommended = _recommended_first_candidate(candidates, operator_preferences)
    completion_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("category") in DIRECT_COMPLETION_CATEGORIES
        and not candidate.get("blocked")
    ]
    non_pm_process_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("category") in PM_PROCESS_CATEGORIES
    ]
    blocked_candidates = [
        candidate for candidate in candidates if candidate.get("blocked")
    ]
    created = created_at or _utc_now()
    resolved_project_id = _project_id(pm_status, project_id)
    packet_seed = {
        "project_id": resolved_project_id,
        "candidate_ids": [candidate.get("candidate_id") for candidate in candidates],
        "recommended": recommended.get("candidate_id"),
        "issue_index": issue_index,
    }
    packet = {
        "schema_version": DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION,
        "packet_id": f"development-workstream-{sha256_payload(packet_seed)[:16]}",
        "created_at": created,
        "project_id": resolved_project_id,
        "source_issue_refs": [
            _issue_ref(
                issue_lifecycle=issue_lifecycle,
                gitea_snapshot=gitea_snapshot,
                issue_index=issue_index,
                expected_title=expected_title,
            )
        ],
        "current_pm_state": _current_pm_state(
            pm_status=pm_status,
            issue_lifecycle=issue_lifecycle,
            gitea_snapshot=gitea_snapshot,
            backlog_selection_packet=pm_backlog_candidates,
            issue_index=issue_index,
            expected_title=expected_title,
        ),
        "completion_objective": (
            "Move crypto_bot toward completion through small, evidence-backed, "
            "simulation-safe development slices while Hermes PM stays in "
            "read/propose/supervise mode until the Operator approves exact "
            "branch-write or forge-write authority."
        ),
        "development_candidates": candidates,
        "direct_completion_candidate_count": len(completion_candidates),
        "recommended_first_development_slice": {
            "candidate_id": recommended.get("candidate_id"),
            "title": recommended.get("title"),
            "category": recommended.get("category"),
            "required_authority_class": recommended.get(
                "required_authority_class"
            ),
            "approval_required": True,
            "blocked": bool(recommended.get("blocked")),
            "risk_level": recommended.get("risk_level"),
            "recommended_next_checkpoint": recommended.get(
                "recommended_next_checkpoint"
            ),
            "why": recommended.get("why_this_advances_completion"),
        },
        "non_pm_process_candidates": non_pm_process_candidates,
        "blocked_candidates": blocked_candidates,
        "approval_requirements": [
            (
                "Operator must approve the exact future checkpoint before any "
                "branch write."
            ),
            (
                "Future branch-write approval must name candidate dev11r-001, "
                "allowed paths, required checks, and rollback expectations."
            ),
            (
                "Forge writes, issue creation, comments, labels, projects, "
                "PRs, workflow runs, runner starts, deploys, runtime actions, "
                "financial actions, and secret access remain unapproved."
            ),
            (
                "PM-11B is parked; it must not be retried unless the Operator "
                "explicitly reselects the frozen PM-10B issue payload."
            ),
        ],
        "evidence_requirements": {
            "recommended_candidate_id": recommended.get("candidate_id"),
            "required_tests_or_checks": recommended.get("required_tests_or_checks")
            or [],
            "completion_proof": [
                "definition_of_done satisfied in the development slice packet",
                "targeted tests pass",
                "secret/governance validation passes",
                "no forbidden surfaces touched",
            ],
        },
        "future_checkpoint_recommendation": {
            "checkpoint": recommended.get("recommended_next_checkpoint"),
            "branch_name_suggestion": (
                "codex/hermes-pm-checkpoint-12a-validation-baseline-hygiene"
            ),
            "candidate_id": recommended.get("candidate_id"),
            "authority_required": recommended.get("required_authority_class"),
            "summary": (
                "Implement only the bounded validation-baseline hygiene slice "
                "after explicit Operator branch-write approval."
            ),
        },
        "telegram_next_step_summary": (
            "Hermes should say Issue #1 exists, pm8-002 is selected but not "
            "created, PM-11B is parked, and the next safe completion slice is "
            "dev11r-001 pending explicit branch-write approval."
        ),
        "source_inputs": {
            "repo_intelligence_docs": repo_intelligence_docs or [],
            "hermes_operator_checklist": hermes_operator_checklist or {},
            "pm_backlog_candidates_schema": (
                pm_backlog_candidates.get("schema_version")
                if isinstance(pm_backlog_candidates, dict)
                else None
            ),
            "safe_planning_docs": safe_planning_docs or [],
            "operator_preferences_used": bool(operator_preferences),
        },
        "calls_gitea_write_api": False,
        "mutation_executed": False,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
    return _redact(packet)


def format_development_workstream_text(packet: dict[str, Any]) -> str:
    recommended = packet.get("recommended_first_development_slice") or {}
    candidates = packet.get("development_candidates") or []
    blocked = packet.get("blocked_candidates") or []
    lines = [
        "Hermes PM development workstream packet",
        f"project: {packet.get('project_id')}",
        f"schema: {packet.get('schema_version')}",
        f"candidates: {len(candidates)}",
        (
            "direct_completion_candidates: "
            f"{packet.get('direct_completion_candidate_count')}"
        ),
        (
            "recommended_first_slice: "
            f"{recommended.get('candidate_id')} - {recommended.get('title')}"
        ),
        f"authority_required: {recommended.get('required_authority_class')}",
        f"next_checkpoint: {recommended.get('recommended_next_checkpoint')}",
        f"blocked_candidates: {len(blocked)}",
        "writes: false",
        "gitea_write_api: false",
        "runtime_or_financial_actions: false",
    ]
    return redact_text("\n".join(lines))
