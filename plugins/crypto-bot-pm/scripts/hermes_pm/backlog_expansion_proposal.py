#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any

try:
    from scripts.hermes_pm.issue_lifecycle_status import (
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )
    from scripts.hermes_pm.task_classifier import classify_task
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )
    from task_classifier import classify_task  # type: ignore[no-redef]


BACKLOG_EXPANSION_SCHEMA_VERSION = "hermes.pm.backlog_expansion_proposal.v1"

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
    "runtime",
    "daemon",
    "worker",
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

BLOCKED_TASK_CLASSES = {
    "deploy",
    "financial",
    "runtime_admin",
    "secret",
    "unknown",
}

APPROVAL_REQUIRED_TASK_CLASSES = {
    "branch_write",
    "ci_trial",
    "forge_write",
}

NON_ACTION_BOOLEANS = {
    "creates_issues": False,
    "creates_labels": False,
    "creates_comments": False,
    "mutates_projects": False,
    "starts_workflows": False,
    "starts_runners": False,
    "branch_writer_invoked": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
}

SECRET_RE = re.compile(
    r"(?i)((?:token|secret|password|passwd|api[_-]?key|private[_ -]?key|"
    r"credential|authorization|cookie)\s*[:=]\s*)[^\s,;\"']+"
)


DEFAULT_CANDIDATE_SPECS = [
    {
        "candidate_id": "pm8-001",
        "title": "Propose crypto-bot-pm plugin reliability hardening",
        "rationale": (
            "PM-7 observed transient Telegram import/module-loading issues. "
            "Hermes PM should keep the read-only plugin wrapper deterministic, "
            "redacted, and easy to rehearse."
        ),
        "source": "pm8_backlog_expansion_catalog",
        "proposed_issue_body_summary": (
            "Audit plugin command allowlists, cwd/PYTHONPATH handling, "
            "structured error output, and secret redaction without registering "
            "write executors."
        ),
        "suggested_priority": "P1",
        "suggested_authority_class": "branch_write",
        "suggested_next_step": (
            "Review as a future branch-scoped PM/platform hardening task; "
            "issue creation would require explicit forge approval."
        ),
    },
    {
        "candidate_id": "pm8-002",
        "title": "Propose issue-only PM backlog expansion model",
        "rationale": (
            "The local Gitea project/card endpoints remain unavailable, so "
            "Hermes PM needs a durable issue-only backlog model that does not "
            "depend on boards."
        ),
        "source": "pm8_backlog_expansion_catalog",
        "proposed_issue_body_summary": (
            "Define how proposal candidates map to future Gitea issues while "
            "preserving dedupe against Issue #1 and requiring exact operation "
            "IDs for any later write."
        ),
        "suggested_priority": "P1",
        "suggested_authority_class": "propose",
        "suggested_next_step": (
            "Review the model and keep candidates proposal-only until a future "
            "approved issue creation checkpoint."
        ),
    },
    {
        "candidate_id": "pm8-003",
        "title": "Propose Gitea read-only snapshot reliability checks",
        "rationale": (
            "Hermes PM status depends on live GET-only Gitea evidence for "
            "issues, PRs, checks, blockers, and workflow run counts."
        ),
        "source": "pm8_backlog_expansion_catalog",
        "proposed_issue_body_summary": (
            "Tighten read-only snapshot summaries, endpoint blocker reporting, "
            "and no-write evidence for Telegram PM status."
        ),
        "suggested_priority": "P2",
        "suggested_authority_class": "propose",
        "suggested_next_step": (
            "Review snapshot evidence gaps and keep all endpoint probes GET or "
            "HEAD only."
        ),
    },
    {
        "candidate_id": "pm8-004",
        "title": "Plan forge approval packet lifecycle for backlog issues",
        "rationale": (
            "Future backlog issue creation must be approval-gated by exact "
            "plan hash, operation IDs, endpoint previews, and post-write "
            "attestation."
        ),
        "source": "pm8_backlog_expansion_catalog",
        "proposed_issue_body_summary": (
            "Document the lifecycle from proposal packet to forge dry-run, "
            "approval packet, explicit token scope, and read-only attestation."
        ),
        "suggested_priority": "P2",
        "suggested_authority_class": "forge_write",
        "suggested_next_step": (
            "Use as a future approval-design item only; do not create issues "
            "or execute forge writes in PM-8."
        ),
    },
    {
        "candidate_id": "pm8-005",
        "title": "Propose Hermes Telegram PM cadence documentation cleanup",
        "rationale": (
            "Telegram should be able to ask for PM status, lifecycle, Kanban, "
            "forge dry-run, and backlog expansion summaries without drifting "
            "into runtime or trading work."
        ),
        "source": "pm8_backlog_expansion_catalog",
        "proposed_issue_body_summary": (
            "Refresh startup prompts, integration notes, and ADR/doc cleanup "
            "guidance for proposal-only PM operations."
        ),
        "suggested_priority": "P3",
        "suggested_authority_class": "propose",
        "suggested_next_step": (
            "Review docs-only PM cadence cleanup as a future branch-scoped "
            "task after Operator selection."
        ),
    },
]


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


def _project_id(pm_status: dict[str, Any] | None, work_state: dict[str, Any]) -> str:
    if work_state.get("project_id"):
        return str(work_state["project_id"])
    project = (
        pm_status.get("project")
        if isinstance(pm_status, dict) and isinstance(pm_status.get("project"), dict)
        else {}
    )
    return str(project.get("project_id") or "crypto_bot")


def _snapshot_issue_refs(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    issues = snapshot.get("issues")
    if not isinstance(issues, dict):
        return []
    refs: list[dict[str, Any]] = []
    for key in ("open", "recently_closed", "closed", "all"):
        for item in issues.get(key) or []:
            if not isinstance(item, dict):
                continue
            refs.append(
                {
                    "issue_index": item.get("number"),
                    "issue_url": item.get("html_url"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "lifecycle_state": (
                        "open_pm_seed_issue"
                        if item.get("number") in {1, "1"}
                        and item.get("title") == EXPECTED_PM_SEED_ISSUE_TITLE
                        and item.get("state") == "open"
                        else None
                    ),
                    "source": "gitea_snapshot",
                }
            )
    return refs


def _existing_issue_refs(
    *,
    gitea_snapshot: dict[str, Any] | None,
    issue_lifecycle: dict[str, Any] | None,
    kanban_packet: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    refs.extend(_snapshot_issue_refs(gitea_snapshot))
    seed_ref = summarize_seed_issue_from_snapshot(gitea_snapshot)
    if seed_ref.get("exists"):
        refs.append(seed_ref)
    lifecycle_ref = compact_lifecycle_summary(issue_lifecycle)
    if lifecycle_ref.get("exists"):
        refs.append(lifecycle_ref)
    if isinstance(kanban_packet, dict):
        for item in kanban_packet.get("existing_issue_refs") or []:
            if isinstance(item, dict):
                refs.append({**item, "source": item.get("source") or "kanban_packet"})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        title = str(ref.get("title") or "")
        issue_index = str(ref.get("issue_index") or "")
        if not title and not issue_index:
            continue
        key = (issue_index, title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "issue_index": ref.get("issue_index"),
                "issue_url": ref.get("issue_url"),
                "title": ref.get("title"),
                "state": ref.get("state"),
                "lifecycle_state": ref.get("lifecycle_state"),
                "source": ref.get("source") or "unknown",
            }
        )
    return unique


def _source_issue(
    refs: list[dict[str, Any]],
    *,
    issue_index: int,
    expected_title: str,
) -> dict[str, Any]:
    for ref in refs:
        if ref.get("issue_index") in {issue_index, str(issue_index)}:
            return ref
    return {
        "issue_index": issue_index,
        "issue_url": None,
        "title": expected_title,
        "state": None,
        "lifecycle_state": "unknown",
        "source": "expected_issue",
    }


def _ci_or_workflow_related(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in ("ci", "workflow", "workflows", "runner", "gitea actions")
    )


def _candidate_from_spec(
    spec: dict[str, Any],
    *,
    existing_titles: set[str],
    forbidden_surfaces: list[str],
) -> dict[str, Any]:
    title = redact_text(str(spec.get("title") or "Untitled PM backlog candidate"))
    rationale = redact_text(str(spec.get("rationale") or "PM proposal candidate."))
    body_summary = redact_text(
        str(
            spec.get("proposed_issue_body_summary")
            or "Proposal-only PM backlog candidate."
        )
    )
    suggested_class = str(spec.get("suggested_authority_class") or "propose")
    classification = classify_task({"title": title})
    classified_class = str(classification.get("task_class") or suggested_class)
    task_class = classified_class
    if classified_class not in BLOCKED_TASK_CLASSES | APPROVAL_REQUIRED_TASK_CLASSES:
        task_class = suggested_class
    blockers: list[str] = []
    if title in existing_titles:
        blockers.append("Candidate duplicates an existing Gitea issue title.")
    if task_class in BLOCKED_TASK_CLASSES:
        blockers.extend(
            str(item)
            for item in classification.get("approval_requirements") or []
        )
        if task_class == "deploy":
            blockers.append("Deploy work is outside PM-8 proposal-only scope.")
    ci_related = _ci_or_workflow_related(f"{title} {rationale} {body_summary}")
    if ci_related and task_class == "ci_trial":
        blockers.append(
            "Runner/workflow execution is not executable in PM-8; planning only."
        )
    blocked = bool(title in existing_titles or task_class in BLOCKED_TASK_CLASSES)
    approval_required = bool(
        blocked
        or spec.get("approval_required")
        or task_class in APPROVAL_REQUIRED_TASK_CLASSES
        or ci_related
    )
    return {
        "candidate_id": str(spec.get("candidate_id") or "pm-candidate"),
        "title": title,
        "rationale": rationale,
        "source": str(spec.get("source") or "backlog_expansion_proposal"),
        "proposed_issue_body_summary": body_summary,
        "suggested_priority": str(spec.get("suggested_priority") or "P2"),
        "suggested_authority_class": task_class,
        "suggested_next_step": redact_text(
            str(
                spec.get("suggested_next_step")
                or "Review with the Operator before any future write."
            )
        ),
        "approval_required": approval_required,
        "blocked": blocked,
        "blockers": sorted(dict.fromkeys(blockers)),
        "forbidden_surfaces_checked": list(forbidden_surfaces),
    }


def build_backlog_expansion_proposal(
    *,
    pm_status: dict[str, Any] | None = None,
    gitea_snapshot: dict[str, Any] | None = None,
    issue_lifecycle: dict[str, Any] | None = None,
    work_state: dict[str, Any] | None = None,
    kanban_packet: dict[str, Any] | None = None,
    local_docs_summaries: list[dict[str, Any]] | None = None,
    candidate_specs: list[dict[str, Any]] | None = None,
    project_id: str | None = None,
    issue_index: int = DEFAULT_ISSUE_INDEX,
    expected_title: str = EXPECTED_PM_SEED_ISSUE_TITLE,
    created_at: str | None = None,
) -> dict[str, Any]:
    state = work_state or {}
    existing_refs = _existing_issue_refs(
        gitea_snapshot=gitea_snapshot,
        issue_lifecycle=issue_lifecycle,
        kanban_packet=kanban_packet,
    )
    source_issue = _source_issue(
        existing_refs,
        issue_index=issue_index,
        expected_title=expected_title,
    )
    existing_titles = {
        str(ref.get("title"))
        for ref in existing_refs
        if ref.get("title")
    }
    specs = list(candidate_specs or DEFAULT_CANDIDATE_SPECS)
    proposed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for spec in specs:
        candidate = _candidate_from_spec(
            spec,
            existing_titles=existing_titles,
            forbidden_surfaces=FORBIDDEN_SURFACES,
        )
        if candidate["blocked"]:
            blocked.append(candidate)
        else:
            proposed.append(candidate)
    proposed = proposed[:5]
    created = created_at or _utc_now()
    resolved_project_id = project_id or _project_id(pm_status, state)
    proposal_id = "backlog-proposal-" + _sha256_payload(
        {
            "project_id": resolved_project_id,
            "source_issue_index": source_issue.get("issue_index"),
            "source_issue_title": source_issue.get("title") or expected_title,
            "candidate_titles": [item["title"] for item in proposed],
            "blocked_titles": [item["title"] for item in blocked],
        }
    )[:16]
    duplicate_titles = [
        item["title"]
        for item in blocked
        if "Candidate duplicates an existing Gitea issue title." in item["blockers"]
    ]
    return _redact(
        {
            "schema_version": BACKLOG_EXPANSION_SCHEMA_VERSION,
            "proposal_id": proposal_id,
            "created_at": created,
            "project_id": resolved_project_id,
            "source_issue_index": source_issue.get("issue_index") or issue_index,
            "source_issue_title": source_issue.get("title") or expected_title,
            "existing_issue_refs": existing_refs,
            "proposed_backlog_items": proposed,
            "dedupe_summary": {
                "source_issue_index": source_issue.get("issue_index") or issue_index,
                "source_issue_title": source_issue.get("title") or expected_title,
                "source_issue_exists": bool(
                    source_issue.get("state")
                    or source_issue.get("lifecycle_state")
                    in {"open_pm_seed_issue", "closed_pm_seed_issue"}
                ),
                "existing_issue_count": len(existing_refs),
                "duplicate_titles_blocked": duplicate_titles,
                "duplicates_of_issue_1_proposed": [
                    title for title in duplicate_titles if title == expected_title
                ],
                "issue_1_recreated": False,
                "proposed_candidate_count": len(proposed),
                "blocked_candidate_count": len(blocked),
            },
            "blocked_candidates": blocked,
            "approval_required_for_write": True,
            "calls_gitea_write_api": False,
            "mutation_executed": False,
            "recommended_next_action": (
                "Request no-mutation Operator review of the proposal-only "
                "backlog expansion, then select exact candidates for a future "
                "approval packet if any should become Gitea issues."
            ),
            "operator_questions": [
                "Which proposed backlog candidates should remain as PM planning notes?",
                (
                    "Which candidates, if any, should be converted into future "
                    "Gitea issue create operations with exact operation IDs?"
                ),
                (
                    "Should the next checkpoint remain proposal-only, or prepare "
                    "a scoped forge-write approval request for selected candidates?"
                ),
            ],
            "local_docs_summaries": local_docs_summaries or [],
            "non_action_booleans": dict(NON_ACTION_BOOLEANS),
        }
    )


def format_backlog_expansion_text(proposal: dict[str, Any]) -> str:
    items = (
        proposal.get("proposed_backlog_items")
        if isinstance(proposal.get("proposed_backlog_items"), list)
        else []
    )
    blocked = (
        proposal.get("blocked_candidates")
        if isinstance(proposal.get("blocked_candidates"), list)
        else []
    )
    lines = [
        "Hermes PM backlog expansion proposal",
        f"Project: {proposal.get('project_id') or '<unknown>'}",
        (
            "Source issue: "
            f"#{proposal.get('source_issue_index') or '<unknown>'} "
            f"{proposal.get('source_issue_title') or ''}"
        ),
        f"Candidates: {len(items)}",
        f"Blocked candidates: {len(blocked)}",
        "Gitea writes: no",
        "Mutation executed: no",
        "Approval required for future write: yes",
    ]
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- "
            f"{item.get('candidate_id')}: "
            f"{item.get('suggested_priority')} "
            f"{item.get('title')}"
        )
    lines.append(f"Next: {proposal.get('recommended_next_action')}")
    return redact_text("\n".join(lines))
