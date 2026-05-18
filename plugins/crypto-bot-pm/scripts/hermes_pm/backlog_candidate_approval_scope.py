#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any

try:
    from scripts.hermes_pm.backlog_selection_packet import (
        BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
    from scripts.hermes_pm.issue_lifecycle_status import (
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )
    from scripts.hermes_pm.task_classifier import classify_task
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from backlog_selection_packet import (  # type: ignore[no-redef]
        BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )
    from task_classifier import classify_task  # type: ignore[no-redef]


BACKLOG_CANDIDATE_APPROVAL_SCOPE_SCHEMA_VERSION = (
    "hermes.pm.backlog_candidate_approval_scope.v1"
)

BLOCKED_TASK_CLASSES = {
    "deploy",
    "financial",
    "runtime_admin",
    "secret",
}

FORBIDDEN_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "trading/broker/live financial work",
        re.compile(
            r"\b(trading|trade|buy|sell|broker|robinhood|exchange|"
            r"live[-_ ]?market|account|order|position|wallet|financial)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "secret access",
        re.compile(
            r"(\.env\b|\b(secret|token|credential|keychain|password|"
            r"api[-_ ]?key|private[-_ ]?key|authorization|cookie)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "runtime service control",
        re.compile(
            r"\b(daemon|runtime|launchd|launchctl|qmd|scheduler|worker|"
            r"service[-_ ]?(start|stop|restart)|start\b.*\brunner)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "deployment",
        re.compile(
            r"\b(deploy|deployment|kubernetes|kubectl|flux|harbor|helm|"
            r"docker build|publish image)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "workflow/runner execution",
        re.compile(r"\b(workflows?|runners?|ci execution)\b", re.IGNORECASE),
    ),
)

EXACT_FUTURE_CONSTRAINTS = {
    "owner": DEFAULT_OWNER,
    "repo": DEFAULT_REPO,
    "gitea_base_url": DEFAULT_GITEA_BASE_URL,
    "operation_type": "create_issue",
    "max_operations": 1,
    "no_labels": True,
    "no_projects": True,
    "no_comments": True,
    "no_prs": True,
    "no_workflows": True,
    "no_runners": True,
    "no_runtime_actions": True,
    "no_financial_actions": True,
    "no_secret_access": True,
}

NON_ACTION_BOOLEANS = {
    "creates_issues": False,
    "creates_labels": False,
    "creates_comments": False,
    "mutates_projects": False,
    "creates_prs": False,
    "starts_workflows": False,
    "starts_runners": False,
    "runs_workflows": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "branch_writer_invoked": False,
    "issue_executor_invoked": False,
}


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


def _candidate_entries(packet: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(packet, dict):
        return []
    entries: list[dict[str, Any]] = []
    for key in (
        "selected_candidates",
        "candidate_summaries",
        "review_pending_candidates",
        "deferred_candidates",
        "rejected_candidates",
        "blocked_candidates",
    ):
        for item in packet.get(key) or []:
            if isinstance(item, dict):
                entries.append(item)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in entries:
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique.append(item)
    return unique


def _selected_candidate_ids(packet: dict[str, Any] | None) -> set[str]:
    if not isinstance(packet, dict):
        return set()
    return {
        str(item.get("candidate_id") or "")
        for item in packet.get("selected_candidates") or []
        if isinstance(item, dict) and item.get("candidate_id")
    }


def _future_scope_for_candidate(
    packet: dict[str, Any] | None,
    candidate_id: str,
) -> dict[str, Any] | None:
    if not isinstance(packet, dict):
        return None
    for item in packet.get("recommended_future_write_scope") or []:
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id:
            return item
    return None


def _find_candidate(
    packet: dict[str, Any] | None,
    candidate_id: str,
) -> dict[str, Any] | None:
    for item in _candidate_entries(packet):
        if item.get("candidate_id") == candidate_id:
            return item
    return None


def _issue_1_reference(
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
        "issue_url": None,
        "title": expected_title,
        "exists": False,
        "state": None,
        "lifecycle_state": "unknown",
        "duplicate_seed_issue_blocker": False,
        "matching_issue_count": 0,
        "source": "expected_issue",
    }


def _candidate_text(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    return " ".join(
        str(candidate.get(key) or "")
        for key in (
            "candidate_id",
            "title",
            "proposed_issue_body_summary",
            "proposed_issue_body",
            "rationale",
            "suggested_next_step",
            "suggested_authority_class",
            "task_class",
        )
    )


def _matched_forbidden_topics(text: str) -> list[str]:
    return sorted(
        dict.fromkeys(
            name for name, pattern in FORBIDDEN_TOPIC_PATTERNS if pattern.search(text)
        )
    )


def _proposed_issue_body(candidate: dict[str, Any], candidate_id: str) -> str:
    title = str(candidate.get("title") or "Untitled PM backlog candidate")
    summary = str(
        candidate.get("proposed_issue_body_summary")
        or "Selected Hermes PM backlog candidate."
    )
    rationale = str(candidate.get("rationale") or "No rationale supplied.")
    return redact_text(
        "\n".join(
            [
                "Hermes PM selected backlog candidate.",
                "",
                f"Candidate ID: {candidate_id}",
                f"Title: {title}",
                f"Summary: {summary}",
                f"Rationale: {rationale}",
                "",
                "Future write scope:",
                "- operation_type: create_issue",
                "- max_operations: 1",
                (
                    "- no labels, projects, comments, PRs, workflows, "
                    "runners, deploys, runtime actions, financial actions, "
                    "or sensitive material access"
                ),
                "",
                (
                    "This issue must not be created until PM-11 receives "
                    "exact Operator approval for the candidate ID, operation "
                    "ID, forge plan SHA-256, and short-lived approval artifact."
                ),
            ]
        )
    )


def _issue_only_candidate(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    authority = str(
        candidate.get("suggested_authority_class")
        or candidate.get("task_class")
        or "propose"
    )
    return authority in {"propose", "plan", "read", "forge_write"}


def build_backlog_candidate_approval_scope(
    *,
    backlog_selection_packet: dict[str, Any] | None,
    selected_candidate_id: str,
    issue_lifecycle: dict[str, Any] | None = None,
    gitea_snapshot: dict[str, Any] | None = None,
    operator_preferences: dict[str, Any] | None = None,
    project_id: str = "crypto_bot",
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    gitea_base_url: str = DEFAULT_GITEA_BASE_URL,
    issue_index: int = DEFAULT_ISSUE_INDEX,
    expected_title: str = EXPECTED_PM_SEED_ISSUE_TITLE,
    created_at: str | None = None,
) -> dict[str, Any]:
    candidate_id = str(selected_candidate_id).strip()
    candidate = _find_candidate(backlog_selection_packet, candidate_id)
    selected_ids = _selected_candidate_ids(backlog_selection_packet)
    future_scope = _future_scope_for_candidate(
        backlog_selection_packet,
        candidate_id,
    )
    issue_1 = _issue_1_reference(
        issue_lifecycle=issue_lifecycle,
        gitea_snapshot=gitea_snapshot,
        issue_index=issue_index,
        expected_title=expected_title,
    )
    created = created_at or _utc_now()
    title = str(candidate.get("title") if isinstance(candidate, dict) else "")
    body = _proposed_issue_body(candidate or {}, candidate_id) if candidate else ""

    blockers: list[str] = []
    warnings: list[str] = []
    if not candidate_id:
        blockers.append("Selected candidate ID is required.")
    if not isinstance(backlog_selection_packet, dict):
        blockers.append("Backlog selection packet is required.")
    elif backlog_selection_packet.get("schema_version") != (
        BACKLOG_SELECTION_PACKET_SCHEMA_VERSION
    ):
        blockers.append("Backlog selection packet schema is not supported.")
    if candidate is None:
        blockers.append("Candidate does not exist in the backlog selection packet.")
    elif candidate_id not in selected_ids:
        blockers.append("Candidate exists but is not selected for review.")

    if isinstance(candidate, dict):
        if candidate.get("blocked"):
            blockers.extend(str(item) for item in candidate.get("blockers") or [])
            blockers.append("Selection packet already marked this candidate blocked.")
        if candidate.get("duplicates_issue_1") or title == expected_title:
            blockers.append("Candidate duplicates the existing PM Issue #1.")
        text = _candidate_text(candidate)
        forbidden_topics = _matched_forbidden_topics(text)
        if forbidden_topics:
            blockers.append(
                "Candidate touches forbidden PM-10 topic(s): "
                + ", ".join(forbidden_topics)
                + "."
            )
        classification = classify_task(
            {
                "title": title,
                "summary": " ".join(
                    [
                        str(candidate.get("proposed_issue_body_summary") or ""),
                        str(candidate.get("rationale") or ""),
                        str(candidate.get("suggested_authority_class") or ""),
                    ]
                ),
            }
        )
        task_class = str(classification.get("task_class") or "unknown")
        if task_class in BLOCKED_TASK_CLASSES:
            blockers.append(
                f"Task class {task_class!r} is blocked for PM-10 approval scope."
            )
        if candidate.get("ci_or_workflow_related"):
            blockers.append(
                "Workflow or runner candidates are not executable in PM-10."
            )
        if not _issue_only_candidate(candidate):
            blockers.append("Candidate is not constrained to issue-only PM scope.")
        if future_scope and future_scope.get("future_operation_type") != (
            "create_issue"
        ):
            blockers.append("Future operation type must be create_issue.")
        if not future_scope:
            warnings.append(
                "Selection packet did not include a future scope entry; "
                "PM-10 constrains any future write to create_issue only."
            )
    else:
        classification = {"task_class": "unknown"}
        task_class = "unknown"

    constraints = dict(EXACT_FUTURE_CONSTRAINTS)
    constraints.update(
        {
            "owner": owner,
            "repo": repo,
            "gitea_base_url": gitea_base_url.rstrip("/"),
        }
    )
    blockers = sorted(dict.fromkeys(item for item in blockers if item))
    warnings = sorted(
        dict.fromkeys(
            [
                *warnings,
                "Future PM-11 issue creation requires exact Operator approval.",
                "This PM-10 approval scope is read-only and not executable.",
            ]
        )
    )
    scope_seed = {
        "project_id": project_id,
        "selection_packet_id": (
            backlog_selection_packet.get("selection_packet_id")
            if isinstance(backlog_selection_packet, dict)
            else None
        ),
        "candidate_id": candidate_id,
        "title": title,
        "issue_1": issue_1,
        "blocked": bool(blockers),
    }
    scope_id = f"approval-scope-{sha256_payload(scope_seed)[:16]}"
    return {
        "schema_version": BACKLOG_CANDIDATE_APPROVAL_SCOPE_SCHEMA_VERSION,
        "approval_scope_id": scope_id,
        "created_at": created,
        "project_id": project_id,
        "selected_candidate_id": candidate_id,
        "selected_candidate_title": redact_text(title) if title else None,
        "source_selection_packet_id": (
            backlog_selection_packet.get("selection_packet_id")
            if isinstance(backlog_selection_packet, dict)
            else None
        ),
        "issue_1_reference": issue_1,
        "proposed_issue_title": redact_text(title) if title else None,
        "proposed_issue_body": body or None,
        "future_operation_type": "create_issue",
        "authority_class": "forge_write",
        "exact_future_constraints": constraints,
        "blocked": bool(blockers),
        "blockers": blockers,
        "warnings": warnings,
        "approval_required_for_write": True,
        "calls_gitea_write_api": False,
        "mutation_executed": False,
        "candidate_task_class": task_class,
        "candidate_classification": classification,
        "operator_preferences_supplied": isinstance(operator_preferences, dict),
        "future_create_issue_executable_now": False,
        "requires_exact_candidate_id": True,
        "requires_exact_operation_id": True,
        "requires_exact_plan_hash": True,
        "requires_exact_approval_token": True,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


def format_backlog_candidate_approval_scope_text(scope: dict[str, Any]) -> str:
    constraints = (
        scope.get("exact_future_constraints")
        if isinstance(scope.get("exact_future_constraints"), dict)
        else {}
    )
    lines = [
        "Hermes PM backlog candidate approval scope",
        f"Project: {scope.get('project_id') or '<unknown>'}",
        (
            "Candidate: "
            f"{scope.get('selected_candidate_id') or '<missing>'} "
            f"{scope.get('selected_candidate_title') or ''}"
        ),
        f"Scope: {scope.get('approval_scope_id') or '<unknown>'}",
        f"Future operation: {scope.get('future_operation_type')}",
        (
            f"Repo: {constraints.get('owner') or '<owner>'}/"
            f"{constraints.get('repo') or '<repo>'}"
        ),
        f"Blocked: {'yes' if scope.get('blocked') else 'no'}",
        f"Blockers: {len(scope.get('blockers') or [])}",
        "Gitea writes performed: no",
        "Issue created: no",
        "Approval required for PM-11 write: yes",
    ]
    return redact_text("\n".join(lines))
