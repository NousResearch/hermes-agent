#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.issue_lifecycle_status import (
        EXPECTED_PM_SEED_ISSUE_TITLE,
        summarize_seed_issue_from_snapshot,
    )
    from scripts.hermes_pm.task_classifier import classify_task
    from scripts.hermes_operator.operator_policy import (
        Operation,
        OperatorMode,
        PolicyDecision,
        evaluate_operation,
        redact_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    repo_root_for_import = Path(__file__).resolve().parents[2]
    if str(repo_root_for_import) not in sys.path:
        sys.path.insert(0, str(repo_root_for_import))
    from scripts.hermes_pm import task_classifier as task_classifier_module
    from scripts.hermes_pm.issue_lifecycle_status import (  # type: ignore[no-redef]
        EXPECTED_PM_SEED_ISSUE_TITLE,
        summarize_seed_issue_from_snapshot,
    )
    from scripts.hermes_operator.operator_policy import (  # type: ignore[no-redef]
        Operation,
        OperatorMode,
        PolicyDecision,
        evaluate_operation,
        redact_text,
    )
    classify_task = task_classifier_module.classify_task  # type: ignore[no-redef]


FORGE_WRITE_PLAN_SCHEMA_VERSION = "hermes.pm.forge_write_plan.v1"

OPERATION_TYPES = {
    "create_issue",
    "update_issue",
    "create_label",
    "create_project_column",
    "create_project_card",
    "comment_on_issue",
    "request_pr_review",
    "no_op",
    "informational",
}

BLOCKED_TASK_CLASSES = {
    "deploy",
    "financial",
    "runtime_admin",
    "secret",
    "unknown",
}

APPROVAL_ONLY_TASK_CLASSES = {
    "branch_write",
    "ci_trial",
    "forge_write",
}

SECRET_KEY_RE = re.compile(
    r"(token|secret|password|api[_-]?key|private[_-]?key|authorization|"
    r"credential|cookie|keychain|bearer)",
    flags=re.IGNORECASE,
)
SAFE_SECRETISH_KEYS = {
    "approval_token_signing",
    "no_secret_access",
    "suggested_token_scope",
    "token_value_exposed",
}

LABEL_COLORS = {
    "approval-required": "d4a72c",
    "blocked": "d73a4a",
    "evidence-needed": "5319e7",
    "hermes-pm": "0e8a16",
    "managed-project": "1d76db",
    "needs-triage": "fbca04",
    "pm-review": "c2e0c6",
    "proposal-only": "bfd4f2",
    "read-only": "ededed",
}

NON_ACTION_BOOLEANS = {
    "no_gitea_write_performed": True,
    "no_issue_created": True,
    "no_label_created": True,
    "no_project_mutated": True,
    "no_workflow_run": True,
    "no_runner_started": True,
    "no_branch_writer_invoked": True,
    "no_secret_access": True,
    "no_runtime_action": True,
    "no_financial_action": True,
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


def redact_secret_values(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if (
                SECRET_KEY_RE.search(text_key)
                and text_key not in SAFE_SECRETISH_KEYS
                and not isinstance(item, bool)
            ):
                redacted[text_key] = "<redacted>"
            else:
                redacted[text_key] = redact_secret_values(item)
        return redacted
    if isinstance(value, list):
        return [redact_secret_values(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secret_values(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _quoted(value: Any) -> str:
    return urllib.parse.quote(str(value), safe="")


def repo_endpoint(owner: str, repo: str, suffix: str = "") -> str:
    return f"/api/v1/repos/{_quoted(owner)}/{_quoted(repo)}{suffix}"


def render_gitea_endpoint(
    operation_type: str,
    *,
    owner: str,
    repo: str,
    project_id: str,
    issue_number: Any = None,
    pr_number: Any = None,
    column: str | None = None,
) -> tuple[str, str]:
    if operation_type == "create_issue":
        return "POST", repo_endpoint(owner, repo, "/issues")
    if operation_type == "update_issue":
        number = _quoted(issue_number or "{issue_index}")
        return "POST", repo_endpoint(owner, repo, f"/issues/{number}/labels")
    if operation_type == "create_label":
        return "POST", repo_endpoint(owner, repo, "/labels")
    if operation_type == "create_project_column":
        project = _quoted(project_id)
        return "POST", repo_endpoint(owner, repo, f"/projects/{project}/columns")
    if operation_type == "create_project_card":
        project = _quoted(project_id)
        column_id = _quoted(column or "{column_id}")
        suffix = f"/projects/{project}/columns/{column_id}/cards"
        return "POST", repo_endpoint(owner, repo, suffix)
    if operation_type == "comment_on_issue":
        number = _quoted(issue_number or "{issue_index}")
        return "POST", repo_endpoint(owner, repo, f"/issues/{number}/comments")
    if operation_type == "request_pr_review":
        number = _quoted(pr_number or "{pull_index}")
        return "POST", repo_endpoint(owner, repo, f"/pulls/{number}/reviews")
    return "NONE", "<none>"


def _existing_label_names(gitea_snapshot: dict[str, Any] | None) -> set[str]:
    if not isinstance(gitea_snapshot, dict):
        return set()
    labels = gitea_snapshot.get("labels")
    if not isinstance(labels, list):
        return set()
    return {
        str(item.get("name")).strip().lower()
        for item in labels
        if isinstance(item, dict) and item.get("name")
    }


def _existing_issue_refs(
    *,
    kanban_packet: dict[str, Any],
    gitea_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in kanban_packet.get("existing_issue_refs") or []:
        if not isinstance(item, dict):
            continue
        refs.append(
            {
                "issue_index": item.get("issue_index"),
                "issue_url": item.get("issue_url"),
                "title": item.get("title"),
                "state": item.get("state"),
                "lifecycle_state": item.get("lifecycle_state"),
                "source": item.get("source") or "kanban_packet",
            }
        )
    seed_ref = summarize_seed_issue_from_snapshot(
        gitea_snapshot,
        issue_index=1,
        expected_title=EXPECTED_PM_SEED_ISSUE_TITLE,
    )
    if seed_ref.get("exists"):
        refs.append(
            {
                "issue_index": seed_ref.get("issue_index"),
                "issue_url": seed_ref.get("issue_url"),
                "title": seed_ref.get("title"),
                "state": seed_ref.get("state"),
                "lifecycle_state": seed_ref.get("lifecycle_state"),
                "source": "gitea_snapshot",
            }
        )
    seen: set[tuple[str, str]] = set()
    unique_refs: list[dict[str, Any]] = []
    for ref in refs:
        key = (str(ref.get("issue_index") or ""), str(ref.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append(ref)
    return unique_refs


def _projects_unavailable(gitea_snapshot: dict[str, Any] | None) -> bool:
    if not isinstance(gitea_snapshot, dict):
        return False
    blockers = gitea_snapshot.get("blockers")
    if not isinstance(blockers, list):
        return False
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        endpoint = str(blocker.get("endpoint") or "")
        error = str(blocker.get("error") or "").lower()
        if endpoint.endswith("/projects") and "404" in error:
            return True
    return False


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _classify_payload(
    *,
    title: str,
    reason: str,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    return classify_task(
        {
            "title": title,
            "summary": reason,
            "labels": labels or [],
        }
    )


def _policy_for_operation(
    *,
    operation_type: str,
    endpoint: str,
    method: str,
    title: str,
    payload: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    policy_payload = dict(payload)
    body = policy_payload.get("body")
    if isinstance(body, str) and "No labels, projects, comments" in body:
        for marker in (
            (
                "No labels, projects, comments, PRs, workflows, runners, "
                "deploys, runtime actions, branch-writer actions, financial "
                "actions, broker/trading actions, or secret access were approved."
            ),
            (
                "No labels, projects, comments, PRs, workflows, runners, "
                "deploys, runtime actions, branch-writer actions, financial "
                "actions, or secret access were approved."
            ),
        ):
            if marker in body:
                policy_payload["body"] = body.replace(
                    marker,
                    "No non-PM side effects were approved.",
                )
                break
    result = evaluate_operation(
        Operation(
            authority_class="forge_write",
            operation_name=operation_type,
            target=endpoint,
            method=method,
            command=f"{title} {canonical_json(policy_payload)}",
            reason=reason,
        ),
        mode=OperatorMode.FORGE_ASSISTED,
    )
    return result.to_dict()


def _blockers_and_warnings(
    *,
    classification: dict[str, Any],
    policy_result: dict[str, Any],
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    task_class = str(classification.get("task_class") or "unknown")
    if task_class in BLOCKED_TASK_CLASSES:
        blockers.append(
            f"Task class {task_class!r} is blocked for PM forge planning."
        )
    if bool(classification.get("forbidden")):
        blockers.extend(
            str(item)
            for item in classification.get("approval_requirements") or []
        )
    if task_class in APPROVAL_ONLY_TASK_CLASSES:
        warnings.append(
            f"Underlying task class {task_class!r} requires separate approval."
        )
    if policy_result.get("decision") == PolicyDecision.DENY.value:
        blockers.extend(str(item) for item in policy_result.get("reasons") or [])
    return sorted(dict.fromkeys(blockers)), sorted(dict.fromkeys(warnings))


def _operation(
    *,
    index: int,
    operation_type: str,
    title: str,
    classification_title: str | None = None,
    classification_reason: str | None = None,
    target_summary: str,
    reason: str,
    payload: dict[str, Any],
    owner: str,
    repo: str,
    project_id: str,
    issue_number: Any = None,
    pr_number: Any = None,
    column: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if operation_type not in OPERATION_TYPES:
        operation_type = "informational"
    method, endpoint = render_gitea_endpoint(
        operation_type,
        owner=owner,
        repo=repo,
        project_id=project_id,
        issue_number=issue_number,
        pr_number=pr_number,
        column=column,
    )
    redacted_payload = redact_secret_values(payload)
    classification = _classify_payload(
        title=classification_title or title,
        reason=classification_reason or reason,
        labels=labels,
    )
    policy_result = _policy_for_operation(
        operation_type=operation_type,
        endpoint=endpoint,
        method=method,
        title=title,
        payload=redacted_payload,
        reason=reason,
    )
    blockers, warnings = _blockers_and_warnings(
        classification=classification,
        policy_result=policy_result,
    )
    return {
        "operation_id": f"fwop-{index:03d}",
        "operation_type": operation_type,
        "title": redact_text(title),
        "target_summary": redact_text(target_summary),
        "reason": redact_text(reason),
        "proposed_payload_redacted": redacted_payload,
        "expected_gitea_endpoint": endpoint,
        "http_method_that_would_be_used": method,
        "authority_class": "forge_write",
        "requires_operator_approval": True,
        "would_mutate_gitea": operation_type not in {"no_op", "informational"},
        "executed": False,
        "blocked": bool(blockers),
        "blockers": blockers,
        "warnings": warnings,
        "task_classification": classification,
        "policy_result": policy_result,
    }


def _label_payload(name: str) -> dict[str, Any]:
    label = name.strip()
    return {
        "name": label,
        "color": LABEL_COLORS.get(label.lower(), "ededed"),
        "description": f"Hermes PM proposed label: {label}",
    }


def _issue_payload_from_card(card: dict[str, Any]) -> dict[str, Any]:
    labels = _text_list(card.get("labels"))
    title = str(card.get("title") or "Untitled Hermes PM card")
    rationale = str(card.get("rationale") or "Hermes PM proposed work item.")
    explicit_body = card.get("issue_body")
    if isinstance(explicit_body, str) and explicit_body.strip():
        return {
            "title": title,
            "body": explicit_body,
            "labels": labels,
        }
    return {
        "title": title,
        "body": (
            "Hermes PM proposed issue from Kanban packet.\n\n"
            f"Source: {card.get('source') or 'unknown'}\n"
            f"Column: {card.get('column') or 'Inbox'}\n"
            f"Rationale: {rationale}\n\n"
            "This is a dry-run preview only. No issue was created."
        ),
        "labels": labels,
    }


def _project_card_payload(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "column": str(card.get("column") or "Inbox"),
        "title": str(card.get("title") or "Untitled Hermes PM card"),
        "source": str(card.get("source") or "kanban_packet"),
        "source_number": card.get("source_number"),
        "labels": _text_list(card.get("labels")),
    }


def _selected_backlog_candidates(
    backlog_selection_packet: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(backlog_selection_packet, dict):
        return []
    return [
        item
        for item in backlog_selection_packet.get("selected_candidates") or []
        if isinstance(item, dict)
    ]


def _unselected_backlog_candidates(
    backlog_selection_packet: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(backlog_selection_packet, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for key in (
        "review_pending_candidates",
        "deferred_candidates",
        "rejected_candidates",
        "blocked_candidates",
    ):
        for item in backlog_selection_packet.get(key) or []:
            if isinstance(item, dict):
                candidates.append(item)
    return candidates


def _issue_payload_from_selected_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    title = str(candidate.get("title") or "Untitled selected PM backlog candidate")
    explicit_body = candidate.get("proposed_issue_body")
    if isinstance(explicit_body, str) and explicit_body.strip():
        return {
            "title": title,
            "body": explicit_body,
        }
    summary = str(
        candidate.get("proposed_issue_body_summary")
        or "Selected Hermes PM backlog candidate."
    )
    candidate_id = str(candidate.get("candidate_id") or "<unknown>")
    return {
        "title": title,
        "body": (
            "Hermes PM selected backlog candidate preview.\n\n"
            f"Candidate ID: {candidate_id}\n"
            f"Summary: {summary}\n\n"
            "This is a dry-run preview only. No issue was created. Future "
            "creation requires exact Operator approval for this candidate ID, "
            "operation ID, and plan hash."
        ),
    }


def _low_risk_selected_candidate(candidate: dict[str, Any]) -> bool:
    authority = str(
        candidate.get("suggested_authority_class")
        or candidate.get("task_class")
        or ""
    )
    return (
        authority in {"propose", "plan", "read"}
        and not candidate.get("blocked")
        and not candidate.get("ci_or_workflow_related")
    )


def _blocked_operation_summary(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_id": operation.get("operation_id"),
        "operation_type": operation.get("operation_type"),
        "title": operation.get("title"),
        "target_summary": operation.get("target_summary"),
        "blockers": operation.get("blockers") or [],
        "task_class": (
            operation.get("task_classification", {}).get("task_class")
            if isinstance(operation.get("task_classification"), dict)
            else None
        ),
        "policy_decision": (
            operation.get("policy_result", {}).get("decision")
            if isinstance(operation.get("policy_result"), dict)
            else None
        ),
    }


def _risk_classification(operations: list[dict[str, Any]]) -> dict[str, Any]:
    risk_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    risk_counts: dict[str, int] = {}
    for operation in operations:
        policy = operation.get("policy_result")
        risk = "CRITICAL"
        if isinstance(policy, dict):
            risk = str(policy.get("risk_level") or "CRITICAL")
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    max_risk = "LOW"
    if risk_counts:
        max_risk = max(risk_counts, key=lambda item: risk_order.get(item, 999))
    return {
        "max_risk": max_risk,
        "risk_counts": risk_counts,
        "all_operations_are_dry_run": True,
        "all_operations_require_operator_approval": all(
            bool(operation.get("requires_operator_approval"))
            for operation in operations
        ),
    }


def _policy_summary(operations: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [item for item in operations if item.get("blocked")]
    approval_required = [
        item
        for item in operations
        if item.get("requires_operator_approval") and not item.get("blocked")
    ]
    return {
        "decision": (
            "request_revision"
            if blocked
            else "approve_selected"
            if approval_required
            else "approve_none"
        ),
        "safe_to_execute_now": False,
        "blocked_operation_count": len(blocked),
        "operations_requiring_approval": len(approval_required),
        "reasons": [
            "This checkpoint emits a dry-run forge-write plan only.",
            "No future Gitea mutation may run without explicit Operator approval.",
        ],
        "forbidden_if_present": [
            "financial",
            "secret",
            "runtime_admin",
            "deploy",
            "unknown",
        ],
    }


def _approval_requirements(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "operation_id": operation["operation_id"],
            "operation_type": operation["operation_type"],
            "approval_required": True,
            "blocked": bool(operation.get("blocked")),
            "required_evidence": [
                "explicit Operator approval naming this operation_id",
                "matching forge-write plan SHA-256 in the approval packet",
                "reviewed endpoint preview and redacted payload preview",
                "confirmation that no forbidden PM/runtime/financial/secret "
                "surface is involved",
            ],
        }
        for operation in operations
    ]


def _evidence_requirements() -> list[str]:
    return [
        "Current PM status report generated in read-only mode.",
        "Current Kanban proposal packet or its SHA-256.",
        "This forge-write dry-run plan and its SHA-256.",
        "Operator approval scoped to selected operation_id values.",
        "Redacted endpoint and payload previews reviewed by the Operator.",
        "Proof that no Gitea write API was called during planning.",
        "Proof that no workflow, runner, deploy, runtime, financial, secret, "
        "or branch-writer action occurred.",
        "Rollback or manual correction notes for each future forge mutation.",
    ]


def build_forge_write_plan(
    *,
    kanban_packet: dict[str, Any],
    gitea_snapshot: dict[str, Any] | None = None,
    backlog_selection_packet: dict[str, Any] | None = None,
    project_id: str | None = None,
    gitea_base_url: str = "http://127.0.0.1:3005",
    owner: str = "preston",
    repo: str = "crypto_bot",
    created_at: str | None = None,
) -> dict[str, Any]:
    created = created_at or _utc_now()
    resolved_project_id = str(
        project_id
        or kanban_packet.get("project_id")
        or (gitea_snapshot or {}).get("project_id")
        or "crypto_bot"
    )
    existing_labels = _existing_label_names(gitea_snapshot)
    existing_issue_refs = _existing_issue_refs(
        kanban_packet=kanban_packet,
        gitea_snapshot=gitea_snapshot,
    )
    existing_issue_titles = {
        str(item.get("title"))
        for item in existing_issue_refs
        if item.get("title")
    }
    project_operations_available = not _projects_unavailable(gitea_snapshot)
    duplicate_issue_deduped: list[dict[str, Any]] = []
    proposal_only_backlog_cards: list[dict[str, Any]] = []
    selected_backlog_preview_operations: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    selection_supplied = isinstance(backlog_selection_packet, dict)

    def add_operation(**kwargs: Any) -> dict[str, Any]:
        operation = _operation(
            index=len(operations) + 1,
            owner=owner,
            repo=repo,
            project_id=resolved_project_id,
            **kwargs,
        )
        operations.append(operation)
        return operation

    if not selection_supplied:
        for label in _text_list(kanban_packet.get("suggested_labels")):
            if label.strip().lower() in existing_labels:
                continue
            add_operation(
                operation_type="create_label",
                title=f"Create Gitea label {label}",
                classification_title=f"Propose PM label {label}",
                classification_reason="PM label proposal for review.",
                target_summary=f"label:{label}",
                reason=(
                    "Kanban packet suggested a PM label that is not known to "
                    "exist."
                ),
                payload=_label_payload(label),
                labels=[label],
            )

        for column in _text_list(kanban_packet.get("proposed_columns")):
            if not project_operations_available:
                continue
            add_operation(
                operation_type="create_project_column",
                title=f"Create PM project column {column}",
                classification_title=f"Propose PM project column {column}",
                classification_reason="PM board planning proposal.",
                target_summary=f"project:{resolved_project_id} column:{column}",
                reason="Kanban packet proposed a PM board column.",
                payload={"name": column},
                column=column,
            )

    if selection_supplied:
        for candidate in _unselected_backlog_candidates(backlog_selection_packet):
            proposal_only_backlog_cards.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "title": candidate.get("title"),
                    "source": "backlog_selection_packet",
                    "selection_status": candidate.get("selection_status"),
                    "reason": (
                        "PM-9 backlog selection packet did not select this "
                        "candidate for future issue creation preview."
                    ),
                    "create_issue_operation_generated": False,
                    "requires_future_approval": True,
                }
            )
        for candidate in _selected_backlog_candidates(backlog_selection_packet):
            title = str(candidate.get("title") or "Untitled selected PM candidate")
            candidate_id = str(candidate.get("candidate_id") or "")
            if (
                title in existing_issue_titles
                or title == EXPECTED_PM_SEED_ISSUE_TITLE
                or candidate.get("duplicates_issue_1")
            ):
                duplicate_issue_deduped.append(
                    {
                        "candidate_id": candidate_id,
                        "title": title,
                        "existing_issue_index": 1
                        if title == EXPECTED_PM_SEED_ISSUE_TITLE
                        else None,
                        "existing_issue_url": None,
                        "reason": "duplicate_existing_issue",
                    }
                )
                continue
            operation = add_operation(
                operation_type="create_issue",
                title=f"Create issue for selected backlog candidate {candidate_id}",
                classification_title=title,
                classification_reason=str(
                    candidate.get("rationale")
                    or candidate.get("proposed_issue_body_summary")
                    or ""
                ),
                target_summary=f"repo:{owner}/{repo} selected backlog issue",
                reason=(
                    "PM-9 backlog selection packet explicitly selected this "
                    "candidate for a future create_issue dry-run preview."
                ),
                payload=_issue_payload_from_selected_candidate(candidate),
                labels=[],
            )
            operation["source_backlog_selection_packet_id"] = (
                backlog_selection_packet.get("selection_packet_id")
            )
            operation["source_candidate_id"] = candidate_id
            operation["source_backlog_candidate_authority_class"] = (
                candidate.get("suggested_authority_class")
            )
            operation["requires_exact_approval_token"] = True
            operation["requires_exact_plan_hash"] = True
            operation["requires_operator_confirmation"] = True
            operation["executable_by_default"] = False
            operation["low_risk_backlog_candidate"] = _low_risk_selected_candidate(
                candidate
            )
            selected_backlog_preview_operations.append(
                {
                    "candidate_id": candidate_id,
                    "operation_id": operation["operation_id"],
                    "title": title,
                    "create_issue_operation_generated": True,
                    "requires_future_approval": True,
                    "executable_by_default": False,
                }
            )

    cards = [
        item
        for item in kanban_packet.get("proposed_cards") or []
        if isinstance(item, dict)
    ]
    for card in [] if selection_supplied else cards:
        title = str(card.get("title") or "Untitled Hermes PM card")
        source = str(card.get("source") or "kanban_packet")
        labels = _text_list(card.get("labels"))
        existing_ref = next(
            (
                item
                for item in existing_issue_refs
                if item.get("title") == title
            ),
            None,
        )
        if source == "existing_gitea_issue":
            continue
        if source == "backlog_expansion_proposal" or bool(
            card.get("proposal_only")
        ):
            proposal_only_backlog_cards.append(
                {
                    "candidate_id": card.get("candidate_id")
                    or card.get("source_number"),
                    "title": title,
                    "source": source,
                    "reason": (
                        "PM-8 backlog candidates are proposal-only by default; "
                        "future issue creation requires an explicit approval "
                        "packet and selected operation IDs."
                    ),
                    "create_issue_operation_generated": False,
                    "requires_future_approval": True,
                }
            )
            continue
        if source in {
            "gitea_issue",
            "gitea_pull_request",
        } or title in existing_issue_titles:
            if not project_operations_available:
                continue
            if existing_ref is not None and source not in {
                "gitea_issue",
                "gitea_pull_request",
                "existing_gitea_issue",
            }:
                duplicate_issue_deduped.append(
                    {
                        "title": title,
                        "existing_issue_index": existing_ref.get("issue_index"),
                        "existing_issue_url": existing_ref.get("issue_url"),
                        "reason": "duplicate_existing_issue",
                    }
                )
            add_operation(
                operation_type="create_project_card",
                title=f"Create project card for {title}",
                classification_title=title,
                classification_reason=str(card.get("rationale") or ""),
                target_summary=(
                    f"project:{resolved_project_id} "
                    f"column:{card.get('column') or 'Inbox'}"
                ),
                reason="Existing Gitea item should appear on the PM board.",
                payload=_project_card_payload(card),
                column=str(card.get("column") or "Inbox"),
                labels=labels,
            )
            continue
        add_operation(
            operation_type="create_issue",
            title=f"Create issue for {title}",
            classification_title=title,
            classification_reason=str(card.get("rationale") or ""),
            target_summary=f"repo:{owner}/{repo} issue backlog",
            reason=(
                "Kanban packet proposed a PM card without an existing Gitea "
                "issue source."
            ),
            payload=_issue_payload_from_card(card),
            labels=labels,
        )
        if not project_operations_available:
            continue
        add_operation(
            operation_type="create_project_card",
            title=f"Create project card for {title}",
            classification_title=title,
            classification_reason=str(card.get("rationale") or ""),
            target_summary=(
                f"project:{resolved_project_id} "
                f"column:{card.get('column') or 'Inbox'}"
            ),
            reason="New proposed issue should be placed on the PM board.",
            payload=_project_card_payload(card),
            column=str(card.get("column") or "Inbox"),
            labels=labels,
        )

    issue_updates = [] if selection_supplied else [
        item
        for item in kanban_packet.get("suggested_issue_updates") or []
        if isinstance(item, dict)
    ]
    for update in issue_updates:
        number = update.get("issue_number")
        labels = _text_list(update.get("suggested_labels_to_add"))
        add_operation(
            operation_type="update_issue",
            title=f"Add labels to issue #{number}",
            classification_title=str(update.get("title") or f"Issue #{number}"),
            classification_reason="PM triage label proposal.",
            target_summary=f"issue:{number}",
            reason="Kanban packet suggested issue labels for PM triage.",
            payload={"labels": labels},
            issue_number=number,
            labels=labels,
        )

    pr_attention = [] if selection_supplied else [
        item
        for item in kanban_packet.get("suggested_pr_attention") or []
        if isinstance(item, dict)
    ]
    for attention in pr_attention:
        number = attention.get("pr_number")
        title = str(attention.get("title") or f"PR #{number}")
        add_operation(
            operation_type="request_pr_review",
            title=f"Request PM review for PR #{number}: {title}",
            classification_title=title,
            classification_reason="Summarize status and propose PR attention.",
            target_summary=f"pull_request:{number}",
            reason="Kanban packet marked an open PR for PM attention.",
            payload={
                "body": attention.get("suggested_attention"),
                "reviewers": [],
            },
            pr_number=number,
        )

    blocked_operations = [
        _blocked_operation_summary(operation)
        for operation in operations
        if operation.get("blocked")
    ]
    endpoint_preview = [
        {
            "operation_id": operation["operation_id"],
            "method": operation["http_method_that_would_be_used"],
            "endpoint": operation["expected_gitea_endpoint"],
            "called": False,
        }
        for operation in operations
    ]
    payload_preview_redacted = [
        {
            "operation_id": operation["operation_id"],
            "payload": operation["proposed_payload_redacted"],
        }
        for operation in operations
    ]
    source_packet_sha = sha256_payload(redact_secret_values(kanban_packet))
    selection_packet_sha = (
        sha256_payload(redact_secret_values(backlog_selection_packet))
        if selection_supplied
        else None
    )
    plan_id_seed = {
        "project_id": resolved_project_id,
        "source_packet_sha256": source_packet_sha,
        "source_backlog_selection_packet_sha256": selection_packet_sha,
        "owner": owner,
        "repo": repo,
        "base_url": redact_text(gitea_base_url),
        "operation_count": len(operations),
    }
    plan_id = f"forge-plan-{sha256_payload(plan_id_seed)[:16]}"
    return redact_secret_values(
        {
            "schema_version": FORGE_WRITE_PLAN_SCHEMA_VERSION,
            "plan_id": plan_id,
            "created_at": created,
            "project_id": resolved_project_id,
            "source_kanban_packet_id": (
                kanban_packet.get("kanban_packet_id")
                or kanban_packet.get("packet_id")
            ),
            "source_kanban_packet_sha256": source_packet_sha,
            "source_backlog_selection_packet_id": (
                backlog_selection_packet.get("selection_packet_id")
                if selection_supplied
                else None
            ),
            "source_backlog_selection_packet_sha256": selection_packet_sha,
            "selected_candidate_ids": [
                item.get("candidate_id")
                for item in _selected_backlog_candidates(backlog_selection_packet)
            ],
            "backlog_selection_supplied": selection_supplied,
            "existing_issue_refs": existing_issue_refs,
            "deduplication_summary": {
                "seed_issue_title": EXPECTED_PM_SEED_ISSUE_TITLE,
                "seed_issue_exists": any(
                    item.get("title") == EXPECTED_PM_SEED_ISSUE_TITLE
                    and item.get("issue_index") in {1, "1"}
                    for item in existing_issue_refs
                ),
                "duplicate_existing_issue": bool(duplicate_issue_deduped),
                "duplicate_issue_deduped": duplicate_issue_deduped,
                "duplicate_create_issue_operations": [
                    operation.get("operation_id")
                    for operation in operations
                    if operation.get("operation_type") == "create_issue"
                    and (
                        operation.get("proposed_payload_redacted", {}).get("title")
                        in existing_issue_titles
                    )
                ],
                "duplicate_create_issue_future_executable": False,
                "project_card_operations_omitted": not project_operations_available,
                "project_card_omission_reason": (
                    "project_endpoint_unavailable"
                    if not project_operations_available
                    else None
                ),
                "proposal_only_backlog_candidate_count": len(
                    proposal_only_backlog_cards
                ),
                "proposal_only_backlog_create_issue_operations": [],
                "selected_backlog_create_issue_operations": [
                    item["operation_id"]
                    for item in selected_backlog_preview_operations
                ],
            },
            "proposal_only_backlog_candidates": proposal_only_backlog_cards,
            "selected_backlog_create_issue_previews": (
                selected_backlog_preview_operations
            ),
            "gitea_base_url": redact_text(gitea_base_url.rstrip("/")),
            "owner": owner,
            "repo": repo,
            "dry_run": True,
            "mutation_executed": False,
            "calls_gitea_write_api": False,
            "approval_required_for_write": True,
            "operations": operations,
            "blocked_operations": blocked_operations,
            "approval_requirements": _approval_requirements(operations),
            "endpoint_preview": endpoint_preview,
            "payload_preview_redacted": payload_preview_redacted,
            "risk_classification": _risk_classification(operations),
            "policy_result": _policy_summary(operations),
            "recommended_operator_decision": _policy_summary(operations)[
                "decision"
            ],
            "rollback_notes": [
                "No rollback is needed for this dry run because no mutation "
                "was executed.",
                "A future applied forge write must include manual rollback or "
                "correction notes for created issues, labels, comments, and "
                "project cards.",
            ],
            "evidence_requirements": _evidence_requirements(),
            "explicit_non_actions": dict(NON_ACTION_BOOLEANS),
        }
    )


def format_forge_write_plan_text(plan: dict[str, Any]) -> str:
    operations = (
        plan.get("operations") if isinstance(plan.get("operations"), list) else []
    )
    blocked = plan.get("blocked_operations")
    blocked_count = len(blocked) if isinstance(blocked, list) else 0
    endpoints = plan.get("endpoint_preview")
    endpoint_count = len(endpoints) if isinstance(endpoints, list) else 0
    lines = [
        "Hermes PM forge-write dry-run plan",
        f"Project: {plan.get('project_id') or '<unknown>'}",
        f"Plan: {plan.get('plan_id') or '<unknown>'}",
        f"Repo: {plan.get('owner') or '<owner>'}/{plan.get('repo') or '<repo>'}",
        f"Dry run: {'yes' if plan.get('dry_run') else 'no'}",
        "Gitea writes performed: no",
        f"Operations: {len(operations)}",
        f"Blocked operations: {blocked_count}",
        f"Endpoint previews: {endpoint_count}",
        f"Decision: {plan.get('recommended_operator_decision')}",
        "Approval required before write: yes",
    ]
    return redact_text("\n".join(lines))
