#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any

try:
    from scripts.hermes_pm.backlog_expansion_proposal import (
        BACKLOG_EXPANSION_SCHEMA_VERSION,
        DEFAULT_CANDIDATE_SPECS,
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
    from backlog_expansion_proposal import (  # type: ignore[no-redef]
        BACKLOG_EXPANSION_SCHEMA_VERSION,
        DEFAULT_CANDIDATE_SPECS,
        redact_text,
    )
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )
    from task_classifier import classify_task  # type: ignore[no-redef]


BACKLOG_SELECTION_PACKET_SCHEMA_VERSION = (
    "hermes.pm.backlog_selection_packet.v1"
)
BACKLOG_SELECTION_PREFERENCES_SCHEMA_VERSION = (
    "hermes.pm.backlog_selection_preferences.v1"
)

DEFAULT_PRIORITY_BIAS = (
    "pm_governance",
    "plugin_reliability",
    "read_only_observability",
)

BLOCKED_TASK_CLASSES = {
    "deploy",
    "financial",
    "runtime_admin",
    "secret",
    "unknown",
}

CI_OR_WORKFLOW_PATTERNS = (
    re.compile(r"\bci\b", re.IGNORECASE),
    re.compile(r"\bgitea actions\b", re.IGNORECASE),
    re.compile(r"\bworkflows?\b", re.IGNORECASE),
    re.compile(r"\brunners?\b", re.IGNORECASE),
)

FORBIDDEN_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("trading", re.compile(r"\b(trading|trade|buy|sell)\b", re.IGNORECASE)),
    ("broker", re.compile(r"\b(broker|robinhood|exchange)\b", re.IGNORECASE)),
    (
        "live financial actions",
        re.compile(
            r"\b(live[-_ ]?market|account|order|position|wallet|financial)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "secrets",
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
        "deployments",
        re.compile(
            r"\b(deploy|deployment|kubernetes|kubectl|flux|harbor|helm|"
            r"docker build|publish image)\b",
            re.IGNORECASE,
        ),
    ),
)

NON_ACTION_BOOLEANS = {
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


def _as_id_set(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {str(item).strip() for item in values if str(item).strip()}


def _preference_ids(
    operator_preferences: dict[str, Any] | None,
    key: str,
) -> set[str]:
    if not isinstance(operator_preferences, dict):
        return set()
    return _as_id_set(operator_preferences.get(key))


def _priority_bias(operator_preferences: dict[str, Any] | None) -> list[str]:
    if not isinstance(operator_preferences, dict):
        return list(DEFAULT_PRIORITY_BIAS)
    raw = operator_preferences.get("priority_bias")
    if not isinstance(raw, list):
        return list(DEFAULT_PRIORITY_BIAS)
    values = [str(item).strip() for item in raw if str(item).strip()]
    return values or list(DEFAULT_PRIORITY_BIAS)


def _forbidden_topics(operator_preferences: dict[str, Any] | None) -> list[str]:
    defaults = ["trading", "broker", "secrets", "runtime", "deploy"]
    if not isinstance(operator_preferences, dict):
        return defaults
    raw = operator_preferences.get("forbidden_topics")
    if not isinstance(raw, list):
        return defaults
    values = [str(item).strip().lower() for item in raw if str(item).strip()]
    return values or defaults


def _source_issue_from_inputs(
    *,
    backlog_expansion_proposal: dict[str, Any] | None,
    issue_lifecycle: dict[str, Any] | None,
    gitea_snapshot: dict[str, Any] | None,
    issue_index: int,
    expected_title: str,
) -> dict[str, Any]:
    if isinstance(backlog_expansion_proposal, dict):
        return {
            "issue_index": backlog_expansion_proposal.get("source_issue_index")
            or issue_index,
            "title": backlog_expansion_proposal.get("source_issue_title")
            or expected_title,
            "state": None,
            "lifecycle_state": None,
            "issue_url": None,
            "source": "backlog_expansion_proposal",
        }
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
        "state": None,
        "lifecycle_state": "unknown",
        "issue_url": None,
        "source": "expected_issue",
    }


def _fallback_backlog_proposal(
    *,
    project_id: str,
    issue_index: int,
    expected_title: str,
) -> dict[str, Any]:
    return {
        "schema_version": BACKLOG_EXPANSION_SCHEMA_VERSION,
        "proposal_id": None,
        "project_id": project_id,
        "source_issue_index": issue_index,
        "source_issue_title": expected_title,
        "proposed_backlog_items": list(DEFAULT_CANDIDATE_SPECS),
        "blocked_candidates": [],
        "calls_gitea_write_api": False,
        "mutation_executed": False,
    }


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in (
            "title",
            "rationale",
            "proposed_issue_body_summary",
            "suggested_next_step",
            "suggested_authority_class",
        )
    )


def _candidate_blocking_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in (
            "title",
            "suggested_authority_class",
        )
    )


def _matched_forbidden_topics(
    text: str,
    *,
    preferred_forbidden_topics: list[str],
) -> list[str]:
    matched = [
        name for name, pattern in FORBIDDEN_TOPIC_PATTERNS if pattern.search(text)
    ]
    preferred = {item.lower() for item in preferred_forbidden_topics}
    if "runtime" in preferred and "runtime service control" in matched:
        pass
    if "deploy" in preferred and "deployments" in matched:
        pass
    return sorted(dict.fromkeys(matched))


def _is_ci_or_workflow_candidate(text: str) -> bool:
    return any(pattern.search(text) for pattern in CI_OR_WORKFLOW_PATTERNS)


def _candidate_classification(
    candidate: dict[str, Any],
    *,
    preferred_forbidden_topics: list[str],
) -> dict[str, Any]:
    text = _candidate_text(candidate)
    blocking_text = _candidate_blocking_text(candidate)
    classification = classify_task(
        {
            "title": candidate.get("title"),
            "summary": blocking_text,
        }
    )
    task_class = str(
        classification.get("task_class")
        or candidate.get("suggested_authority_class")
        or "unknown"
    )
    suggested = str(candidate.get("suggested_authority_class") or "")
    if (
        suggested
        and task_class not in BLOCKED_TASK_CLASSES
        and task_class not in {"ci_trial", "forge_write", "branch_write"}
    ):
        task_class = suggested
    forbidden_topics = _matched_forbidden_topics(
        blocking_text,
        preferred_forbidden_topics=preferred_forbidden_topics,
    )
    ci_related = _is_ci_or_workflow_candidate(text)
    blockers = list(candidate.get("blockers") or [])
    if candidate.get("blocked"):
        blockers.append(
            "Source backlog proposal already marked this candidate blocked."
        )
    if task_class in BLOCKED_TASK_CLASSES:
        blockers.extend(
            str(item) for item in classification.get("approval_requirements") or []
        )
    if forbidden_topics:
        blockers.append(
            "Candidate touches forbidden PM-9 topic(s): "
            + ", ".join(forbidden_topics)
            + "."
        )
    blocked = bool(candidate.get("blocked") or task_class in BLOCKED_TASK_CLASSES)
    if forbidden_topics:
        blocked = True
    approval_required = bool(
        blocked
        or candidate.get("approval_required")
        or task_class in {"branch_write", "ci_trial", "forge_write"}
        or ci_related
    )
    return {
        "task_class": task_class,
        "classification": classification,
        "forbidden_topics": forbidden_topics,
        "ci_or_workflow_related": ci_related,
        "blocked": blocked,
        "approval_required": approval_required,
        "blockers": sorted(dict.fromkeys(str(item) for item in blockers if item)),
    }


def _candidate_summary(
    candidate: dict[str, Any],
    *,
    selection_status: str,
    classification: dict[str, Any],
    source_issue_title: str,
) -> dict[str, Any]:
    title = str(candidate.get("title") or "Untitled PM backlog candidate")
    duplicates_seed = (
        title == source_issue_title or title == EXPECTED_PM_SEED_ISSUE_TITLE
    )
    blockers = list(classification["blockers"])
    if duplicates_seed:
        blockers.append("Candidate duplicates the existing PM seed issue.")
    blocked = bool(classification["blocked"] or duplicates_seed)
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "title": redact_text(title),
        "proposed_issue_body_summary": redact_text(
            str(
                candidate.get("proposed_issue_body_summary")
                or "Proposal-only PM backlog candidate."
            )
        ),
        "rationale": redact_text(str(candidate.get("rationale") or "")),
        "source": str(candidate.get("source") or "backlog_expansion_proposal"),
        "suggested_priority": str(candidate.get("suggested_priority") or "P2"),
        "suggested_authority_class": str(
            candidate.get("suggested_authority_class")
            or classification.get("task_class")
            or "propose"
        ),
        "task_class": classification.get("task_class"),
        "selection_status": selection_status,
        "approval_required": bool(classification["approval_required"] or blocked),
        "blocked": blocked,
        "blockers": sorted(dict.fromkeys(blockers)),
        "ci_or_workflow_related": bool(classification["ci_or_workflow_related"]),
        "non_executable": True,
        "proposal_only": True,
        "duplicates_issue_1": duplicates_seed,
        "future_create_issue_requires_approval": True,
    }


def _score_candidate(candidate: dict[str, Any], priority_bias: list[str]) -> int:
    text = _candidate_text(candidate).lower()
    authority = str(
        candidate.get("suggested_authority_class")
        or candidate.get("task_class")
        or ""
    ).lower()
    priority = str(candidate.get("suggested_priority") or "").upper()
    score = 0
    if authority in {"propose", "plan", "read"}:
        score += 100
    if authority == "branch_write":
        score -= 30
    if authority == "forge_write":
        score -= 20
    if authority == "ci_trial":
        score -= 40
    score += {"P0": 40, "P1": 30, "P2": 20, "P3": 10}.get(priority, 0)
    for bias in priority_bias:
        if bias == "pm_governance" and any(
            term in text
            for term in (
                "pm",
                "governance",
                "issue-only",
                "backlog",
                "approval",
                "lifecycle",
                "operator",
            )
        ):
            score += 30
        if bias == "plugin_reliability" and any(
            term in text for term in ("plugin", "reliability", "wrapper")
        ):
            score += 20
        if bias == "read_only_observability" and any(
            term in text for term in ("read-only", "snapshot", "status", "evidence")
        ):
            score += 20
    if "issue-only pm backlog expansion model" in text:
        score += 40
    if "trading" in text or "broker" in text:
        score -= 1000
    return score


def _recommended_first_selection(
    candidates: list[dict[str, Any]],
    *,
    priority_bias: list[str],
) -> dict[str, Any] | None:
    available = [
        item
        for item in candidates
        if not item.get("blocked")
        and not item.get("ci_or_workflow_related")
        and item.get("candidate_id")
    ]
    if not available:
        return None
    ranked = sorted(
        available,
        key=lambda item: (
            -_score_candidate(item, priority_bias),
            str(item.get("candidate_id")),
        ),
    )
    first = ranked[0]
    return {
        "candidate_id": first.get("candidate_id"),
        "title": first.get("title"),
        "suggested_priority": first.get("suggested_priority"),
        "suggested_authority_class": first.get("suggested_authority_class"),
        "reason": (
            "Recommended first because it is PM/platform governance work, "
            "does not duplicate Issue #1, and stays proposal-only until a "
            "future exact approval."
        ),
        "requires_future_approval_before_issue_creation": True,
    }


def _future_issue_scope(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "proposed_title": candidate.get("title"),
        "proposed_body_summary": candidate.get("proposed_issue_body_summary"),
        "authority_class": "forge_write",
        "future_operation_type": "create_issue",
        "requires_exact_approval_token": True,
        "requires_exact_plan_hash": True,
        "requires_operator_confirmation": True,
        "approval_required": True,
        "executable_by_default": False,
    }


def build_backlog_selection_packet(
    *,
    backlog_expansion_proposal: dict[str, Any] | None = None,
    pm_status: dict[str, Any] | None = None,
    issue_lifecycle: dict[str, Any] | None = None,
    kanban_packet: dict[str, Any] | None = None,
    gitea_snapshot: dict[str, Any] | None = None,
    operator_preferences: dict[str, Any] | None = None,
    select_candidate_ids: list[str] | None = None,
    defer_candidate_ids: list[str] | None = None,
    reject_candidate_ids: list[str] | None = None,
    project_id: str | None = None,
    issue_index: int = DEFAULT_ISSUE_INDEX,
    expected_title: str = EXPECTED_PM_SEED_ISSUE_TITLE,
    created_at: str | None = None,
) -> dict[str, Any]:
    proposal = backlog_expansion_proposal or _fallback_backlog_proposal(
        project_id=project_id or "crypto_bot",
        issue_index=issue_index,
        expected_title=expected_title,
    )
    source_issue = _source_issue_from_inputs(
        backlog_expansion_proposal=proposal,
        issue_lifecycle=issue_lifecycle,
        gitea_snapshot=gitea_snapshot,
        issue_index=issue_index,
        expected_title=expected_title,
    )
    resolved_project_id = str(
        project_id
        or proposal.get("project_id")
        or (
            pm_status.get("project", {}).get("project_id")
            if isinstance(pm_status, dict)
            and isinstance(pm_status.get("project"), dict)
            else None
        )
        or (kanban_packet or {}).get("project_id")
        or "crypto_bot"
    )
    preference_select = _preference_ids(
        operator_preferences,
        "selected_candidate_ids",
    )
    preference_defer = _preference_ids(
        operator_preferences,
        "deferred_candidate_ids",
    )
    preference_reject = _preference_ids(
        operator_preferences,
        "rejected_candidate_ids",
    )
    selected_ids = preference_select | _as_id_set(select_candidate_ids)
    deferred_ids = preference_defer | _as_id_set(defer_candidate_ids)
    rejected_ids = preference_reject | _as_id_set(reject_candidate_ids)
    priority_bias = _priority_bias(operator_preferences)
    forbidden_topics = _forbidden_topics(operator_preferences)
    raw_candidates = [
        item
        for item in proposal.get("proposed_backlog_items") or []
        if isinstance(item, dict)
    ] + [
        item
        for item in proposal.get("blocked_candidates") or []
        if isinstance(item, dict)
    ]

    summaries: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for raw in raw_candidates:
        candidate_id = str(raw.get("candidate_id") or "")
        classification = _candidate_classification(
            raw,
            preferred_forbidden_topics=forbidden_topics,
        )
        if classification["blocked"]:
            status = "blocked"
        elif candidate_id in rejected_ids:
            status = "rejected"
        elif candidate_id in deferred_ids:
            status = "deferred"
        elif candidate_id in selected_ids:
            status = "selected_for_review"
        else:
            status = "review_pending"
        summary = _candidate_summary(
            raw,
            selection_status=status,
            classification=classification,
            source_issue_title=str(source_issue.get("title") or expected_title),
        )
        if summary["blocked"]:
            summary["selection_status"] = "blocked"
            blocked.append(summary)
        elif status == "selected_for_review":
            selected.append(summary)
        elif status == "deferred":
            deferred.append(summary)
        elif status == "rejected":
            rejected.append(summary)
        summaries.append(summary)

    recommended = _recommended_first_selection(
        summaries,
        priority_bias=priority_bias,
    )
    future_scopes = [_future_issue_scope(candidate) for candidate in selected]
    created = created_at or _utc_now()
    packet_seed = {
        "project_id": resolved_project_id,
        "source_backlog_proposal_id": proposal.get("proposal_id"),
        "source_issue_index": source_issue.get("issue_index") or issue_index,
        "candidate_ids": [item.get("candidate_id") for item in summaries],
        "selected_ids": [item.get("candidate_id") for item in selected],
        "deferred_ids": [item.get("candidate_id") for item in deferred],
        "rejected_ids": [item.get("candidate_id") for item in rejected],
        "blocked_ids": [item.get("candidate_id") for item in blocked],
    }
    packet_id = f"backlog-selection-{_sha256_payload(packet_seed)[:16]}"
    next_recommendation = (
        "If the Operator approves a future write checkpoint, generate a "
        "forge dry-run plan for exactly the selected candidate IDs and require "
        "the exact plan hash, operation ID, and approval token before any "
        "Gitea issue creation."
        if selected
        else (
            "Operator should review the recommendation and explicitly select, "
            "defer, or reject candidate IDs before any future issue-creation "
            "planning."
        )
    )
    return _redact(
        {
            "schema_version": BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
            "selection_packet_id": packet_id,
            "created_at": created,
            "project_id": resolved_project_id,
            "source_backlog_proposal_id": proposal.get("proposal_id"),
            "source_issue_index": source_issue.get("issue_index") or issue_index,
            "source_issue_title": source_issue.get("title") or expected_title,
            "candidate_count": len(
                [
                    item
                    for item in summaries
                    if item.get("selection_status") != "blocked"
                ]
            ),
            "candidate_summaries": summaries,
            "review_pending_candidates": [
                item
                for item in summaries
                if item.get("selection_status") == "review_pending"
            ],
            "selected_candidates": selected,
            "deferred_candidates": deferred,
            "rejected_candidates": rejected,
            "blocked_candidates": blocked,
            "recommended_first_selection": recommended,
            "recommended_future_write_scope": future_scopes,
            "approval_required_for_write": True,
            "calls_gitea_write_api": False,
            "mutation_executed": False,
            "operator_review_required": True,
            "next_checkpoint_recommendation": next_recommendation,
            "selection_inputs": {
                "selected_candidate_ids": sorted(selected_ids),
                "deferred_candidate_ids": sorted(deferred_ids),
                "rejected_candidate_ids": sorted(rejected_ids),
                "priority_bias": priority_bias,
                "forbidden_topics": forbidden_topics,
            },
            "write_guardrails": {
                "issue_1_recreated": False,
                "future_write_plan_executable_by_default": False,
                "requires_exact_candidate_id": True,
                "requires_exact_operation_id": True,
                "requires_exact_plan_hash": True,
                "requires_explicit_approval_token": True,
                "project_card_api_fallback": "issue_only_pm_fallback",
            },
            "non_action_booleans": dict(NON_ACTION_BOOLEANS),
        }
    )


def format_backlog_selection_text(packet: dict[str, Any]) -> str:
    selected = packet.get("selected_candidates")
    deferred = packet.get("deferred_candidates")
    rejected = packet.get("rejected_candidates")
    blocked = packet.get("blocked_candidates")
    recommended = (
        packet.get("recommended_first_selection")
        if isinstance(packet.get("recommended_first_selection"), dict)
        else {}
    )
    lines = [
        "Hermes PM backlog selection packet",
        f"Project: {packet.get('project_id') or '<unknown>'}",
        (
            "Source issue: "
            f"#{packet.get('source_issue_index') or '<unknown>'} "
            f"{packet.get('source_issue_title') or ''}"
        ),
        f"Candidates: {packet.get('candidate_count', 0)}",
        f"Selected: {len(selected) if isinstance(selected, list) else 0}",
        f"Deferred: {len(deferred) if isinstance(deferred, list) else 0}",
        f"Rejected: {len(rejected) if isinstance(rejected, list) else 0}",
        f"Blocked: {len(blocked) if isinstance(blocked, list) else 0}",
        (
            "Recommended first: "
            f"{recommended.get('candidate_id') or '<none>'} "
            f"{recommended.get('title') or ''}"
        ),
        "Gitea writes: no",
        "Mutation executed: no",
        "Approval required for future write: yes",
    ]
    return redact_text("\n".join(lines))
