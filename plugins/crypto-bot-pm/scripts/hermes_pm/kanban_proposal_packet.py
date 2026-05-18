#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from typing import Any

try:
    from scripts.hermes_pm.issue_lifecycle_status import (
        EXPECTED_PM_SEED_ISSUE_TITLE,
        summarize_seed_issue_from_snapshot,
    )
    from scripts.hermes_pm.task_classifier import classify_task
    from scripts.hermes_pm.work_state import build_work_state
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        EXPECTED_PM_SEED_ISSUE_TITLE,
        summarize_seed_issue_from_snapshot,
    )
    from task_classifier import classify_task  # type: ignore[no-redef]
    from work_state import build_work_state  # type: ignore[no-redef]


KANBAN_PACKET_SCHEMA_VERSION = "hermes.pm.kanban_proposal_packet.v1"

PROPOSED_COLUMNS = [
    "Inbox",
    "Needs Triage",
    "Ready To Plan",
    "Waiting For Approval",
    "Ready For Implementation",
    "In Review",
    "Blocked",
    "Evidence Needed",
    "Done",
]

SUGGESTED_LABELS = [
    "hermes-pm",
    "managed-project",
    "read-only",
    "proposal-only",
    "approval-required",
    "blocked",
    "needs-triage",
    "evidence-needed",
    "pm-review",
]

CHECKPOINT_6_FIRST_WRITE_TITLE = EXPECTED_PM_SEED_ISSUE_TITLE
CHECKPOINT_6_FIRST_WRITE_BODY = (
    "Created by Hermes PM Checkpoint 6.\n\n"
    "This is the first approval-gated forge-write rehearsal.\n\n"
    "Scope is project-management metadata only.\n\n"
    "No labels, projects, comments, PRs, workflows, runners, deploys, "
    "runtime actions, branch-writer actions, financial actions, or secret "
    "access were approved.\n\n"
    "The Operator should review whether Hermes PM may continue toward "
    "proposal-only backlog expansion."
)

NON_ACTION_BOOLEANS = {
    "writes_files": False,
    "calls_gitea_write_api": False,
    "creates_issues": False,
    "creates_prs": False,
    "comments": False,
    "edits_labels": False,
    "starts_workflows": False,
    "starts_runner": False,
    "runs_workflows": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "branch_writer_invoked": False,
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _issues(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(snapshot, dict) and isinstance(snapshot.get("issues"), dict):
        return snapshot["issues"]
    return {}


def _prs(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(snapshot, dict) and isinstance(snapshot.get("pull_requests"), dict):
        return snapshot["pull_requests"]
    return {}


def _project_id(pm_status: dict[str, Any] | None, work_state: dict[str, Any]) -> str:
    if work_state.get("project_id"):
        return str(work_state["project_id"])
    project = (
        pm_status.get("project")
        if isinstance(pm_status, dict) and isinstance(pm_status.get("project"), dict)
        else {}
    )
    return str(project.get("project_id") or "crypto_bot")


def _labels(item: dict[str, Any]) -> list[str]:
    return [str(label) for label in item.get("labels") or []]


def _card(
    *,
    column: str,
    title: str,
    source: str,
    labels: list[str],
    number: Any = None,
    rationale: str = "",
    issue_body: str | None = None,
    issue_index: Any = None,
    issue_url: str | None = None,
    lifecycle_state: str | None = None,
) -> dict[str, Any]:
    classification = classify_task(f"Propose Kanban card: {title}")
    card = {
        "column": column,
        "title": title,
        "source": source,
        "source_number": number,
        "issue_index": issue_index,
        "issue_url": issue_url,
        "lifecycle_state": lifecycle_state,
        "labels": sorted(dict.fromkeys(labels)),
        "rationale": rationale,
        "task_class": classification.get("task_class"),
        "approval_required": bool(classification.get("approval_required")),
    }
    if issue_body is not None:
        card["issue_body"] = issue_body
    return card


def _snapshot_has_issue_title(
    snapshot: dict[str, Any] | None,
    title: str,
) -> bool:
    issues = _issues(snapshot)
    for key in ("open", "recently_closed", "closed", "all"):
        for issue in issues.get(key) or []:
            if isinstance(issue, dict) and issue.get("title") == title:
                return True
    return False


def _existing_seed_issue_ref(
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    return summarize_seed_issue_from_snapshot(
        snapshot,
        issue_index=1,
        expected_title=CHECKPOINT_6_FIRST_WRITE_TITLE,
    )


def _checkpoint_6_first_write_cards(
    snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if _snapshot_has_issue_title(snapshot, CHECKPOINT_6_FIRST_WRITE_TITLE):
        return []
    return [
        _card(
            column="Inbox",
            title=CHECKPOINT_6_FIRST_WRITE_TITLE,
            source="hermes_pm_checkpoint_6_first_write",
            labels=[],
            rationale=(
                "Propose PM backlog expansion review for the approved "
                "PM-6 one issue rehearsal."
            ),
            issue_body=CHECKPOINT_6_FIRST_WRITE_BODY,
        )
    ]


def _issue_cards(
    open_issues: list[dict[str, Any]],
    *,
    seed_issue_ref: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for issue in open_issues:
        labels = _labels(issue)
        lower_labels = {label.lower() for label in labels}
        title = str(issue.get("title") or f"Issue #{issue.get('number')}")
        is_seed_issue = (
            title == CHECKPOINT_6_FIRST_WRITE_TITLE
            and int(issue.get("number") or -1) == 1
        )
        if is_seed_issue:
            column = "Ready To Plan"
            extra = ["pm-review", "read-only"]
            source = "existing_gitea_issue"
            rationale = (
                "Issue #1 is durable PM work state from PM-6B; represent it "
                "instead of proposing duplicate issue creation."
            )
        elif not labels or "triage" in lower_labels or "untriaged" in lower_labels:
            column = "Needs Triage"
            extra = ["needs-triage"]
            source = "gitea_issue"
            rationale = "Read-only issue snapshot mapped to PM board proposal."
        elif "blocked" in lower_labels or "approval-required" in lower_labels:
            column = "Blocked"
            extra = ["blocked"]
            source = "gitea_issue"
            rationale = "Read-only issue snapshot mapped to PM board proposal."
        elif "evidence-needed" in lower_labels or "ci-evidence" in lower_labels:
            column = "Evidence Needed"
            extra = ["evidence-needed"]
            source = "gitea_issue"
            rationale = "Read-only issue snapshot mapped to PM board proposal."
        else:
            column = "Ready To Plan"
            extra = ["pm-review"]
            source = "gitea_issue"
            rationale = "Read-only issue snapshot mapped to PM board proposal."
        cards.append(
            _card(
                column=column,
                title=title,
                source=source,
                number=issue.get("number"),
                labels=["hermes-pm", "managed-project", *labels, *extra],
                rationale=rationale,
                issue_index=issue.get("number"),
                issue_url=issue.get("html_url")
                or (
                    seed_issue_ref.get("issue_url")
                    if is_seed_issue and isinstance(seed_issue_ref, dict)
                    else None
                ),
                lifecycle_state=(
                    seed_issue_ref.get("lifecycle_state")
                    if is_seed_issue and isinstance(seed_issue_ref, dict)
                    else None
                ),
            )
        )
    return cards


def _pr_cards(open_prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for pr in open_prs:
        title = str(pr.get("title") or f"PR #{pr.get('number')}")
        cards.append(
            _card(
                column="In Review",
                title=title,
                source="gitea_pull_request",
                number=pr.get("number"),
                labels=["hermes-pm", "pm-review", "evidence-needed"],
                rationale="Open PR requires PM evidence review and attention.",
            )
        )
    return cards


def _status_cards(pm_status: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(pm_status, dict):
        return []
    cards: list[dict[str, Any]] = []
    for blocker in (pm_status.get("known_blockers") or [])[:4]:
        cards.append(
            _card(
                column="Blocked",
                title=str(blocker),
                source="pm_status_blocker",
                labels=["hermes-pm", "blocked", "read-only"],
                rationale="PM status reports this as a current blocker.",
            )
        )
    return cards


def _backlog_candidate_cards(
    backlog_expansion_proposal: dict[str, Any] | None,
    backlog_selection_packet: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(backlog_expansion_proposal, dict):
        return []
    selection_by_id: dict[str, dict[str, Any]] = {}
    recommended_id = None
    if isinstance(backlog_selection_packet, dict):
        recommended = backlog_selection_packet.get("recommended_first_selection")
        if isinstance(recommended, dict):
            recommended_id = recommended.get("candidate_id")
        for key in (
            "selected_candidates",
            "deferred_candidates",
            "rejected_candidates",
            "blocked_candidates",
            "review_pending_candidates",
        ):
            for item in backlog_selection_packet.get(key) or []:
                if isinstance(item, dict) and item.get("candidate_id"):
                    selection_by_id[str(item["candidate_id"])] = item
    cards: list[dict[str, Any]] = []
    for item in backlog_expansion_proposal.get("proposed_backlog_items") or []:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        selection = selection_by_id.get(candidate_id, {})
        selection_status = str(
            selection.get("selection_status") or "proposal_only"
        )
        column = (
            "Waiting For Approval"
            if selection_status == "selected_for_review"
            else "Blocked"
            if selection_status == "rejected"
            else "Ready To Plan"
        )
        labels = [
            "hermes-pm",
            "managed-project",
            "proposal-only",
            "approval-required",
        ]
        authority = item.get("suggested_authority_class")
        if authority:
            labels.append(str(authority))
        card = _card(
            column=column,
            title=str(item.get("title") or "Untitled PM backlog candidate"),
            source="backlog_expansion_proposal",
            number=candidate_id,
            labels=labels,
            rationale=str(item.get("rationale") or ""),
            issue_body=str(item.get("proposed_issue_body_summary") or ""),
        )
        card["candidate_id"] = candidate_id
        card["proposal_only"] = True
        card["existing_gitea_issue"] = False
        card["future_create_issue_requires_approval"] = True
        card["suggested_priority"] = item.get("suggested_priority")
        card["suggested_authority_class"] = item.get("suggested_authority_class")
        card["suggested_next_step"] = item.get("suggested_next_step")
        card["selection_status"] = selection_status
        card["selected_for_review"] = selection_status == "selected_for_review"
        card["deferred"] = selection_status == "deferred"
        card["rejected"] = selection_status == "rejected"
        card["recommended_first_selection"] = candidate_id == recommended_id
        cards.append(card)
    return cards


def _suggested_issue_updates(open_issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for issue in open_issues:
        labels = _labels(issue)
        missing: list[str] = []
        if not labels:
            missing.extend(["needs-triage", "managed-project"])
        if "blocked" in str(issue.get("title") or "").lower():
            missing.append("blocked")
        if missing:
            updates.append(
                {
                    "issue_number": issue.get("number"),
                    "title": issue.get("title"),
                    "suggested_labels_to_add": sorted(dict.fromkeys(missing)),
                    "requires_gitea_write": True,
                    "approval_required": True,
                }
            )
    return updates


def _suggested_pr_attention(open_prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attention: list[dict[str, Any]] = []
    for pr in open_prs:
        attention.append(
            {
                "pr_number": pr.get("number"),
                "title": pr.get("title"),
                "head": pr.get("head"),
                "base": pr.get("base"),
                "suggested_attention": (
                    "Review evidence, checks, and requested approvals before "
                    "any merge or comment action."
                ),
                "requires_gitea_write": False,
            }
        )
    return attention


def _suggested_approvals(work_state: dict[str, Any]) -> list[dict[str, Any]]:
    approvals = [
        {
            "action": "apply_kanban_packet_to_gitea",
            "task_class": "forge_write",
            "approval_required": True,
            "reason": (
                "Creating or editing Gitea cards, labels, comments, or "
                "issues is a mutation."
            ),
        }
    ]
    if work_state.get("open_pr_count"):
        approvals.append(
            {
                "action": "comment_or_label_pr_attention_items",
                "task_class": "forge_write",
                "approval_required": True,
                "reason": (
                    "PR comments and labels require explicit forge-write "
                    "approval."
                ),
            }
        )
    return approvals


def build_kanban_proposal_packet(
    *,
    pm_status: dict[str, Any] | None = None,
    gitea_snapshot: dict[str, Any] | None = None,
    work_state: dict[str, Any] | None = None,
    backlog_expansion_proposal: dict[str, Any] | None = None,
    backlog_selection_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = work_state or build_work_state(
        pm_status=pm_status,
        gitea_snapshot=gitea_snapshot,
    )
    issues = _issues(gitea_snapshot)
    prs = _prs(gitea_snapshot)
    open_issues = [
        item for item in (issues.get("open") or []) if isinstance(item, dict)
    ]
    open_prs = [
        item for item in (prs.get("open") or []) if isinstance(item, dict)
    ]
    seed_issue_ref = _existing_seed_issue_ref(gitea_snapshot)
    proposed_cards = (
        _checkpoint_6_first_write_cards(gitea_snapshot)
        + _issue_cards(open_issues, seed_issue_ref=seed_issue_ref)
        + _backlog_candidate_cards(
            backlog_expansion_proposal,
            backlog_selection_packet=backlog_selection_packet,
        )
        + _pr_cards(open_prs)
        + _status_cards(pm_status)
    )
    if not proposed_cards:
        proposed_cards.append(
            _card(
                column="Inbox",
                title="Review Hermes PM status and Gitea snapshot",
                source="hermes_pm_default",
                labels=["hermes-pm", "read-only", "proposal-only"],
                rationale=(
                    "No open Gitea items were available; keep daily PM "
                    "review visible."
                ),
            )
        )
    project_id = _project_id(pm_status, state)
    blocked_items = state.get("blocked_items") or []
    daily_summary = {
        "project_id": project_id,
        "repo": state.get("repo"),
        "branch": state.get("branch"),
        "dirty_state": state.get("dirty_state"),
        "open_issue_count": state.get("open_issue_count"),
        "open_pr_count": state.get("open_pr_count"),
        "blocked_count": len(blocked_items),
        "untriaged_count": len(state.get("untriaged_items") or []),
        "backlog_candidate_count": len(
            backlog_expansion_proposal.get("proposed_backlog_items") or []
        )
        if isinstance(backlog_expansion_proposal, dict)
        else 0,
        "next_pm_action": (
            state.get("recommended_next_actions") or ["Review status."]
        )[0],
        "seed_pm_issue_exists": bool(seed_issue_ref.get("exists")),
        "duplicate_seed_issue_blocker": bool(
            seed_issue_ref.get("duplicate_seed_issue_blocker")
        ),
    }
    existing_issue_refs = []
    if seed_issue_ref.get("exists"):
        existing_issue_refs.append(
            {
                "issue_index": seed_issue_ref.get("issue_index"),
                "issue_url": seed_issue_ref.get("issue_url"),
                "title": seed_issue_ref.get("title"),
                "state": seed_issue_ref.get("state"),
                "lifecycle_state": seed_issue_ref.get("lifecycle_state"),
                "source": "existing_gitea_issue",
            }
        )
    return {
        "schema_version": KANBAN_PACKET_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "project_id": project_id,
        "proposed_columns": list(PROPOSED_COLUMNS),
        "proposed_cards": proposed_cards,
        "suggested_labels": list(SUGGESTED_LABELS),
        "suggested_issue_updates": _suggested_issue_updates(open_issues),
        "suggested_pr_attention": _suggested_pr_attention(open_prs),
        "suggested_approvals": _suggested_approvals(state),
        "existing_issue_refs": existing_issue_refs,
        "deduplication_summary": {
            "seed_issue_title": CHECKPOINT_6_FIRST_WRITE_TITLE,
            "seed_issue_exists": bool(seed_issue_ref.get("exists")),
            "seed_issue_index": seed_issue_ref.get("issue_index"),
            "seed_issue_url": seed_issue_ref.get("issue_url"),
            "seed_lifecycle_state": seed_issue_ref.get("lifecycle_state"),
            "duplicate_seed_issue_blocker": bool(
                seed_issue_ref.get("duplicate_seed_issue_blocker")
            ),
            "create_issue_card_for_seed_proposed": False
            if seed_issue_ref.get("exists")
            else bool(_checkpoint_6_first_write_cards(gitea_snapshot)),
            "recommended_next_actions": [
                "Operator review of existing Issue #1",
                "PM-8 proposal-only backlog expansion",
                "Future approval planning without mutation",
            ]
            if seed_issue_ref.get("exists")
            else ["Verify missing seed issue before any future write request."],
        },
        "backlog_expansion_summary": {
            "proposal_id": (
                backlog_expansion_proposal.get("proposal_id")
                if isinstance(backlog_expansion_proposal, dict)
                else None
            ),
            "proposed_candidate_count": len(
                backlog_expansion_proposal.get("proposed_backlog_items") or []
            )
            if isinstance(backlog_expansion_proposal, dict)
            else 0,
            "blocked_candidate_count": len(
                backlog_expansion_proposal.get("blocked_candidates") or []
            )
            if isinstance(backlog_expansion_proposal, dict)
            else 0,
            "proposal_only": True,
            "creates_gitea_issues_now": False,
        },
        "backlog_selection_summary": {
            "selection_packet_id": (
                backlog_selection_packet.get("selection_packet_id")
                if isinstance(backlog_selection_packet, dict)
                else None
            ),
            "selected_for_review_count": len(
                backlog_selection_packet.get("selected_candidates") or []
            )
            if isinstance(backlog_selection_packet, dict)
            else 0,
            "deferred_count": len(
                backlog_selection_packet.get("deferred_candidates") or []
            )
            if isinstance(backlog_selection_packet, dict)
            else 0,
            "rejected_count": len(
                backlog_selection_packet.get("rejected_candidates") or []
            )
            if isinstance(backlog_selection_packet, dict)
            else 0,
            "recommended_first_selection": (
                backlog_selection_packet.get("recommended_first_selection")
                if isinstance(backlog_selection_packet, dict)
                else None
            ),
            "proposal_only": True,
            "creates_gitea_issues_now": False,
        },
        "blocked_items": blocked_items,
        "daily_summary": daily_summary,
        "next_operator_questions": [
            (
                "Should Hermes apply any proposed Gitea changes in a future "
                "approved forge-write checkpoint?"
            ),
            (
                "Which blocked or untriaged items should be prioritized for "
                "the next PM cycle?"
            ),
            (
                "Is the next checkpoint still read-only, or should it request "
                "scoped forge-write approval?"
            ),
        ],
        "work_state": state,
        "calls_gitea_write_api": False,
        "mutation_required_for_application": True,
        "approval_required_for_write": True,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


def format_kanban_packet_text(packet: dict[str, Any]) -> str:
    summary = (
        packet.get("daily_summary")
        if isinstance(packet.get("daily_summary"), dict)
        else {}
    )
    lines = [
        "Hermes PM Kanban proposal packet",
        f"Project: {packet.get('project_id') or '<unknown>'}",
        f"Branch: {summary.get('branch') or '<unknown>'}",
        f"Worktree: {summary.get('dirty_state') or '<unknown>'}",
        f"Open issues: {summary.get('open_issue_count', 0)}",
        f"Open PRs: {summary.get('open_pr_count', 0)}",
        f"Blocked: {summary.get('blocked_count', 0)}",
        f"Untriaged: {summary.get('untriaged_count', 0)}",
        f"Proposed columns: {len(packet.get('proposed_columns') or [])}",
        f"Proposed cards: {len(packet.get('proposed_cards') or [])}",
        f"Suggested approvals: {len(packet.get('suggested_approvals') or [])}",
        f"Next: {summary.get('next_pm_action') or 'Review status.'}",
        "Gitea writes: no",
        "Approval required for write: yes",
    ]
    return "\n".join(lines)
