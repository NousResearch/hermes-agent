#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from typing import Any

try:
    from scripts.hermes_pm.issue_lifecycle_status import (
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        EXPECTED_PM_SEED_ISSUE_TITLE,
        compact_lifecycle_summary,
        summarize_seed_issue_from_snapshot,
    )


WORK_STATE_SCHEMA_VERSION = "hermes.pm.work_state.v1"
STALE_AFTER_DAYS = 14

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


def _parse_dt(value: Any) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _project_id(pm_status: dict[str, Any] | None) -> str:
    project = (
        pm_status.get("project")
        if isinstance(pm_status, dict) and isinstance(pm_status.get("project"), dict)
        else {}
    )
    return str(project.get("project_id") or "crypto_bot")


def _repo_name(
    pm_status: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
) -> str:
    if isinstance(snapshot, dict):
        owner = snapshot.get("owner")
        repo = snapshot.get("repo")
        if owner and repo:
            return f"{owner}/{repo}"
    project = (
        pm_status.get("project")
        if isinstance(pm_status, dict) and isinstance(pm_status.get("project"), dict)
        else {}
    )
    return str(project.get("gitea_remote") or "preston/crypto_bot")


def _git(pm_status: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(pm_status, dict) and isinstance(pm_status.get("git"), dict):
        return pm_status["git"]
    return {}


def _issues(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(snapshot, dict) and isinstance(snapshot.get("issues"), dict):
        return snapshot["issues"]
    return {}


def _prs(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(snapshot, dict) and isinstance(snapshot.get("pull_requests"), dict):
        return snapshot["pull_requests"]
    return {}


def _checks(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(snapshot, dict) and isinstance(snapshot.get("checks"), dict):
        return snapshot["checks"]
    return {}


def _is_blocked(item: dict[str, Any]) -> bool:
    labels = [str(label).lower() for label in item.get("labels") or []]
    text = f"{item.get('title') or ''} {' '.join(labels)}".lower()
    return "blocked" in text or "waiting" in text or "approval" in text


def _is_untriaged(item: dict[str, Any]) -> bool:
    labels = [str(label).lower() for label in item.get("labels") or []]
    return not labels or "triage" in labels or "untriaged" in labels


def _compact_item(item: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "number": item.get("number"),
        "title": item.get("title"),
        "state": item.get("state"),
        "labels": item.get("labels") or [],
        "updated_at": item.get("updated_at"),
        "html_url": item.get("html_url"),
    }


def _stale_items(
    *,
    open_issues: list[dict[str, Any]],
    open_prs: list[dict[str, Any]],
    now: dt.datetime,
) -> list[dict[str, Any]]:
    cutoff = now - dt.timedelta(days=STALE_AFTER_DAYS)
    stale: list[dict[str, Any]] = []
    for kind, items in (("issue", open_issues), ("pull_request", open_prs)):
        for item in items:
            updated_at = _parse_dt(item.get("updated_at"))
            if updated_at and updated_at < cutoff:
                entry = _compact_item(item, kind)
                entry["stale_days"] = (now - updated_at).days
                stale.append(entry)
    return stale


def _recent_activity(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    repository = (
        snapshot.get("repository")
        if isinstance(snapshot.get("repository"), dict)
        else {}
    )
    commits = repository.get("recent_commits") or []
    prs = _prs(snapshot).get("recently_closed_or_merged") or []
    issues = _issues(snapshot).get("recently_closed") or []
    activity: list[dict[str, Any]] = []
    for item in commits[:5]:
        if isinstance(item, dict):
            activity.append(
                {
                    "kind": "commit",
                    "title": item.get("message"),
                    "sha": item.get("sha"),
                    "updated_at": item.get("author_date"),
                }
            )
    for item in prs[:3]:
        if isinstance(item, dict):
            activity.append(_compact_item(item, "pull_request"))
    for item in issues[:3]:
        if isinstance(item, dict):
            activity.append(_compact_item(item, "issue"))
    return activity[:10]


def _ci_status_summary(
    pm_status: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    statuses = [
        item
        for item in (_checks(snapshot).get("statuses") or [])
        if isinstance(item, dict)
    ]
    states: dict[str, int] = {}
    for status in statuses:
        state = str(status.get("state") or "unknown")
        states[state] = states.get(state, 0) + 1
    combined = _checks(snapshot).get("combined_status") or {}
    pm_ci = (
        pm_status.get("ci_locality_readiness")
        if isinstance(pm_status, dict)
        and isinstance(pm_status.get("ci_locality_readiness"), dict)
        else {}
    )
    return {
        "gitea_target_sha": _checks(snapshot).get("target_sha"),
        "gitea_status_counts": states,
        "gitea_combined_state": (
            combined.get("state") if isinstance(combined, dict) else None
        ),
        "local_ci_evidence_available": bool(pm_ci.get("available")),
        "evidence_ready_for_future_workflow_trial": bool(
            pm_ci.get("evidence_ready_for_future_workflow_trial")
        ),
        "workflow_runs_observed": (
            snapshot.get("workflows", {}).get("recent_run_count", 0)
            if isinstance(snapshot, dict)
            and isinstance(snapshot.get("workflows"), dict)
            else 0
        ),
    }


def _runner_status_summary(pm_status: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(pm_status, dict) and isinstance(
        pm_status.get("runner_readiness"),
        dict,
    ):
        runner = dict(pm_status["runner_readiness"])
    else:
        runner = {
            "registration": "unknown",
            "online": "unknown",
            "workflow_trial": "blocked",
        }
    runner["starts_runner"] = False
    runner["approval_required_before_start"] = True
    return runner


def build_work_state(
    *,
    pm_status: dict[str, Any] | None = None,
    gitea_snapshot: dict[str, Any] | None = None,
    issue_lifecycle: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    issues = _issues(gitea_snapshot)
    prs = _prs(gitea_snapshot)
    open_issues = [
        item for item in (issues.get("open") or []) if isinstance(item, dict)
    ]
    open_prs = [
        item for item in (prs.get("open") or []) if isinstance(item, dict)
    ]
    blocked = [
        _compact_item(item, "issue")
        for item in open_issues
        if _is_blocked(item)
    ]
    blocked.extend(
        _compact_item(item, "pull_request")
        for item in open_prs
        if _is_blocked(item)
    )
    untriaged = [
        _compact_item(item, "issue")
        for item in open_issues
        if _is_untriaged(item)
    ]
    git = _git(pm_status)
    project = (
        pm_status.get("project")
        if isinstance(pm_status, dict) and isinstance(pm_status.get("project"), dict)
        else {}
    )
    pm_blockers = (
        pm_status.get("known_blockers")
        if isinstance(pm_status, dict)
        and isinstance(pm_status.get("known_blockers"), list)
        else []
    )
    approval_gates = (
        pm_status.get("outstanding_approval_gates")
        if isinstance(pm_status, dict)
        and isinstance(pm_status.get("outstanding_approval_gates"), list)
        else []
    )
    seed_issue = (
        compact_lifecycle_summary(issue_lifecycle)
        if issue_lifecycle is not None
        else summarize_seed_issue_from_snapshot(gitea_snapshot)
    )
    seed_issue_exists = bool(seed_issue.get("exists"))
    duplicate_seed_issue_blocker = bool(seed_issue.get("duplicate_seed_issue_blocker"))
    recommended_next_actions = [
        "Generate or review the Kanban proposal packet before any forge mutation.",
        "Ask the Operator for scoped approval before creating or editing Gitea items.",
    ]
    if seed_issue_exists:
        recommended_next_actions.insert(
            0,
            (
                f"Review existing Issue #{seed_issue.get('issue_index') or 1} "
                "and plan proposal-only backlog expansion; do not recreate the "
                "PM seed issue."
            ),
        )
    else:
        recommended_next_actions.insert(
            0,
            (
                "Verify whether the initial PM-managed backlog issue exists "
                "before proposing any seed issue creation."
            ),
        )
    if untriaged:
        recommended_next_actions.insert(
            1 if seed_issue_exists else 0,
            "Triage unlabeled or triage-labeled open issues.",
        )
    if open_prs:
        recommended_next_actions.insert(0, "Review open PRs and requested evidence.")
    if pm_blockers or blocked:
        recommended_next_actions.insert(
            0,
            "Resolve blocked PM items or request explicit approvals.",
        )

    return {
        "schema_version": WORK_STATE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "project_id": _project_id(pm_status),
        "repo": _repo_name(pm_status, gitea_snapshot),
        "branch": git.get("branch") or "<unknown>",
        "dirty_state": "dirty" if git.get("dirty") else "clean",
        "open_issue_count": int(issues.get("open_count") or len(open_issues)),
        "open_pr_count": int(prs.get("open_count") or len(open_prs)),
        "seed_pm_issue": {
            **seed_issue,
            "expected_title": EXPECTED_PM_SEED_ISSUE_TITLE,
            "duplicate_seed_issue_blocker": duplicate_seed_issue_blocker,
        },
        "stale_items": _stale_items(
            open_issues=open_issues,
            open_prs=open_prs,
            now=now,
        ),
        "blocked_items": blocked,
        "untriaged_items": untriaged,
        "recent_activity": _recent_activity(gitea_snapshot),
        "approval_gates": [str(item) for item in approval_gates],
        "ci_status_summary": _ci_status_summary(pm_status, gitea_snapshot),
        "runner_status_summary": _runner_status_summary(pm_status),
        "forbidden_surfaces": [
            str(item) for item in project.get("forbidden_surfaces") or []
        ],
        "recommended_next_actions": recommended_next_actions,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
