#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.backlog_expansion_proposal import (
        build_backlog_expansion_proposal,
    )
    from scripts.hermes_pm.backlog_selection_packet import (
        build_backlog_selection_packet,
    )
    from scripts.hermes_pm.development_workstream_packet import (
        DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_development_workstream_packet,
        format_development_workstream_text,
        load_safe_doc_summaries,
        redact_text,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        NON_ACTION_BOOLEANS as GITEA_NON_ACTION_BOOLEANS,
        SNAPSHOT_SCHEMA_VERSION,
        capture_gitea_snapshot,
    )
    from scripts.hermes_pm.issue_lifecycle_status import (
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        ISSUE_LIFECYCLE_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS as ISSUE_LIFECYCLE_NON_ACTION_BOOLEANS,
        capture_issue_lifecycle_status,
    )
    from scripts.hermes_pm.kanban_proposal_packet import (
        build_kanban_proposal_packet,
    )
    from scripts.hermes_pm.project_status import (
        RefusedPathError,
        build_project_status,
        ensure_safe_input_path,
    )
    from scripts.hermes_pm.work_state import build_work_state
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from backlog_expansion_proposal import (  # type: ignore[no-redef]
        build_backlog_expansion_proposal,
    )
    from backlog_selection_packet import (  # type: ignore[no-redef]
        build_backlog_selection_packet,
    )
    from development_workstream_packet import (  # type: ignore[no-redef]
        DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_development_workstream_packet,
        format_development_workstream_text,
        load_safe_doc_summaries,
        redact_text,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        NON_ACTION_BOOLEANS as GITEA_NON_ACTION_BOOLEANS,
        SNAPSHOT_SCHEMA_VERSION,
        capture_gitea_snapshot,
    )
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
        ISSUE_LIFECYCLE_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS as ISSUE_LIFECYCLE_NON_ACTION_BOOLEANS,
        capture_issue_lifecycle_status,
    )
    from kanban_proposal_packet import (  # type: ignore[no-redef]
        build_kanban_proposal_packet,
    )
    from project_status import (  # type: ignore[no-redef]
        RefusedPathError,
        build_project_status,
        ensure_safe_input_path,
    )
    from work_state import build_work_state  # type: ignore[no-redef]


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_generate_development_workstream_packet",
    "authority_class": "propose",
    "schema_version": DEVELOPMENT_WORKSTREAM_PACKET_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    "calls_gitea_write_api": False,
    **NON_ACTION_BOOLEANS,
}


def _load_json(path: Path | None, *, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    ensure_safe_input_path(path, label=label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object.")
    return payload


def _fallback_snapshot(error: str) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": None,
        "gitea_base_url": DEFAULT_GITEA_BASE_URL,
        "owner": DEFAULT_OWNER,
        "repo": DEFAULT_REPO,
        "auth_used": False,
        "token_value_exposed": False,
        "http_methods_used": [],
        "repository": {"recent_commits": []},
        "issues": {"open": [], "open_count": 0, "recently_closed_count": 0},
        "pull_requests": {
            "open": [],
            "open_count": 0,
            "recently_closed_or_merged_count": 0,
        },
        "labels": [],
        "milestones": [],
        "checks": {"statuses": [], "combined_status": {}},
        "workflows": {"recent_run_count": 0},
        "projects": [],
        "blockers": [
            {
                "endpoint": "<live-gitea-read>",
                "error": f"Live Gitea snapshot read failed: {redact_text(error)}",
            }
        ],
        "warnings": [],
        "non_action_booleans": dict(GITEA_NON_ACTION_BOOLEANS),
    }


def _fallback_issue_lifecycle(
    *,
    error: str,
    issue_index: int,
    expected_title: str,
) -> dict[str, Any]:
    return {
        "schema_version": ISSUE_LIFECYCLE_SCHEMA_VERSION,
        "observed_at": None,
        "gitea_base_url": DEFAULT_GITEA_BASE_URL,
        "owner": DEFAULT_OWNER,
        "repo": DEFAULT_REPO,
        "auth_used": False,
        "auth_env_var_used": None,
        "token_value_exposed": False,
        "http_methods_used": [],
        "issue_index": issue_index,
        "issue_url": None,
        "title": expected_title,
        "expected_title": expected_title,
        "state": None,
        "labels": [],
        "unexpected_labels": [],
        "milestone": None,
        "assignees": [],
        "is_pull_request": False,
        "body_marker_present": False,
        "matches_expected_pm_issue": False,
        "comments_summary": {"checked": False, "state": "unknown"},
        "project_summary": {"checked": False, "state": "unknown"},
        "workflow_summary": {"checked": False, "state": "unknown"},
        "runner_summary": {
            "checked": False,
            "state": "unknown",
            "online": "unknown",
            "safely_knowable": False,
            "starts_runners": False,
        },
        "lifecycle_state": "unknown",
        "no_mutation_observed": True,
        "blockers": [
            f"Live Gitea issue lifecycle read failed: {redact_text(error)}"
        ],
        "warnings": [],
        "non_action_booleans": dict(ISSUE_LIFECYCLE_NON_ACTION_BOOLEANS),
    }


def _live_gitea_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        snapshot = capture_gitea_snapshot()
    except Exception as exc:  # pragma: no cover - defensive CLI path
        snapshot = _fallback_snapshot(str(exc))
    try:
        lifecycle = capture_issue_lifecycle_status(
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI path
        lifecycle = _fallback_issue_lifecycle(
            error=str(exc),
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
    return snapshot, lifecycle


def build_packet_from_args(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = _load_json(args.gitea_snapshot, label="Gitea snapshot path")
    lifecycle = _load_json(args.issue_lifecycle, label="Issue lifecycle path")
    preferences = _load_json(
        args.operator_preferences,
        label="Operator preferences path",
    )
    backlog_selection = _load_json(
        args.backlog_selection_packet,
        label="Backlog selection packet path",
    )
    if args.live_gitea_read:
        snapshot, lifecycle = _live_gitea_inputs(args)
    pm_status = build_project_status(
        project_id=args.project_id,
        repo_root=args.repo_root,
        gitea_snapshot=snapshot,
        issue_lifecycle=lifecycle,
    )
    if backlog_selection is None:
        work_state = build_work_state(
            pm_status=pm_status,
            gitea_snapshot=snapshot,
            issue_lifecycle=lifecycle,
        )
        kanban = build_kanban_proposal_packet(
            pm_status=pm_status,
            gitea_snapshot=snapshot,
            work_state=work_state,
        )
        backlog = build_backlog_expansion_proposal(
            pm_status=pm_status,
            gitea_snapshot=snapshot,
            issue_lifecycle=lifecycle,
            work_state=work_state,
            kanban_packet=kanban,
            project_id=args.project_id,
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
        backlog_selection = build_backlog_selection_packet(
            backlog_expansion_proposal=backlog,
            pm_status=pm_status,
            issue_lifecycle=lifecycle,
            kanban_packet=kanban,
            gitea_snapshot=snapshot,
            operator_preferences=preferences,
            select_candidate_ids=["pm8-002"],
            project_id=args.project_id,
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
    safe_docs = load_safe_doc_summaries(args.repo_root)
    repo_docs = [
        item
        for item in safe_docs
        if item.get("path") == "docs/architecture/hermes_operator_repo_intelligence.md"
    ]
    checklist = {
        "path": "docs/implementation/hermes_operator_checklist.md",
        "present": any(
            item.get("path") == "docs/implementation/hermes_operator_checklist.md"
            for item in safe_docs
        ),
    }
    return build_development_workstream_packet(
        pm_status=pm_status,
        issue_lifecycle=lifecycle,
        gitea_snapshot=snapshot,
        repo_intelligence_docs=repo_docs,
        hermes_operator_checklist=checklist,
        pm_backlog_candidates=backlog_selection,
        safe_planning_docs=safe_docs,
        operator_preferences=preferences,
        project_id=args.project_id,
        issue_index=args.issue_index,
        expected_title=args.expected_title,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a read-only Hermes PM development workstream packet. "
            "The tool proposes completion-oriented candidates and never "
            "mutates Gitea, runs workflows, starts runners, invokes branch "
            "writers, invokes issue executors, deploys, touches runtime "
            "surfaces, accesses secrets, or performs financial actions."
        )
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--project-id", default="crypto_bot")
    parser.add_argument(
        "--live-gitea-read",
        action="store_true",
        help="Capture live read-only Gitea and Issue #1 evidence with GET only.",
    )
    parser.add_argument("--issue-index", type=int, default=DEFAULT_ISSUE_INDEX)
    parser.add_argument("--expected-title", default=EXPECTED_PM_SEED_ISSUE_TITLE)
    parser.add_argument("--gitea-snapshot", type=Path)
    parser.add_argument("--issue-lifecycle", type=Path)
    parser.add_argument("--backlog-selection-packet", type=Path)
    parser.add_argument("--operator-preferences", type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--describe-authority",
        action="store_true",
        help="Print this tool's Hermes PM authority metadata as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe_authority:
        print(json.dumps(OPERATOR_AUTHORITY_METADATA, sort_keys=True))
        return 0
    try:
        packet = build_packet_from_args(args)
    except (RefusedPathError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")
    if args.format == "text":
        print(format_development_workstream_text(packet))
    else:
        print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
