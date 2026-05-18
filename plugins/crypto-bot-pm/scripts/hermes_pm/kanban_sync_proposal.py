#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

KANBAN_SYNC_SCHEMA_VERSION = "hermes.pm.kanban_sync_proposal.v1"

NON_ACTION_BOOLEANS = {
    "writes_files": False,
    "calls_gitea_write_api": False,
    "calls_gitea_read_api": False,
    "starts_runner": False,
    "runs_workflows": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "branch_writer_invoked": False,
}

DEFAULT_COLUMNS = [
    "Inbox",
    "Triage",
    "Ready for plan",
    "Proposal",
    "Awaiting approval",
    "In progress",
    "Evidence review",
    "Blocked",
    "Done",
]

DEFAULT_LABELS = [
    "hermes-pm",
    "managed-project",
    "read-only",
    "proposal-only",
    "approval-required",
    "forbidden-surface",
    "ci-evidence",
]


def _load_json(path: Path | None) -> Any:
    if path is None:
        return None
    if str(path) == "-":
        return json.loads(sys.stdin.read())
    return json.loads(path.read_text(encoding="utf-8"))


def _project_id(status: Any) -> str:
    if isinstance(status, dict):
        project = status.get("project")
        if isinstance(project, dict) and project.get("project_id"):
            return str(project["project_id"])
    return "crypto_bot"


def build_kanban_sync_proposal(
    *,
    project_status: Any = None,
    task_intake: Any = None,
) -> dict[str, Any]:
    project_id = _project_id(project_status)
    suggested_issue_titles = [
        "Review Hermes PM control-plane handover pack",
        "Approve initial Hermes PM Kanban columns and labels",
        "Triage managed-project backlog for crypto_bot without runtime action",
    ]
    suggested_next_cards = [
        {
            "column": "Triage",
            "title": "Separate Hermes PM platform tasks from crypto_bot daemon tasks",
            "labels": ["hermes-pm", "managed-project"],
        },
        {
            "column": "Evidence review",
            "title": "Review local CI evidence without starting runner or workflows",
            "labels": ["ci-evidence", "read-only"],
        },
        {
            "column": "Awaiting approval",
            "title": "Prepare approval request format for any future forge writes",
            "labels": ["approval-required", "proposal-only"],
        },
    ]
    if isinstance(project_status, dict):
        for blocker in (project_status.get("known_blockers") or [])[:4]:
            suggested_next_cards.append(
                {
                    "column": "Blocked",
                    "title": str(blocker),
                    "labels": ["blocked", "read-only"],
                }
            )
    if isinstance(task_intake, dict):
        title = task_intake.get("title") or task_intake.get("request")
        if title:
            suggested_issue_titles.append(f"Triage intake: {title}")
            suggested_next_cards.append(
                {
                    "column": "Inbox",
                    "title": str(title),
                    "labels": ["hermes-pm", "triage"],
                }
            )
    return {
        "schema_version": KANBAN_SYNC_SCHEMA_VERSION,
        "tool": "hermes_pm_kanban_sync_proposal",
        "project_id": project_id,
        "read_only": True,
        "proposal_only": True,
        "mutation_capability": False,
        "approval_required_for_write": True,
        "calls_gitea_write_api": False,
        "suggested_issue_titles": suggested_issue_titles,
        "suggested_labels": DEFAULT_LABELS,
        "suggested_project_columns": DEFAULT_COLUMNS,
        "suggested_next_cards": suggested_next_cards,
        "approval_notes": [
            "Creating issues, labels, project columns, cards, PR comments, "
            "or checks requires explicit operator approval.",
            "This proposal does not call Gitea APIs and has no apply mode.",
        ],
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_kanban_sync_proposal",
    "authority_class": "propose",
    "schema_version": KANBAN_SYNC_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a proposal-only Hermes Kanban/Gitea sync plan. The tool never "
            "calls Gitea write APIs and has no apply mode."
        )
    )
    parser.add_argument("--project-status", type=Path)
    parser.add_argument("--task-intake", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--describe-authority",
        action="store_true",
        help="Print this tool's Hermes PM authority metadata as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    indent = 2 if args.pretty else None
    if args.describe_authority:
        print(json.dumps(OPERATOR_AUTHORITY_METADATA, indent=indent, sort_keys=True))
        return 0
    proposal = build_kanban_sync_proposal(
        project_status=_load_json(args.project_status),
        task_intake=_load_json(args.task_intake),
    )
    print(json.dumps(proposal, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
