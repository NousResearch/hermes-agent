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
        BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_backlog_selection_packet,
        format_backlog_selection_text,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        capture_gitea_snapshot,
        redact_text,
    )
    from scripts.hermes_pm.issue_lifecycle_status import (
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
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
        BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_backlog_selection_packet,
        format_backlog_selection_text,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        capture_gitea_snapshot,
        redact_text,
    )
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
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
    "tool": "hermes_pm_generate_backlog_selection_packet",
    "authority_class": "propose",
    "schema_version": BACKLOG_SELECTION_PACKET_SCHEMA_VERSION,
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


def _build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    proposal = _load_json(args.backlog_proposal, label="Backlog proposal path")
    preferences = _load_json(
        args.operator_preferences,
        label="Operator preferences path",
    )
    snapshot = None
    lifecycle = None
    if args.live_gitea_read:
        snapshot = capture_gitea_snapshot()
        lifecycle = capture_issue_lifecycle_status(
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
    pm_status = build_project_status(
        project_id=args.project_id,
        repo_root=args.repo_root,
        gitea_snapshot=snapshot,
        issue_lifecycle=lifecycle,
    )
    work_state = build_work_state(
        pm_status=pm_status,
        gitea_snapshot=snapshot,
        issue_lifecycle=lifecycle,
    )
    if proposal is None:
        base_kanban = build_kanban_proposal_packet(
            pm_status=pm_status,
            gitea_snapshot=snapshot,
            work_state=work_state,
        )
        proposal = build_backlog_expansion_proposal(
            pm_status=pm_status,
            gitea_snapshot=snapshot,
            issue_lifecycle=lifecycle,
            work_state=work_state,
            kanban_packet=base_kanban,
            project_id=args.project_id,
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
    kanban = build_kanban_proposal_packet(
        pm_status=pm_status,
        gitea_snapshot=snapshot,
        work_state=work_state,
        backlog_expansion_proposal=proposal,
    )
    return {
        "proposal": proposal,
        "preferences": preferences,
        "snapshot": snapshot,
        "lifecycle": lifecycle,
        "pm_status": pm_status,
        "kanban": kanban,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a read-only Hermes PM backlog selection packet from "
            "proposal-only backlog candidates. The tool never mutates Gitea "
            "and has no apply, execute, or write mode."
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
    parser.add_argument("--backlog-proposal", type=Path)
    parser.add_argument(
        "--select-candidate",
        action="append",
        default=[],
        metavar="CANDIDATE_ID",
    )
    parser.add_argument(
        "--defer-candidate",
        action="append",
        default=[],
        metavar="CANDIDATE_ID",
    )
    parser.add_argument(
        "--reject-candidate",
        action="append",
        default=[],
        metavar="CANDIDATE_ID",
    )
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
        inputs = _build_inputs(args)
        packet = build_backlog_selection_packet(
            backlog_expansion_proposal=inputs["proposal"],
            pm_status=inputs["pm_status"],
            issue_lifecycle=inputs["lifecycle"],
            kanban_packet=inputs["kanban"],
            gitea_snapshot=inputs["snapshot"],
            operator_preferences=inputs["preferences"],
            select_candidate_ids=args.select_candidate,
            defer_candidate_ids=args.defer_candidate,
            reject_candidate_ids=args.reject_candidate,
            project_id=args.project_id,
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
    except (RefusedPathError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")
    if args.format == "text":
        print(format_backlog_selection_text(packet))
    else:
        print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
