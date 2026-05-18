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
    from scripts.hermes_pm.forge_write_plan import (
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_forge_write_plan,
        format_forge_write_plan_text,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        capture_gitea_snapshot,
        redact_text,
    )
    from scripts.hermes_pm.kanban_proposal_packet import build_kanban_proposal_packet
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
    from forge_write_plan import (  # type: ignore[no-redef]
        FORGE_WRITE_PLAN_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        build_forge_write_plan,
        format_forge_write_plan_text,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        capture_gitea_snapshot,
        redact_text,
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
    "tool": "hermes_pm_generate_forge_write_plan",
    "authority_class": "propose",
    "schema_version": FORGE_WRITE_PLAN_SCHEMA_VERSION,
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


def _fallback_snapshot(
    *,
    base_url: str,
    owner: str,
    repo: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema_version": "hermes.pm.gitea_readonly_snapshot.v1",
        "created_at": None,
        "gitea_base_url": redact_text(base_url),
        "owner": owner,
        "repo": repo,
        "auth_used": False,
        "token_value_exposed": False,
        "http_methods_used": [],
        "issues": {"open": [], "open_count": 0},
        "pull_requests": {"open": [], "open_count": 0},
        "labels": [],
        "checks": {"statuses": [], "combined_status": {}},
        "workflows": {"recent_run_count": 0},
        "repository": {"recent_commits": []},
        "blockers": [{"endpoint": "<read-only-snapshot>", "error": error}],
        "warnings": [],
    }


def _snapshot_from_args(args: argparse.Namespace) -> dict[str, Any]:
    loaded = _load_json(args.gitea_snapshot, label="Gitea snapshot path")
    if loaded is not None:
        return loaded
    try:
        return capture_gitea_snapshot(
            base_url=args.base_url,
            owner=args.owner,
            repo=args.repo,
            timeout=3,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI path
        return _fallback_snapshot(
            base_url=args.base_url,
            owner=args.owner,
            repo=args.repo,
            error=redact_text(str(exc)),
        )


def _build_packet_from_status(
    *,
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    backlog_selection_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pm_status = build_project_status(
        project_id=args.project_id,
        repo_root=args.repo_root,
        gitea_snapshot=snapshot,
    )
    work_state = build_work_state(
        pm_status=pm_status,
        gitea_snapshot=snapshot,
    )
    backlog_proposal = build_backlog_expansion_proposal(
        pm_status=pm_status,
        gitea_snapshot=snapshot,
        work_state=work_state,
    )
    return build_kanban_proposal_packet(
        pm_status=pm_status,
        gitea_snapshot=snapshot,
        work_state=work_state,
        backlog_expansion_proposal=backlog_proposal,
        backlog_selection_packet=backlog_selection_packet,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dry-run Hermes PM forge-write plan from a Kanban "
            "proposal packet. This tool renders future Gitea endpoints and "
            "redacted payloads only; it never mutates Gitea."
        )
    )
    parser.add_argument("--kanban-packet", type=Path)
    parser.add_argument("--gitea-snapshot", type=Path)
    parser.add_argument("--backlog-selection-packet", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--project-id", default="crypto_bot")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--base-url", default=DEFAULT_GITEA_BASE_URL)
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
        packet = _load_json(args.kanban_packet, label="Kanban packet path")
        backlog_selection = _load_json(
            args.backlog_selection_packet,
            label="Backlog selection packet path",
        )
        snapshot = (
            _snapshot_from_args(args)
            if packet is None or args.gitea_snapshot is not None
            else None
        )
        if packet is None:
            if snapshot is None:
                snapshot = _snapshot_from_args(args)
            packet = _build_packet_from_status(
                args=args,
                snapshot=snapshot,
                backlog_selection_packet=backlog_selection,
            )
        plan = build_forge_write_plan(
            kanban_packet=packet,
            gitea_snapshot=snapshot,
            backlog_selection_packet=backlog_selection,
            project_id=args.project_id,
            gitea_base_url=args.base_url,
            owner=args.owner,
            repo=args.repo,
        )
    except (RefusedPathError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")
    if args.format == "text":
        print(format_forge_write_plan_text(plan))
    else:
        print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
