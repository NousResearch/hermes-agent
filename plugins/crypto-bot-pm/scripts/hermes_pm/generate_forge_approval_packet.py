#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.forge_approval_packet import (
        EXPLICIT_NON_ACTIONS,
        FORGE_APPROVAL_PACKET_SCHEMA_VERSION,
        build_forge_approval_packet,
        format_forge_approval_packet_text,
    )
    from scripts.hermes_pm.forge_write_plan import build_forge_write_plan
    from scripts.hermes_pm.generate_forge_write_plan import (
        _build_packet_from_status,
        _load_json,
        _snapshot_from_args,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
    from scripts.hermes_pm.project_status import RefusedPathError
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from forge_approval_packet import (  # type: ignore[no-redef]
        EXPLICIT_NON_ACTIONS,
        FORGE_APPROVAL_PACKET_SCHEMA_VERSION,
        build_forge_approval_packet,
        format_forge_approval_packet_text,
    )
    from forge_write_plan import build_forge_write_plan  # type: ignore[no-redef]
    from generate_forge_write_plan import (  # type: ignore[no-redef]
        _build_packet_from_status,
        _load_json,
        _snapshot_from_args,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
    from project_status import RefusedPathError  # type: ignore[no-redef]


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_generate_forge_approval_packet",
    "authority_class": "propose",
    "schema_version": FORGE_APPROVAL_PACKET_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    "calls_gitea_write_api": False,
    **EXPLICIT_NON_ACTIONS,
}


def _build_plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.forge_write_plan, label="Forge write plan path")
    if plan is not None:
        return plan
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
    return build_forge_write_plan(
        kanban_packet=packet,
        gitea_snapshot=snapshot,
        backlog_selection_packet=backlog_selection,
        project_id=args.project_id,
        gitea_base_url=args.base_url,
        owner=args.owner,
        repo=args.repo,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reviewable Hermes PM forge approval packet from a "
            "dry-run forge-write plan. This tool records review scope only "
            "and never executes a Gitea mutation."
        )
    )
    parser.add_argument("--forge-write-plan", type=Path)
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
        plan = _build_plan_from_args(args)
        packet = build_forge_approval_packet(forge_write_plan=plan)
    except (RefusedPathError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")
    if args.format == "text":
        print(format_forge_approval_packet_text(packet))
    else:
        print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
