#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.hermes_pm.generate_selected_candidate_forge_plan import (
        build_scope_from_args,
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
    )
    from scripts.hermes_pm.project_status import (
        RefusedPathError,
        ensure_safe_input_path,
    )
    from scripts.hermes_pm.selected_candidate_approval_packet import (
        EXPLICIT_NON_ACTIONS,
        SELECTED_CANDIDATE_APPROVAL_PACKET_SCHEMA_VERSION,
        build_selected_candidate_approval_packet,
        format_selected_candidate_approval_packet_text,
    )
    from scripts.hermes_pm.selected_candidate_execution_payload import (
        approval_scope_from_execution_payload,
        load_selected_candidate_execution_payload,
    )
    from scripts.hermes_pm.selected_candidate_forge_plan import (
        build_selected_candidate_forge_plan,
        build_selected_candidate_forge_plan_from_execution_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from generate_selected_candidate_forge_plan import (  # type: ignore[no-redef]
        build_scope_from_args,
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
    )
    from project_status import (  # type: ignore[no-redef]
        RefusedPathError,
        ensure_safe_input_path,
    )
    from selected_candidate_approval_packet import (  # type: ignore[no-redef]
        EXPLICIT_NON_ACTIONS,
        SELECTED_CANDIDATE_APPROVAL_PACKET_SCHEMA_VERSION,
        build_selected_candidate_approval_packet,
        format_selected_candidate_approval_packet_text,
    )
    from selected_candidate_execution_payload import (  # type: ignore[no-redef]
        approval_scope_from_execution_payload,
        load_selected_candidate_execution_payload,
    )
    from selected_candidate_forge_plan import (  # type: ignore[no-redef]
        build_selected_candidate_forge_plan,
        build_selected_candidate_forge_plan_from_execution_payload,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_generate_selected_candidate_approval_packet",
    "authority_class": "propose",
    "schema_version": SELECTED_CANDIDATE_APPROVAL_PACKET_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    "calls_gitea_write_api": False,
    **EXPLICIT_NON_ACTIONS,
}


def _load_json(path: Path | None, *, label: str) -> dict | None:
    if path is None:
        return None
    ensure_safe_input_path(path, label=label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PM-10 selected-candidate approval packet for future "
            "PM-11 issue creation review. This tool never mutates Gitea and "
            "does not sign or execute an approval token."
        )
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--project-id", default="crypto_bot")
    parser.add_argument("--candidate-id", default="pm8-002")
    parser.add_argument("--approval-scope", type=Path)
    parser.add_argument("--execution-payload", type=Path)
    parser.add_argument("--forge-write-plan", type=Path)
    parser.add_argument("--backlog-selection-packet", type=Path)
    parser.add_argument("--operator-preferences", type=Path)
    parser.add_argument("--live-gitea-read", action="store_true")
    parser.add_argument("--issue-index", type=int, default=DEFAULT_ISSUE_INDEX)
    parser.add_argument("--expected-title", default=EXPECTED_PM_SEED_ISSUE_TITLE)
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
        frozen_payload_path = None
        if args.execution_payload is not None:
            ensure_safe_input_path(
                args.execution_payload,
                label="Execution payload path",
            )
            execution_payload = load_selected_candidate_execution_payload(
                args.execution_payload
            )
            frozen_payload_path = str(args.execution_payload)
            scope = approval_scope_from_execution_payload(execution_payload)
            plan = build_selected_candidate_forge_plan_from_execution_payload(
                execution_payload=execution_payload,
                frozen_payload_path=frozen_payload_path,
            )
            if args.live_gitea_read:
                plan["live_gitea_read_requested"] = True
                plan["live_gitea_read_mode"] = (
                    "skipped_for_frozen_execution_payload"
                )
                plan["live_gitea_read_reason"] = (
                    "Frozen execution payload mode avoids token lookup and "
                    "keeps live snapshot metadata out of the approval identity."
                )
        else:
            scope = _load_json(args.approval_scope, label="Approval scope path")
            if scope is None:
                scope = build_scope_from_args(args)
            plan = _load_json(args.forge_write_plan, label="Forge write plan path")
            if plan is None:
                plan = build_selected_candidate_forge_plan(approval_scope=scope)
        packet = build_selected_candidate_approval_packet(
            approval_scope=scope,
            forge_write_plan=plan,
            frozen_payload_path=frozen_payload_path,
        )
    except (RefusedPathError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {redact_text(str(exc))}\n")
    if args.format == "text":
        print(format_selected_candidate_approval_packet_text(packet))
    else:
        print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
