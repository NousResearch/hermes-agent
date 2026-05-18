#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.hermes_pm.development_slice_packet import (
        DEVELOPMENT_SLICE_PACKET_SCHEMA_VERSION,
        build_development_slice_packet,
        format_development_slice_text,
    )
    from scripts.hermes_pm.development_workstream_packet import (
        NON_ACTION_BOOLEANS,
        build_development_workstream_packet,
        load_safe_doc_summaries,
        redact_text,
    )
    from scripts.hermes_pm.generate_development_workstream_packet import (
        _live_gitea_inputs,
    )
    from scripts.hermes_pm.issue_lifecycle_status import (
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
    )
    from scripts.hermes_pm.project_status import (
        RefusedPathError,
        build_project_status,
        ensure_safe_input_path,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from development_slice_packet import (  # type: ignore[no-redef]
        DEVELOPMENT_SLICE_PACKET_SCHEMA_VERSION,
        build_development_slice_packet,
        format_development_slice_text,
    )
    from development_workstream_packet import (  # type: ignore[no-redef]
        NON_ACTION_BOOLEANS,
        build_development_workstream_packet,
        load_safe_doc_summaries,
        redact_text,
    )
    from generate_development_workstream_packet import (  # type: ignore[no-redef]
        _live_gitea_inputs,
    )
    from issue_lifecycle_status import (  # type: ignore[no-redef]
        DEFAULT_ISSUE_INDEX,
        EXPECTED_PM_SEED_ISSUE_TITLE,
    )
    from project_status import (  # type: ignore[no-redef]
        RefusedPathError,
        build_project_status,
        ensure_safe_input_path,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_generate_development_slice_packet",
    "authority_class": "propose",
    "schema_version": DEVELOPMENT_SLICE_PACKET_SCHEMA_VERSION,
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


def build_packet_from_args(args: argparse.Namespace) -> dict[str, Any]:
    workstream = _load_json(
        args.development_workstream_packet,
        label="Development workstream packet path",
    )
    snapshot = None
    lifecycle = None
    if args.live_gitea_read:
        snapshot, lifecycle = _live_gitea_inputs(args)
    if workstream is None:
        pm_status = build_project_status(
            project_id=args.project_id,
            repo_root=args.repo_root,
            gitea_snapshot=snapshot,
            issue_lifecycle=lifecycle,
        )
        safe_docs = load_safe_doc_summaries(args.repo_root)
        workstream = build_development_workstream_packet(
            pm_status=pm_status,
            issue_lifecycle=lifecycle,
            gitea_snapshot=snapshot,
            safe_planning_docs=safe_docs,
            project_id=args.project_id,
            issue_index=args.issue_index,
            expected_title=args.expected_title,
        )
    return build_development_slice_packet(
        workstream_packet=workstream,
        candidate_id=args.candidate_id,
        project_id=args.project_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a non-mutating Hermes PM development slice packet for "
            "one candidate. It is implementation-ready planning evidence only "
            "and has no apply, write, branch-writer, issue-executor, workflow, "
            "runner, deploy, runtime, secret, broker, trading, or financial "
            "mode."
        )
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--project-id", default="crypto_bot")
    parser.add_argument("--candidate-id")
    parser.add_argument("--development-workstream-packet", type=Path)
    parser.add_argument(
        "--live-gitea-read",
        action="store_true",
        help="Capture live read-only Gitea and Issue #1 evidence with GET only.",
    )
    parser.add_argument("--issue-index", type=int, default=DEFAULT_ISSUE_INDEX)
    parser.add_argument("--expected-title", default=EXPECTED_PM_SEED_ISSUE_TITLE)
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
        print(format_development_slice_text(packet))
    else:
        print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
