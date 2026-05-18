#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        NON_ACTION_BOOLEANS,
        SNAPSHOT_SCHEMA_VERSION,
        capture_gitea_snapshot,
        redact_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        NON_ACTION_BOOLEANS,
        SNAPSHOT_SCHEMA_VERSION,
        capture_gitea_snapshot,
        redact_text,
    )


def format_snapshot_text(snapshot: dict[str, Any]) -> str:
    repository = (
        snapshot.get("repository")
        if isinstance(snapshot.get("repository"), dict)
        else {}
    )
    issues = snapshot.get("issues") if isinstance(snapshot.get("issues"), dict) else {}
    prs = (
        snapshot.get("pull_requests")
        if isinstance(snapshot.get("pull_requests"), dict)
        else {}
    )
    checks = snapshot.get("checks") if isinstance(snapshot.get("checks"), dict) else {}
    workflows = (
        snapshot.get("workflows")
        if isinstance(snapshot.get("workflows"), dict)
        else {}
    )
    blockers = snapshot.get("blockers") or []
    warnings = snapshot.get("warnings") or []
    methods = ", ".join(snapshot.get("http_methods_used") or ["<none>"])
    lines = [
        "Hermes PM Gitea read-only snapshot",
        f"Repo: {snapshot.get('owner')}/{snapshot.get('repo')}",
        f"Base: {snapshot.get('gitea_base_url')}",
        f"Auth used: {'yes' if snapshot.get('auth_used') else 'no'}",
        f"Methods: {methods}",
        f"Default branch: {repository.get('default_branch') or '<unknown>'}",
        f"Open issues: {issues.get('open_count', 0)}",
        f"Open PRs: {prs.get('open_count', 0)}",
        f"Statuses: {len(checks.get('statuses') or [])}",
        f"Workflow runs: {workflows.get('recent_run_count', 0)}",
        f"Blockers: {len(blockers)}",
    ]
    for item in blockers[:4]:
        if isinstance(item, dict):
            lines.append(f"- {item.get('endpoint')}: {item.get('error')}")
        else:
            lines.append(f"- {item}")
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")
        for item in warnings[:3]:
            if isinstance(item, dict):
                lines.append(f"- {item.get('code')}: {item.get('message')}")
            else:
                lines.append(f"- {item}")
    lines.append("Gitea writes: no")
    return redact_text("\n".join(lines))


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_gitea_snapshot",
    "authority_class": "read",
    "schema_version": SNAPSHOT_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a read-only Hermes PM Gitea issue/PR/check snapshot. "
            "Only GET/HEAD methods are permitted; no mutation mode exists."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_GITEA_BASE_URL)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Ignore read token environment variables and use unauthenticated GETs.",
    )
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
    snapshot = capture_gitea_snapshot(
        base_url=args.base_url,
        owner=args.owner,
        repo=args.repo,
        timeout=args.timeout,
        no_token=args.no_token,
    )
    if args.format == "text":
        print(format_snapshot_text(snapshot))
    else:
        print(json.dumps(snapshot, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
