#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    from scripts.hermes_pm.gitea_forge_capability_map import (
        GITEA_FORGE_CAPABILITY_MAP_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        capture_gitea_forge_capability_map,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from gitea_forge_capability_map import (  # type: ignore[no-redef]
        GITEA_FORGE_CAPABILITY_MAP_SCHEMA_VERSION,
        NON_ACTION_BOOLEANS,
        capture_gitea_forge_capability_map,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        redact_text,
    )


OPERATOR_AUTHORITY_METADATA = {
    "tool": "hermes_pm_map_gitea_forge_capabilities",
    "authority_class": "read",
    "schema_version": GITEA_FORGE_CAPABILITY_MAP_SCHEMA_VERSION,
    "read_only": True,
    "mutation_capability": False,
    **NON_ACTION_BOOLEANS,
}


def format_capability_map_text(capability_map: dict[str, Any]) -> str:
    methods = ", ".join(capability_map.get("http_methods_used") or ["<none>"])
    ready = ", ".join(
        capability_map.get("endpoint_ready_operation_types") or ["<none>"]
    )
    blocked = ", ".join(
        capability_map.get("blocked_or_unknown_operation_types") or ["<none>"]
    )
    project_status = capability_map.get("project_endpoint_status")
    if not isinstance(project_status, dict):
        project_status = {}
    fallback = (
        "yes" if project_status.get("issue_only_fallback_recommended") else "no"
    )
    lines = [
        "Hermes PM Gitea forge capability map",
        f"Repo: {capability_map.get('owner')}/{capability_map.get('repo')}",
        f"Base: {capability_map.get('gitea_base_url')}",
        f"Auth used: {'yes' if capability_map.get('auth_used') else 'no'}",
        f"Methods: {methods}",
        f"Endpoint-ready operations: {ready}",
        f"Blocked/unknown operations: {blocked}",
        (
            "Repo projects status: "
            f"{project_status.get('repo_projects_status_code')}"
        ),
        f"Issue-only fallback: {fallback}",
        f"Blockers: {len(capability_map.get('blockers') or [])}",
        "Permission proof: no",
        "Gitea writes performed: no",
    ]
    return redact_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only Hermes PM Gitea forge capability map using "
            "GET/HEAD/OPTIONS-only client enforcement. The default probe uses "
            "GET requests and never calls Gitea write APIs."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_GITEA_BASE_URL)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Ignore read-token environment variables and use unauthenticated GETs.",
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
    capability_map = capture_gitea_forge_capability_map(
        base_url=args.base_url,
        owner=args.owner,
        repo=args.repo,
        timeout=args.timeout,
        no_token=args.no_token,
        limit=args.limit,
    )
    if args.format == "text":
        print(format_capability_map_text(capability_map))
    else:
        print(json.dumps(capability_map, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
